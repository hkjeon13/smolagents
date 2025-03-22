import inspect
import textwrap
import time
from typing import Any, Dict, List, Optional, Union, AsyncGenerator, Awaitable, TypeVar, Callable
import os
import json
import yaml
import jinja2
from pathlib import Path
import importlib
from huggingface_hub import create_repo, metadata_update, snapshot_download, upload_folder
from jinja2 import StrictUndefined, Template
from rich.console import Group
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text
import asyncio
from .agent_types import AgentType, handle_agent_output_types, AgentImage, AgentAudio
from .remote_executors import DockerExecutor, E2BExecutor
from .local_python_executor import BASE_BUILTIN_MODULES, PythonExecutor, fix_final_answer_code
from .async_local_python_executor import AsyncLocalPythonExecutor
from .agents import MultiStepAgent, PromptTemplates, EMPTY_PROMPT_TEMPLATES, populate_template, truncate_content
from .async_default_tools import AsyncFinalAnswerTool, ASYNC_TOOL_MAPPING
from .async_monitoring import (
    LogLevel,
    AsyncMonitor,
    AsyncAgentLogger
)
from .async_tools import AsyncTool
from .memory import ActionStep, SystemPromptStep, TaskStep, PlanningStep, AgentMemory, ToolCall
from .models import (
    ChatMessage,
    MessageRole,
)
from .utils import (
    AgentError,
    AgentParsingError,
    AgentGenerationError,
    AgentMaxStepsError,
    AgentExecutionError,
    make_init_file_async,
    async_write_file,
)
from .async_monitoring import YELLOW_HEX
from .utils import parse_code_blobs


R = TypeVar("R")

class AsyncMultiStepAgentBase:
    def __init__(
            self,
            tools: List[AsyncTool],
            model: Callable[[List[Dict[str, str]]], Awaitable[ChatMessage]],
            prompt_templates: Optional[PromptTemplates] = None,
            max_steps: int = 20,
            add_base_tools: bool = False,
            verbosity_level: LogLevel = LogLevel.INFO,
            grammar: Optional[Dict[str, str]] = None,
            managed_agents: Optional[List] = None,
            step_callbacks: Optional[List[Callable]] = None,
            planning_interval: Optional[int] = None,
            name: Optional[str] = None,
            description: Optional[str] = None,
            provide_run_summary: bool = False,
            final_answer_checks: Optional[List[Callable]] = None,
    ) -> None:
        raise NotImplementedError

    async def run(
            self,
            task: str,
            stream: bool,
            reset: bool,
            images: Optional[List[str]] = None,
            additional_args: Optional[Dict] = None,
            max_steps: Optional[int] = None
    ):
        raise NotImplementedError

    async def _run(
            self,
            task: str,
            stream: bool,
            images: Optional[List[str]] = None,
    ) -> AsyncGenerator[ActionStep | AgentType, None, None]:
        raise NotImplementedError

    async def _execute_step(self, task: str, memory_step: ActionStep) -> Union[None, Any]:
        raise NotImplementedError

    async def _finalize_step(self, memory_step: ActionStep, step_start_time: float):
        raise NotImplementedError

    async def _handle_max_steps_reached(self, task: str, images: List[str], step_start_time: float) -> Any:
        raise NotImplementedError

    async def planning_step(self, task, is_first_step: bool, step: int) -> None:
        raise NotImplementedError

    async def _generate_initial_plan(
            self,
            task: str
    ) -> tuple[list[dict[str, list[dict[str, str]] | MessageRole]], ChatMessage, ChatMessage]:
        raise NotImplementedError

    async def _generate_updated_plan(
            self,
            task: str,
            step: int
    ) -> tuple[list[dict[str, str] | dict[str, list[dict[str, str]] | MessageRole]], ChatMessage, ChatMessage]:
        raise NotImplementedError

    async def _record_planning_step(
            self,
            input_messages: list[dict[str, list[dict[str, str]] | MessageRole]],
            facts_message: ChatMessage,
            plan_message: ChatMessage,
            is_first_step: bool
    ) -> None:
        raise NotImplementedError

    async def provide_final_answer(self, task: str, images: Optional[list[str]]) -> str:
        raise NotImplementedError

    async def execute_tool_call(self, tool_name: str, arguments: Union[Dict[str, str], str]) -> Any:
        raise NotImplementedError

    async def step(self, memory_step: ActionStep) -> Union[None, Any]:
        raise NotImplementedError

    async def save(self, output_dir: str, relative_path: Optional[str] = None):
        raise NotImplementedError

    async def __call__(task: str, **kwargs):
        raise NotImplementedError

    async def _setup_tools(self):
        raise NotImplementedError

class AsyncMultiStepAgent(AsyncMultiStepAgentBase, MultiStepAgent):
    def __init__(
            self,
            tools: List[AsyncTool],
            model: Callable[[List[Dict[str, str]]], Awaitable[ChatMessage]],
            prompt_templates: Optional[PromptTemplates] = None,
            max_steps: int = 20,
            add_base_tools: bool = False,
            verbosity_level: LogLevel = LogLevel.INFO,
            grammar: Optional[Dict[str, str]] = None,
            managed_agents: Optional[List] = None,
            step_callbacks: Optional[List[Callable]] = None,
            planning_interval: Optional[int] = None,
            name: Optional[str] = None,
            description: Optional[str] = None,
            provide_run_summary: bool = False,
            final_answer_checks: Optional[List[Callable]] = None,
    ) -> None:
        self.agent_name = self.__class__.__name__
        self.model = model
        self.prompt_templates = prompt_templates or EMPTY_PROMPT_TEMPLATES
        self.max_steps = max_steps
        self.step_number = 0
        self.grammar = grammar
        self.planning_interval = planning_interval
        self.state = {}
        self.name = name
        self.description = description
        self.provide_run_summary = provide_run_summary
        self.final_answer_checks = final_answer_checks

        self._setup_managed_agents(managed_agents)
        self._setup_tools(tools, add_base_tools)
        self._validate_tools_and_managed_agents(tools, managed_agents)

        self.system_prompt: str = self.initialize_system_prompt()
        self.input_messages = None
        self.task = None
        self.memory = AgentMemory(self.system_prompt)
        self.logger = AsyncAgentLogger(level=verbosity_level)
        self.monitor = AsyncMonitor(self.model, self.logger)
        self.step_callbacks = step_callbacks if step_callbacks is not None else []
        self.step_callbacks.append(self.monitor.update_metrics)

    def _setup_tools(self, tools, add_base_tools):
        assert all(isinstance(tool, AsyncTool) for tool in tools), "All elements must be instance of AsyncTool (or a subclass)"
        self.tools: Dict[str, AsyncTool] = {tool.name: tool for tool in tools}
        if add_base_tools:
            self.tools.update(
                {
                    name: cls()
                    for name, cls in ASYNC_TOOL_MAPPING.items()
                    if name != "python_interpreter" or self.__class__.__name__ == "AsyncToolCallingAgent"
                }
            )

        self.tools.setdefault("final_answer", AsyncFinalAnswerTool())

    async def run(
            self,
            task: str,
            stream: bool,
            reset: bool,
            images: Optional[List[str]] = None,
            additional_args: Optional[Dict] = None,
            max_steps: Optional[int] = None
    ):
        max_steps = max_steps or self.max_steps
        self.task = task
        if additional_args is not None:
            self.state.update(additional_args)
            self.task += f"""
        You have been provided with these additional arguments, that you can access using the keys as variables in your python code:
        {str(additional_args)}."""

        self.system_prompt = self.initialize_system_prompt()
        self.memory.system_prompt = SystemPromptStep(system_prompt=self.system_prompt)
        if reset:
            self.memory.reset()
            self.monitor.reset()

        await self.logger.log_task(
            content=self.task.strip(),
            subtitle=f"{type(self.model).__name__} - {(self.model.model_id if hasattr(self.model, 'model_id') else '')}",
            level=LogLevel.INFO,
            title=self.name if hasattr(self, "name") else None,
        )
        self.memory.steps.append(TaskStep(task=self.task, task_images=images))

        if getattr(self, "python_executor", None):
            self.python_executor.send_variables(variables=self.state)
            self.python_executor.send_tools({**self.tools, **self.managed_agents})

        if stream:
            # The steps are returned as they are executed through a generator to iterate on.
            return await self._run(task=self.task, max_steps=max_steps, images=images)
        # Outputs are returned only at the end. We only look at the last step.
        results = [step async for step in self._run(task=self.task, max_steps=max_steps, images=images)]
        return results[-1] if results else None

    async def _run(
            self, task: str, max_steps: int, images: List[str] | None = None
    ) -> AsyncGenerator[ActionStep | AgentType, None, None]:
        final_answer = None
        self.step_number = 1
        step_start_time = time.time()
        memory_step = None
        while final_answer is None and self.step_number <= max_steps:
            step_start_time = time.time()
            memory_step = self._create_memory_step(step_start_time, images)
            try:
                final_answer = await self._execute_step(task, memory_step)
            except AgentGenerationError as e:
                # Agent generation errors are not caused by a Model error but an implementation error: so we should raise them and exit.
                raise e
            except AgentError as e:
                # Other AgentError types are caused by the Model, so we should log them and iterate.
                memory_step.error = e
            finally:
                await self._finalize_step(memory_step, step_start_time)
                yield memory_step
                self.step_number += 1

        if final_answer is None and self.step_number == max_steps + 1:
            final_answer = await self._handle_max_steps_reached(task, images, step_start_time)
            yield memory_step
        yield handle_agent_output_types(final_answer)

    async def _execute_step(self, task: str, memory_step: ActionStep) -> Union[None, Any]:
        if self.planning_interval is not None and self.step_number % self.planning_interval == 1:
            await self.planning_step(task, is_first_step=(self.step_number == 1), step=self.step_number)
        await self.logger.log_rule(f"Step {self.step_number}", level=LogLevel.INFO)
        final_answer = await self.step(memory_step)
        if final_answer is not None and self.final_answer_checks:
            self._validate_final_answer(final_answer)
        return final_answer

    async def _finalize_step(self, memory_step: ActionStep, step_start_time: float):
        memory_step.end_time = time.time()
        memory_step.duration = memory_step.end_time - step_start_time
        self.memory.steps.append(memory_step)
        for callback in self.step_callbacks:
            # For compatibility with old callbacks that don't take the agent as an argument
            if len(inspect.signature(callback).parameters) == 1:
                await callback(memory_step)
            else:
                await callback(
                    memory_step, agent=self
                )

    async def _handle_max_steps_reached(self, task: str, images: List[str], step_start_time: float) -> Any:
        final_answer = self.provide_final_answer(task, images)
        final_memory_step = ActionStep(
            step_number=self.step_number, error=AgentMaxStepsError("Reached max steps.", self.logger)
        )
        final_memory_step.action_output = final_answer
        final_memory_step.end_time = time.time()
        final_memory_step.duration = final_memory_step.end_time - step_start_time
        self.memory.steps.append(final_memory_step)
        for callback in self.step_callbacks:
            if len(inspect.signature(callback).parameters) == 1:
                await callback(final_memory_step)
            else:
                await callback(final_memory_step, agent=self)

        return final_answer

    async def planning_step(self, task, is_first_step: bool, step: int) -> None:
        if is_first_step:
            input_messages, facts_message, plan_message = await self._generate_initial_plan(task)
        else:
            input_messages, facts_message, plan_message = await self._generate_updated_plan(task, step)

        await self._record_planning_step(input_messages, facts_message, plan_message, is_first_step)

    async def _record_planning_step(
            self,
            input_messages: list[dict[str, list[dict[str, str]] | MessageRole]],
            facts_message: ChatMessage,
            plan_message: ChatMessage,
            is_first_step: bool
    ) -> None:
        if is_first_step:
            facts = textwrap.dedent(f"""Here are the facts that I know so far:\n```\n{facts_message.content}\n```""")
            plan = textwrap.dedent(
                f"""Here is the plan of action that I will follow to solve the task:\n```\n{plan_message.content}\n```"""
            )
            log_message = "Initial plan"
        else:
            facts = textwrap.dedent(
                f"""Here is the updated list of the facts that I know:\n```\n{facts_message.content}\n```"""
            )
            plan = textwrap.dedent(
                f"""I still need to solve the task I was given:\n```\n{self.task}\n```\n\nHere is my new/updated plan of action to solve the task:\n```\n{plan_message.content}\n```"""
            )
            log_message = "Updated plan"
        self.memory.steps.append(
            PlanningStep(
                model_input_messages=input_messages,
                facts=facts,
                plan=plan,
                model_output_message_plan=plan_message,
                model_output_message_facts=facts_message,
            )
        )
        await self.logger.log(Rule(f"[bold]{log_message}", style="orange"), Text(plan), level=LogLevel.INFO)

    async def provide_final_answer(self, task: str, images: Optional[list[str]]) -> str:
        """
        Provide the final answer to the task, based on the logs of the agent's interactions.

        Args:
            task (`str`): Task to perform.
            images (`list[str]`, *optional*): Paths to image(s).

        Returns:
            `str`: Final answer to the task.
        """
        messages = [
            {
                "role": MessageRole.SYSTEM,
                "content": [
                    {
                        "type": "text",
                        "text": self.prompt_templates["final_answer"]["pre_messages"],
                    }
                ],
            }
        ]
        if images:
            messages[0]["content"].append({"type": "image"})
        messages += self.write_memory_to_messages()[1:]
        messages += [
            {
                "role": MessageRole.USER,
                "content": [
                    {
                        "type": "text",
                        "text": populate_template(
                            self.prompt_templates["final_answer"]["post_messages"], variables={"task": task}
                        ),
                    }
                ],
            }
        ]
        try:
            chat_message: ChatMessage = await self.model(messages)
            return chat_message.content
        except Exception as e:
            return f"Error in generating final LLM output:\n{e}"

    async def execute_tool_call(self, tool_name: str, arguments: Union[Dict[str, str], str]) -> Any:
        """
        Execute tool with the provided input and returns the result.
        This method replaces arguments with the actual values from the state if they refer to state variables.

        Args:
            tool_name (`str`): Name of the Tool to execute (should be one from self.tools).
            arguments (Dict[str, str]): Arguments passed to the Tool.
        """
        available_tools = {**self.tools, **self.managed_agents}
        if tool_name not in available_tools:
            error_msg = f"Unknown tool {tool_name}, should be instead one of {list(available_tools.keys())}."
            raise AgentExecutionError(error_msg, self.logger)

        try:
            if isinstance(arguments, str):
                if tool_name in self.managed_agents:
                    observation = await available_tools[tool_name].__call__(arguments)
                else:
                    observation = await available_tools[tool_name].__call__(arguments, sanitize_inputs_outputs=True)
            elif isinstance(arguments, dict):
                for key, value in arguments.items():
                    if isinstance(value, str) and value in self.state:
                        arguments[key] = self.state[value]
                if tool_name in self.managed_agents:
                    observation = await available_tools[tool_name].__call__(**arguments)
                else:
                    observation = await available_tools[tool_name].__call__(**arguments, sanitize_inputs_outputs=True)
            else:
                error_msg = f"Arguments passed to tool should be a dict or string: got a {type(arguments)}."
                raise AgentExecutionError(error_msg, self.logger)
            return observation

        except Exception as e:
            if tool_name in self.tools:
                tool = self.tools[tool_name]
                error_msg = (
                    f"Error when executing tool {tool_name} with arguments {arguments}: {type(e).__name__}: {e}\nYou should only use this tool with a correct input.\n"
                    f"As a reminder, this tool's description is the following: '{tool.description}'.\nIt takes inputs: {tool.inputs} and returns output type {tool.output_type}"
                )
                raise AgentExecutionError(error_msg, self.logger)
            elif tool_name in self.managed_agents:
                error_msg = (
                    f"Error in calling team member: {e}\nYou should only ask this team member with a correct request.\n"
                    f"As a reminder, this team member's description is the following:\n{available_tools[tool_name]}"
                )
                raise AgentExecutionError(error_msg, self.logger)

    async def _generate_initial_plan(
            self,
            task: str
    ) -> tuple[list[dict[str, list[dict[str, str]] | MessageRole]], ChatMessage, ChatMessage]:
        input_messages = [
            {
                "role": MessageRole.USER,
                "content": [
                    {
                        "type": "text",
                        "text": populate_template(
                            self.prompt_templates["planning"]["initial_facts"], variables={"task": task}
                        ),
                    }
                ],
            },
        ]
        facts_message = await self.model(input_messages)

        message_prompt_plan = {
            "role": MessageRole.USER,
            "content": [
                {
                    "type": "text",
                    "text": populate_template(
                        self.prompt_templates["planning"]["initial_plan"],
                        variables={
                            "task": task,
                            "tools": self.tools,
                            "managed_agents": self.managed_agents,
                            "answer_facts": facts_message.content,
                        },
                    ),
                }
            ],
        }
        plan_message = await self.model([message_prompt_plan], stop_sequences=["<end_plan>"])
        return input_messages, facts_message, plan_message

    async def _generate_updated_plan(
            self,
            task: str,
            step: int
    ) -> tuple[list[dict[str, str] | dict[str, list[dict[str, str]] | MessageRole]], ChatMessage, ChatMessage]:
        # Do not take the system prompt message from the memory
        # summary_mode=False: Do not take previous plan steps to avoid influencing the new plan
        memory_messages = self.write_memory_to_messages()[1:]
        facts_update_pre = {
            "role": MessageRole.SYSTEM,
            "content": [{"type": "text", "text": self.prompt_templates["planning"]["update_facts_pre_messages"]}],
        }
        facts_update_post = {
            "role": MessageRole.USER,
            "content": [{"type": "text", "text": self.prompt_templates["planning"]["update_facts_post_messages"]}],
        }
        input_messages = [facts_update_pre] + memory_messages + [facts_update_post]
        facts_message = await self.model(input_messages)

        update_plan_pre = {
            "role": MessageRole.SYSTEM,
            "content": [
                {
                    "type": "text",
                    "text": populate_template(
                        self.prompt_templates["planning"]["update_plan_pre_messages"], variables={"task": task}
                    ),
                }
            ],
        }
        update_plan_post = {
            "role": MessageRole.USER,
            "content": [
                {
                    "type": "text",
                    "text": populate_template(
                        self.prompt_templates["planning"]["update_plan_post_messages"],
                        variables={
                            "task": task,
                            "tools": self.tools,
                            "managed_agents": self.managed_agents,
                            "facts_update": facts_message.content,
                            "remaining_steps": (self.max_steps - step),
                        },
                    ),
                }
            ],
        }
        plan_message = await self.model(
            [update_plan_pre] + memory_messages + [update_plan_post], stop_sequences=["<end_plan>"]
        )
        return input_messages, facts_message, plan_message

    async def save(self, output_dir: str, relative_path: Optional[str] = None):
        """
        에이전트의 관련 코드 파일들을 저장합니다.
        - tools 폴더: 각 도구의 코드를 별도 .py 파일로 저장
        - managed_agents 폴더: 관리하는 에이전트들의 코드를 저장
        - prompts.yaml: 프롬프트 템플릿
        - agent.json: 에이전트 설정 정보를 담은 JSON 파일
        - requirements.txt: 필요한 모듈 목록
        - app.py: Space에서 UI를 제공하는 Gradio 앱 코드
        """
        # output_dir에 __init__.py 생성
        await make_init_file_async(output_dir)

        # 관리하는 에이전트가 있다면 재귀적으로 저장 (각각 비동기로 저장)
        if self.managed_agents:
            managed_agents_dir = os.path.join(output_dir, "managed_agents")
            await make_init_file_async(managed_agents_dir)
            for agent_name, agent in self.managed_agents.items():
                agent_suffix = f"managed_agents.{agent_name}"
                if relative_path:
                    agent_suffix = relative_path + "." + agent_suffix
                # 각 managed agent의 save 메서드가 비동기로 구현되어 있다고 가정합니다.
                await agent.save(os.path.join(managed_agents_dir, agent_name), relative_path=agent_suffix)

        class_name = self.__class__.__name__

        # tools 폴더에 각 도구들을 저장 (각 도구의 save 메서드가 비동기로 구현되어 있다고 가정)
        tools_dir = os.path.join(output_dir, "tools")
        await make_init_file_async(tools_dir)
        for tool in self.tools.values():
            await tool.save(tools_dir, tool_file_name=tool.name, make_gradio_app=False)

        # 프롬프트 템플릿을 YAML 형식으로 저장
        yaml_prompts = yaml.safe_dump(
            self.prompt_templates,
            default_style="|",  # 모든 문자열을 블록 리터럴로 표시
            default_flow_style=False,
            width=float("inf"),
            sort_keys=False,
            allow_unicode=True,
            indent=2,
        )
        prompts_path = os.path.join(output_dir, "prompts.yaml")
        await async_write_file(prompts_path, yaml_prompts)

        # 에이전트 정보를 담은 JSON 파일 저장
        agent_dict = self.to_dict()
        agent_dict["tools"] = [tool.name for tool in self.tools.values()]
        agent_json = json.dumps(agent_dict, indent=4)
        agent_json_path = os.path.join(output_dir, "agent.json")
        await async_write_file(agent_json_path, agent_json)

        # requirements.txt 저장
        requirements_text = "".join(f"{r}\n" for r in agent_dict["requirements"])
        requirements_path = os.path.join(output_dir, "requirements.txt")
        await async_write_file(requirements_path, requirements_text)

        # app.py 파일 생성 (Gradio UI 제공)
        agent_name = f"agent_{self.name}" if getattr(self, "name", None) else "agent"
        managed_agent_relative_path = (relative_path + ".") if relative_path is not None else ""
        app_template = textwrap.dedent("""
            import yaml
            import os
            from smolagents import GradioUI, {{ class_name }}, {{ agent_dict['model']['class'] }}

            # 현재 디렉토리 경로
            CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

            {% for tool in tools.values() -%}
            from {{ managed_agent_relative_path }}tools.{{ tool.name }} import {{ tool.__class__.__name__ }} as {{ tool.name | camelcase }}
            {% endfor %}
            {% for managed_agent in managed_agents.values() -%}
            from {{ managed_agent_relative_path }}managed_agents.{{ managed_agent.name }}.app import agent_{{ managed_agent.name }}
            {% endfor %}

            model = {{ agent_dict['model']['class'] }}(
            {% for key in agent_dict['model']['data'] if key not in ['class', 'last_input_token_count', 'last_output_token_count'] -%}
                {{ key }}={{ agent_dict['model']['data'][key]|repr }},
            {% endfor %})

            {% for tool in tools.values() -%}
            {{ tool.name }} = {{ tool.name | camelcase }}()
            {% endfor %}

            with open(os.path.join(CURRENT_DIR, "prompts.yaml"), 'r') as stream:
                prompt_templates = yaml.safe_load(stream)

            {{ agent_name }} = {{ class_name }}(
                model=model,
                tools=[{% for tool_name in tools.keys() if tool_name != "final_answer" %}{{ tool_name }}{% if not loop.last %}, {% endif %}{% endfor %}],
                managed_agents=[{% for subagent_name in managed_agents.keys() %}agent_{{ subagent_name }}{% if not loop.last %}, {% endif %}{% endfor %}],
                {% for attribute_name, value in agent_dict.items() if attribute_name not in ["model", "tools", "prompt_templates", "authorized_imports", "managed_agents", "requirements"] -%}
                {{ attribute_name }}={{ value|repr }},
                {% endfor %}prompt_templates=prompt_templates
            )
            if __name__ == "__main__":
                GradioUI({{ agent_name }}).launch()
        """).strip()

        template_env = jinja2.Environment(loader=jinja2.BaseLoader(), undefined=jinja2.StrictUndefined)
        template_env.filters["repr"] = repr
        template_env.filters["camelcase"] = lambda value: "".join(word.capitalize() for word in value.split("_"))
        template = template_env.from_string(app_template)
        app_text = template.render(
            {
                "agent_name": agent_name,
                "class_name": class_name,
                "agent_dict": agent_dict,
                "tools": self.tools,
                "managed_agents": self.managed_agents,
                "managed_agent_relative_path": managed_agent_relative_path,
            }
        )
        app_path = os.path.join(output_dir, "app.py")
        await async_write_file(app_path, app_text + "\n")

    @classmethod
    async def from_hub(
            cls,
            repo_id: str,
            token: Optional[str] = None,
            trust_remote_code: bool = False,
            **kwargs,
    ):
        """
        Loads an agent defined on the Hub asynchronously.

        Loading an agent from the Hub means that you'll download the tool and execute it locally.
        ALWAYS inspect the tool you're downloading before loading it within your runtime, as you would do when
        installing a package using pip/npm/apt.

        Args:
            repo_id (str): The name of the repo on the Hub where your tool is defined.
            token (str, optional): The token to identify you on hf.co.
            trust_remote_code (bool, optional, defaults to False): Acknowledge the risk of running remote code.
            **kwargs: Additional keyword arguments; arguments relevant to the Hub (such as `cache_dir`, `revision`, `subfolder`)
                      will be used when downloading the files for your agent, and others will be passed along to its init.
        """
        if not trust_remote_code:
            raise ValueError(
                "Loading an agent from Hub requires to acknowledge you trust its code: to do so, pass `trust_remote_code=True`."
            )

        download_kwargs = {"token": token, "repo_type": "space"} | {
            key: kwargs.pop(key)
            for key in ["cache_dir", "force_download", "proxies", "revision", "local_files_only"]
            if key in kwargs
        }

        # snapshot_download 호출을 별도 스레드에서 실행합니다.
        download_folder = Path(
            await asyncio.to_thread(snapshot_download, repo_id=repo_id, **download_kwargs)
        )
        return await cls.from_folder(download_folder, **kwargs)

    @classmethod
    async def from_folder(cls, folder: Union[str, Path], **kwargs):
        """로컬 폴더에서 에이전트를 로드합니다.

        Args:
            folder (str 또는 Path): 에이전트가 저장된 폴더.
            **kwargs: 에이전트 초기화 시 전달할 추가 키워드 인자.
        """
        folder = Path(folder)
        # agent.json 파일을 비동기적으로 읽음
        agent_json_path = folder / "agent.json"
        agent_json_text = await asyncio.to_thread(agent_json_path.read_text)
        agent_dict = json.loads(agent_json_text)

        # 재귀적으로 managed_agents 로딩
        managed_agents = []
        for managed_agent_name, managed_agent_class in agent_dict.get("managed_agents", {}).items():
            agent_module = importlib.import_module("smolagents.async_agents")
            agent_cls = getattr(agent_module, managed_agent_class)
            managed_agent = await agent_cls.from_folder(folder / "managed_agents" / managed_agent_name)
            managed_agents.append(managed_agent)

        tools = []
        for tool_name in agent_dict.get("tools", []):
            tool_path = folder / "tools" / f"{tool_name}.py"
            tool_code = await asyncio.to_thread(tool_path.read_text)
            tools.append(AsyncTool.from_code(tool_code))

        model_module = importlib.import_module("smolagents.async_models")
        model_class = getattr(model_module, agent_dict["model"]["class"])
        model = await asyncio.to_thread(model_class.from_dict, agent_dict["model"]["data"])

        args = dict(
            model=model,
            tools=tools,
            managed_agents=managed_agents,
            name=agent_dict["name"],
            description=agent_dict["description"],
            max_steps=agent_dict["max_steps"],
            planning_interval=agent_dict["planning_interval"],
            grammar=agent_dict["grammar"],
            verbosity_level=agent_dict["verbosity_level"],
            prompt_templates=agent_dict["prompt_templates"],
        )

        if cls.__name__ == "AsyncCodeAgent":
            args["additional_authorized_imports"] = agent_dict["authorized_imports"]
            args["executor_type"] = agent_dict.get("executor_type")
            args["executor_kwargs"] = agent_dict.get("executor_kwargs")
            args["max_print_outputs_length"] = agent_dict.get("max_print_outputs_length")
        args.update(kwargs)
        return cls(**args)


class ToolCallingAgent(AsyncMultiStepAgent):
    def __init__(
            self,
            tools: List[AsyncTool],
            model: Callable[[List[Dict[str, str]]], Awaitable[ChatMessage]],
            prompt_templates: Optional[PromptTemplates] = None,
            max_steps: int = 20,
            add_base_tools: bool = False,
            verbosity_level: LogLevel = LogLevel.INFO,
            grammar: Optional[Dict[str, str]] = None,
            managed_agents: Optional[List] = None,
            step_callbacks: Optional[List[Callable]] = None,
            planning_interval: Optional[int] = None,
            name: Optional[str] = None,
            description: Optional[str] = None,
            provide_run_summary: bool = False,
            final_answer_checks: Optional[List[Callable]] = None,
    ) -> None:
        prompt_templates = prompt_templates or yaml.safe_load(
            importlib.resources.files("smolagents.prompts").joinpath("toolcalling_agent.yaml").read_text()
        )
        super().__init__(
            tools=tools,
            model=model,
            prompt_templates=prompt_templates,
            max_steps=max_steps,
            add_base_tools=add_base_tools,
            verbosity_level=verbosity_level,
            grammar=grammar,
            managed_agents=managed_agents,
            step_callbacks=step_callbacks,
            planning_interval=planning_interval,
            name=name,
            description=description,
            provide_run_summary=provide_run_summary,
            final_answer_checks=final_answer_checks,
        )


    def initialize_system_prompt(self) -> str:
        system_prompt = populate_template(
            self.prompt_templates["system_prompt"],
            variables={"tools": self.tools, "managed_agents": self.managed_agents},
        )
        return system_prompt

    async def step(self, memory_step: ActionStep) -> Union[None, Any]:
        """
        Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        Returns None if the step is not final.
        """
        memory_messages = self.write_memory_to_messages()

        self.input_messages = memory_messages

        # Add new step in logs
        memory_step.model_input_messages = memory_messages.copy()

        try:
            model_message: ChatMessage = await self.model(
                memory_messages,
                tools_to_call_from=list(self.tools.values()),
                stop_sequences=["Observation:"],
            )
            memory_step.model_output_message = model_message
        except Exception as e:
            raise AgentGenerationError(f"Error in generating tool call with model:\n{e}", self.logger) from e

        await self.logger.log_markdown(
            content=model_message.content if model_message.content else str(model_message.raw),
            title="Output message of the LLM:",
            level=LogLevel.DEBUG,
        )

        if model_message.tool_calls is None or len(model_message.tool_calls) == 0:
            raise AgentParsingError(
                "Model did not call any tools. Call `final_answer` tool to return a final answer.", self.logger
            )

        tool_call = model_message.tool_calls[0]
        tool_name, tool_call_id = tool_call.function.name, tool_call.id
        tool_arguments = tool_call.function.arguments

        memory_step.tool_calls = [ToolCall(name=tool_name, arguments=tool_arguments, id=tool_call_id)]

        # Execute
        await self.logger.log(
            Panel(Text(f"Calling tool: '{tool_name}' with arguments: {tool_arguments}")),
            level=LogLevel.INFO,
        )
        if tool_name == "final_answer":
            if isinstance(tool_arguments, dict):
                if "answer" in tool_arguments:
                    answer = tool_arguments["answer"]
                else:
                    answer = tool_arguments
            else:
                answer = tool_arguments
            if (
                isinstance(answer, str) and answer in self.state.keys()
            ):  # if the answer is a state variable, return the value
                final_answer = self.state[answer]
                await self.logger.log(
                    f"[bold {YELLOW_HEX}]Final answer:[/bold {YELLOW_HEX}] Extracting key '{answer}' from state to return value '{final_answer}'.",
                    level=LogLevel.INFO,
                )
            else:
                final_answer = answer
                await self.logger.log(
                    Text(f"Final answer: {final_answer}", style=f"bold {YELLOW_HEX}"),
                    level=LogLevel.INFO,
                )

            memory_step.action_output = final_answer
            return final_answer
        else:
            if tool_arguments is None:
                tool_arguments = {}
            observation = self.execute_tool_call(tool_name, tool_arguments)
            observation_type = type(observation)
            if observation_type in [AgentImage, AgentAudio]:
                if observation_type == AgentImage:
                    observation_name = "image.png"
                elif observation_type == AgentAudio:
                    observation_name = "audio.mp3"
                # TODO: observation naming could allow for different names of same type

                self.state[observation_name] = observation
                updated_information = f"Stored '{observation_name}' in memory."
            else:
                updated_information = str(observation).strip()

            await self.logger.log(
                f"Observations: {updated_information.replace('[', '|')}",  # escape potential rich-tag-like components
                level=LogLevel.INFO,
            )

            memory_step.observations = updated_information
            return None


class AsyncCodeAgent(AsyncMultiStepAgent):
    def __init__(
        self,
        tools: List[AsyncTool],
        model: Callable[[List[Dict[str, str]]], ChatMessage],
        prompt_templates: Optional[PromptTemplates] = None,
        grammar: Optional[Dict[str, str]] = None,
        additional_authorized_imports: Optional[List[str]] = None,
        planning_interval: Optional[int] = None,
        executor_type: str | None = "local",
        executor_kwargs: Optional[Dict[str, Any]] = None,
        max_print_outputs_length: Optional[int] = None,
        **kwargs,
    ):
        self.additional_authorized_imports = additional_authorized_imports if additional_authorized_imports else []
        self.authorized_imports = sorted(set(BASE_BUILTIN_MODULES) | set(self.additional_authorized_imports))
        self.max_print_outputs_length = max_print_outputs_length
        prompt_templates = prompt_templates or yaml.safe_load(
            importlib.resources.files("smolagents.prompts").joinpath("code_agent.yaml").read_text()
        )
        super().__init__(
            tools=tools,
            model=model,
            prompt_templates=prompt_templates,
            grammar=grammar,
            planning_interval=planning_interval,
            **kwargs,
        )
        if "*" in self.additional_authorized_imports:
            self.logger.log(
                "Caution: you set an authorization for all imports, meaning your agent can decide to import any package it deems necessary. This might raise issues if the package is not installed in your environment.",
                0,
            )
        self.executor_type = executor_type or "local"
        self.executor_kwargs = executor_kwargs or {}
        self.python_executor: PythonExecutor = self.create_python_executor()

    def create_python_executor(self) -> PythonExecutor:
        match self.executor_type:
            case "e2b" | "docker":
                if self.managed_agents:
                    raise Exception("Managed agents are not yet supported with remote code execution.")
                if self.executor_type == "e2b":
                    return E2BExecutor(self.additional_authorized_imports, self.logger, **self.executor_kwargs)
                else:
                    return DockerExecutor(self.additional_authorized_imports, self.logger, **self.executor_kwargs)
            case "local":
                return AsyncLocalPythonExecutor(
                    self.additional_authorized_imports,
                    max_print_outputs_length=self.max_print_outputs_length,
                )
            case _:  # if applicable
                raise ValueError(f"Unsupported executor type: {self.executor_type}")

    def initialize_system_prompt(self) -> str:
        system_prompt = populate_template(
            self.prompt_templates["system_prompt"],
            variables={
                "tools": self.tools,
                "managed_agents": self.managed_agents,
                "authorized_imports": (
                    "You can import from any package you want."
                    if "*" in self.authorized_imports
                    else str(self.authorized_imports)
                ),
            },
        )
        return system_prompt

    async def step(self, memory_step: ActionStep) -> Union[None, Any]:
        """
        Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        Returns None if the step is not final.
        """
        memory_messages = self.write_memory_to_messages()

        self.input_messages = memory_messages.copy()

        # Add new step in logs
        memory_step.model_input_messages = memory_messages.copy()
        try:
            additional_args = {"grammar": self.grammar} if self.grammar is not None else {}
            chat_message: ChatMessage = await self.model(
                self.input_messages,
                stop_sequences=["<end_code>", "Observation:"],
                **additional_args,
            )
            memory_step.model_output_message = chat_message
            model_output = chat_message.content
            memory_step.model_output = model_output
        except Exception as e:
            raise AgentGenerationError(f"Error in generating model output:\n{e}", self.logger) from e

        await self.logger.log_markdown(
            content=model_output,
            title="Output message of the LLM:",
            level=LogLevel.DEBUG,
        )

        # Parse
        try:
            code_action = fix_final_answer_code(parse_code_blobs(model_output))
        except Exception as e:
            error_msg = f"Error in code parsing:\n{e}\nMake sure to provide correct code blobs."
            raise AgentParsingError(error_msg, self.logger)

        memory_step.tool_calls = [
            ToolCall(
                name="python_interpreter",
                arguments=code_action,
                id=f"call_{len(self.memory.steps)}",
            )
        ]

        # Execute
        await self.logger.log_code(title="Executing parsed code:", content=code_action, level=LogLevel.INFO)
        is_final_answer = False
        try:
            output, execution_logs, is_final_answer = await self.python_executor(code_action)
            execution_outputs_console = []
            if len(execution_logs) > 0:
                execution_outputs_console += [
                    Text("Execution logs:", style="bold"),
                    Text(execution_logs),
                ]
            observation = "Execution logs:\n" + execution_logs
        except Exception as e:
            if hasattr(self.python_executor, "state") and "_print_outputs" in self.python_executor.state:
                execution_logs = str(self.python_executor.state["_print_outputs"])
                if len(execution_logs) > 0:
                    execution_outputs_console = [
                        Text("Execution logs:", style="bold"),
                        Text(execution_logs),
                    ]
                    memory_step.observations = "Execution logs:\n" + execution_logs
                    await self.logger.log(Group(*execution_outputs_console), level=LogLevel.INFO)
            error_msg = str(e)
            if "Import of " in error_msg and " is not allowed" in error_msg:
                await self.logger.log(
                    "[bold red]Warning to user: Code execution failed due to an unauthorized import - Consider passing said import under `additional_authorized_imports` when initializing your CodeAgent.",
                    level=LogLevel.INFO,
                )
            raise AgentExecutionError(error_msg, self.logger)

        truncated_output = truncate_content(str(output))
        observation += "Last output from code snippet:\n" + truncated_output
        memory_step.observations = observation

        execution_outputs_console += [
            Text(
                f"{('Out - Final answer' if is_final_answer else 'Out')}: {truncated_output}",
                style=(f"bold {YELLOW_HEX}" if is_final_answer else ""),
            ),
        ]
        await self.logger.log(Group(*execution_outputs_console), level=LogLevel.INFO)
        memory_step.action_output = output
        return output if is_final_answer else None