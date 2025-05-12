import importlib
import inspect
import json
import os
import re
import tempfile
import textwrap
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, AsyncGenerator
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

import jinja2
import yaml
from huggingface_hub import create_repo, metadata_update, snapshot_download, upload_folder
from jinja2 import StrictUndefined, Template
from rich.console import Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text


if TYPE_CHECKING:
    import PIL.Image

from .agent_types import AgentAudio, AgentImage, handle_agent_output_types
from .default_tools import TOOL_MAPPING, FinalAnswerTool
from .local_python_executor import BASE_BUILTIN_MODULES, LocalPythonExecutor, PythonExecutor, fix_final_answer_code
from .memory import (
    ActionStep,
    AgentMemory,
    FinalAnswerStep,
    Message,
    PlanningStep,
    SystemPromptStep,
    TaskStep,
    ToolCall,
)
from .models import ChatMessage, ChatMessageStreamDelta, MessageRole, parse_json_if_needed
from .async_models import AsyncModel
from .monitoring import (
    YELLOW_HEX,
    LogLevel,
)
from .remote_executors import DockerExecutor, E2BExecutor
from .tools import Tool
from .utils import (
    AgentError,
    AgentExecutionError,
    AgentGenerationError,
    AgentMaxStepsError,
    AgentParsingError,
    AgentToolCallError,
    AgentToolExecutionError,
    is_valid_name,
    make_init_file,
    parse_code_blobs,
    truncate_content,
)
from .agents import PromptTemplates, EMPTY_PROMPT_TEMPLATES, MultiStepAgent, populate_template
from .async_monitoring import AsyncMonitor, AsyncAgentLogger
logger = getLogger(__name__)


class AsyncMultiStepAgentBase:
    def __init__(
            self,
            tools: list[Tool],
            model: AsyncModel ,
            prompt_templates: PromptTemplates | None = None,
            max_steps: int = 20,
            add_base_tools: bool = False,
            verbosity_level: LogLevel = LogLevel.INFO,
            grammar: dict[str, str] | None = None,
            managed_agents: list | None = None,
            step_callbacks: list[Callable] | None = None,
            planning_interval: int | None = None,
            name: str | None = None,
            description: str | None = None,
            provide_run_summary: bool = False,
            final_answer_checks: list[Callable] | None = None,
            logger: AsyncAgentLogger | None = None,
    ):
        raise NotImplementedError

    async def run(
        self,
        task: str,
        stream: bool = False,
        reset: bool = True,
        images: list["PIL.Image.Image"] | None = None,
        additional_args: dict | None = None,
        max_steps: int | None = None,
    ):
        raise NotImplementedError

    async def _run_stream(
        self, task: str, max_steps: int, images: list["PIL.Image.Image"] | None = None
    ) -> AsyncGenerator[ActionStep | PlanningStep | FinalAnswerStep]:
        raise NotImplementedError

    async def _execute_step(self, memory_step: ActionStep) -> AsyncGenerator[Any]:
        raise NotImplementedError

    async def _finalize_step(self, memory_step: ActionStep, step_start_time: float):
        raise NotImplementedError

    async def _handle_max_steps_reached(self, task: str, images: list["PIL.Image.Image"], step_start_time: float) -> Any:
        raise NotImplementedError


    async def _generate_planning_step(
            self, task, is_first_step: bool, step: int
    ) -> AsyncGenerator[PlanningStep]:
        raise NotImplementedError

    async def _step_stream(self, memory_step: ActionStep) -> AsyncGenerator[ActionStep]:
        raise NotImplementedError

    async def step(self, memory_step: ActionStep) -> Any:
        """
        Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        Returns either None if the step is not final, or the final answer.
        """
        raise NotImplementedError

    async def provide_final_answer(self, task: str, images: list["PIL.Image.Image"] | None = None) -> str:
        """
        Provide the final answer to the user.
        """
        raise NotImplementedError

    async def __call__(self, task:str, **kwargs) -> str:
        """
        Call the agent with a task and return the final answer.
        """
        raise NotImplementedError

    async def execute_tool_call(self, tool_name: str, arguments: dict[str, str] | str) -> Any:
        """
        Execute a tool or managed agent with the provided arguments.
        """
        raise NotImplementedError


class AsyncMultiStepAgent(AsyncMultiStepAgentBase, MultiStepAgent, ABC):
    def __init__(
        self,
        tools: list[Tool],
        model: AsyncModel , #TODO: Change to the AsyncModel
        prompt_templates: PromptTemplates | None = None,
        max_steps: int = 20,
        add_base_tools: bool = False,
        verbosity_level: LogLevel = LogLevel.INFO,
        grammar: dict[str, str] | None = None,
        managed_agents: list | None = None,
        step_callbacks: list[Callable] | None = None,
        planning_interval: int | None = None,
        name: str | None = None,
        description: str | None = None,
        provide_run_summary: bool = False,
        final_answer_checks: list[Callable] | None = None,
        logger: AsyncAgentLogger | None = None,
    ):
        self.agent_name = self.__class__.__name__
        self.model = model

        self.prompt_templates = prompt_templates or EMPTY_PROMPT_TEMPLATES
        if prompt_templates is not None:
            missing_keys = set(EMPTY_PROMPT_TEMPLATES.keys()) - set(prompt_templates.keys())
            assert not missing_keys, (
                f"Some prompt templates are missing from your custom `prompt_templates`: {missing_keys}"
            )
            for key, value in EMPTY_PROMPT_TEMPLATES.items():
                if isinstance(value, dict):
                    for subkey in value.keys():
                        assert key in prompt_templates.keys() and (subkey in prompt_templates[key].keys()), (
                            f"Some prompt templates are missing from your custom `prompt_templates`: {subkey} under {key}"
                        )

        self.max_steps = max_steps
        self.step_number = 0
        self.grammar = grammar
        self.planning_interval = planning_interval
        self.state: dict[str, Any] = {}
        self.name = self._validate_name(name)
        self.description = description
        self.provide_run_summary = provide_run_summary
        self.final_answer_checks = final_answer_checks

        self._setup_managed_agents(managed_agents)
        self._setup_tools(tools, add_base_tools)
        self._validate_tools_and_managed_agents(tools, managed_agents)

        self.system_prompt = self.initialize_system_prompt()
        self.task: str | None = None
        self.memory = AgentMemory(self.system_prompt)

        if logger is None:
            self.logger = AsyncAgentLogger(level=verbosity_level)
        else:
            self.logger = logger

        self.monitor = AsyncMonitor(self.model, self.logger)
        self.step_callbacks = step_callbacks if step_callbacks is not None else []
        self.step_callbacks.append(self.monitor.update_metrics)
        self.stream_outputs = False


    async def run(
        self,
        task: str,
        stream: bool = False,
        reset: bool = True,
        images: list["PIL.Image.Image"] | None = None,
        additional_args: dict | None = None,
        max_steps: int | None = None,
    ):
        """
        Run the agent with the given task and return the final answer.
        """
        max_steps = max_steps or self.max_steps
        self.task = task
        self.interrupt_switch = False
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
            #self.python_executor.send_tools({**self.tools, **self.managed_agents})
            self.python_executor.send_tools({**self.tools, })

        if stream:
            # The steps are returned as they are executed through a generator to iterate on.
            return self._run_stream(task=self.task, max_steps=max_steps, images=images)

        # Outputs are returned only at the end. We only look at the last step.
        steps = []
        async for step in self._run_stream(task=self.task, max_steps=max_steps, images=images):
            steps.append(step)
        return steps[-1].final_answer

    async def _run_stream(
        self, task: str, max_steps: int, images: list["PIL.Image.Image"] | None = None
    ) -> AsyncGenerator[ActionStep | PlanningStep | FinalAnswerStep]:
        """
        Run the agent with the given task and return the final answer.
        """
        final_answer = None
        action_step = None
        step_start_time = time.time()
        self.step_number = 1
        while final_answer is None and self.step_number <= max_steps:
            if self.interrupt_switch:
                raise AgentError("Agent interrupted.", self.logger)
            step_start_time = time.time()
            if self.planning_interval is not None and (
                    self.step_number == 1 or (self.step_number - 1) % self.planning_interval == 0
            ):
                async for element in self._generate_planning_step(
                        task, is_first_step=(self.step_number == 1), step=self.step_number
                ):
                    yield element
                self.memory.steps.append(element)
            action_step = ActionStep(
                step_number=self.step_number, start_time=step_start_time, observations_images=images
            )
            try:
                steps = self._execute_step(action_step)
                el = ""
                async for el in steps:
                    yield el
                final_answer = el
            except AgentGenerationError as e:
                # Agent generation errors are not caused by a Model error but an implementation error: so we should raise them and exit.
                raise e
            except AgentError as e:
                # Other AgentError types are caused by the Model, so we should log them and iterate.
                action_step.error = e
            finally:
                await self._finalize_step(action_step, step_start_time)
                self.memory.steps.append(action_step)
                yield action_step
                self.step_number += 1

        if final_answer is None and self.step_number == max_steps + 1:
            final_answer = await self._handle_max_steps_reached(task, images, step_start_time)
            yield action_step
        yield FinalAnswerStep(handle_agent_output_types(final_answer))


    async def _execute_step(self, memory_step: ActionStep) -> AsyncGenerator[Any]:
        await self.logger.log_rule(f"Step {self.step_number}", level=LogLevel.INFO)
        final_answer = None
        async for el in self._step_stream(memory_step):
            final_answer = el
            yield el
        if final_answer is not None and self.final_answer_checks:
            self._validate_final_answer(final_answer)

        yield final_answer

    async def _finalize_step(self, memory_step: ActionStep, step_start_time: float):
        memory_step.end_time = time.time()
        memory_step.duration = memory_step.end_time - step_start_time
        for callback in self.step_callbacks:
            # For compatibility with old callbacks that don't take the agent as an argument
            if len(inspect.signature(callback).parameters) == 1:
                await callback(memory_step)
            else:
                await callback(
                    memory_step, agent=self
                )

    async def _handle_max_steps_reached(self, task: str, images: list["PIL.Image.Image"], step_start_time: float) -> Any:
        final_answer = await self.provide_final_answer(task, images)
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
                await callback(
                    final_memory_step, agent=self
                )

        return final_answer

    async def _generate_planning_step(
            self, task, is_first_step: bool, step: int
    ) -> AsyncGenerator[PlanningStep, ChatMessageStreamDelta]:
        if is_first_step:
            input_messages = [
                {
                    "role": MessageRole.USER,
                    "content": [
                        {
                            "type": "text",
                            "text": populate_template(
                                self.prompt_templates["planning"]["initial_plan"],
                                variables={"task": task, "tools": self.tools, "managed_agents": self.managed_agents},
                            ),
                        }
                    ],
                }
            ]
            if self.stream_outputs and hasattr(self.model, "generate_stream"):
                plan_message_content = ""
                async for completion_delta in self.model.generate_stream(input_messages, stop_sequences=["<end_plan>"]):  # type: ignore
                    plan_message_content += completion_delta.content
                    yield completion_delta
            else:
                # TODO: Make sure the model is AsyncModel
                plan_message_content = await self.model.generate(input_messages, stop_sequences=["<end_plan>"])
                plan_message_content = plan_message_content.content
            plan = textwrap.dedent(
                f"""Here are the facts I know and the plan of action that I will follow to solve the task:\n```\n{plan_message_content}\n```"""
            )
        else:
            # Summary mode removes the system prompt and previous planning messages output by the model.
            # Removing previous planning messages avoids influencing too much the new plan.
            memory_messages = self.write_memory_to_messages(summary_mode=True)
            plan_update_pre = {
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
            plan_update_post = {
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
                                "remaining_steps": (self.max_steps - step),
                            },
                        ),
                    }
                ],
            }
            input_messages = [plan_update_pre] + memory_messages + [plan_update_post]
            if self.stream_outputs and hasattr(self.model, "generate_stream"):
                plan_message_content = ""
                async for completion_delta in self.model.generate_stream(input_messages, stop_sequences=["<end_plan>"]):  # type: ignore
                    plan_message_content += completion_delta.content
                    yield completion_delta
            else:
                plan_message_content = await self.model.generate(input_messages, stop_sequences=["<end_plan>"])
                plan_message_content = plan_message_content.content

            plan = textwrap.dedent(
                f"""I still need to solve the task I was given:\n```\n{self.task}\n```\n\nHere are the facts I know and my new/updated plan of action to solve the task:\n```\n{plan_message_content}\n```"""
            )
        log_headline = "Initial plan" if is_first_step else "Updated plan"
        await self.logger.log(Rule(f"[bold]{log_headline}", style="orange"), Text(plan), level=LogLevel.INFO)

        yield PlanningStep(
            model_input_messages=input_messages,
            plan=plan,
            model_output_message=ChatMessage(role=MessageRole.ASSISTANT, content=plan_message_content),
        )

    async def _step_stream(self, memory_step: ActionStep) -> AsyncGenerator[ActionStep]:
        """

        Args:
            memory_step:

        Returns:

        """

        yield None

    async def step(self, memory_step: ActionStep) -> Any:
        """
        Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        Returns either None if the step is not final, or the final answer.
        """
        last = None
        async for item in self._step_stream(memory_step):
            last = item
        return last

    async def provide_final_answer(self, task: str, images: list["PIL.Image.Image"] | None = None) -> str:
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

    async def __call__(self, task: str, **kwargs) -> str:
        """
        Call the agent with a task and return the final answer.
        """
        full_task = populate_template(
            self.prompt_templates["managed_agent"]["task"],
            variables=dict(name=self.name, task=task),
        )

        report = self.run(full_task, **kwargs)

        answer = populate_template(
            self.prompt_templates["managed_agent"]["report"], variables=dict(name=self.name, final_answer=report)
        )
        if self.provide_run_summary:
            answer += "\n\nFor more detail, find below a summary of this agent's work:\n<summary_of_work>\n"
            for message in self.write_memory_to_messages(summary_mode=True):
                content = message["content"]
                answer += "\n" + truncate_content(str(content)) + "\n---"
            answer += "\n</summary_of_work>"
        return answer

class AsyncToolCallingAgent(AsyncMultiStepAgent):
    def __init__(
        self,
        tools: list[Tool],
        model: Callable[[list[dict[str, str]]], ChatMessage],
        prompt_templates: PromptTemplates | None = None,
        planning_interval: int | None = None,
        **kwargs,
    ):
        prompt_templates = prompt_templates or yaml.safe_load(
            importlib.resources.files("smolagents.prompts").joinpath("toolcalling_agent.yaml").read_text()
        )
        super().__init__(
            tools=tools,
            model=model, # type: ignore
            prompt_templates=prompt_templates,
            planning_interval=planning_interval,
            **kwargs,
        )

    def initialize_system_prompt(self) -> str:
        system_prompt = populate_template(
            self.prompt_templates["system_prompt"],
            variables={"tools": self.tools, "managed_agents": self.managed_agents},
        )
        return system_prompt

    async def _step_stream(self, memory_step: ActionStep) -> AsyncGenerator[Any]:
        """
        Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        Yields either None if the step is not final, or the final answer.
        """
        memory_messages = self.write_memory_to_messages()

        input_messages = memory_messages.copy()

        # Add new step in logs
        memory_step.model_input_messages = input_messages

        try:
            chat_message: ChatMessage = await self.model(
                input_messages,
                stop_sequences=["Observation:", "Calling tools:"],
                tools_to_call_from=list(self.tools.values()),
            )
            memory_step.model_output_message = chat_message
            model_output = chat_message.content
            await self.logger.log_markdown(
                content=model_output if model_output else str(chat_message.raw),
                title="Output message of the LLM:",
                level=LogLevel.DEBUG,
            )

            memory_step.model_output_message.content = model_output
            memory_step.model_output = model_output
        except Exception as e:
            raise AgentGenerationError(f"Error while generating output:\n{e}", self.logger) from e

        if chat_message.tool_calls is None or len(chat_message.tool_calls) == 0:
            try:
                chat_message = self.model.parse_tool_calls(chat_message)
            except Exception as e:
                raise AgentParsingError(f"Error while parsing tool call from model output: {e}", self.logger)
        else:
            for tool_call in chat_message.tool_calls:
                tool_call.function.arguments = parse_json_if_needed(tool_call.function.arguments)
        tool_call = chat_message.tool_calls[0]  # type: ignore
        tool_name, tool_call_id = tool_call.function.name, tool_call.id
        tool_arguments = tool_call.function.arguments
        memory_step.model_output = str(f"Called Tool: '{tool_name}' with arguments: {tool_arguments}")
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
            if isinstance(answer, str) and answer in self.state.keys():
                # if the answer is a state variable, return the value
                # State variables are not JSON-serializable (AgentImage, AgentAudio) so can't be passed as arguments to execute_tool_call
                final_answer = self.state[answer]
                await self.logger.log(
                    f"[bold {YELLOW_HEX}]Final answer:[/bold {YELLOW_HEX}] Extracting key '{answer}' from state to return value '{final_answer}'.",
                    level=LogLevel.INFO,
                )
            else:
                final_answer = await self.execute_tool_call("final_answer", {"answer": answer})
                await self.logger.log(
                    Text(f"Final answer: {final_answer}", style=f"bold {YELLOW_HEX}"),
                    level=LogLevel.INFO,
                )

            memory_step.action_output = final_answer
            yield final_answer
        else:
            if tool_arguments is None:
                tool_arguments = {}
            observation = await self.execute_tool_call(tool_name, tool_arguments)
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
            yield None

    def _substitute_state_variables(self, arguments: dict[str, str] | str) -> dict[str, Any] | str:
        """Replace string values in arguments with their corresponding state values if they exist."""
        if isinstance(arguments, dict):
            return {
                key: self.state.get(value, value) if isinstance(value, str) else value
                for key, value in arguments.items()
            }
        return arguments

    async def execute_tool_call(self, tool_name: str, arguments: dict[str, str] | str) -> Any:
        """
        Execute a tool or managed agent with the provided arguments.

        The arguments are replaced with the actual values from the state if they refer to state variables.

        Args:
            tool_name (`str`): Name of the tool or managed agent to execute.
            arguments (dict[str, str] | str): Arguments passed to the tool call.
        """
        # Check if the tool exists
        available_tools = {**self.tools, **self.managed_agents}
        if tool_name not in available_tools:
            raise AgentToolExecutionError(
                f"Unknown tool {tool_name}, should be one of: {', '.join(available_tools)}.", self.logger
            )

        # Get the tool and substitute state variables in arguments
        tool = available_tools[tool_name]
        arguments = self._substitute_state_variables(arguments)
        is_managed_agent = tool_name in self.managed_agents

        try:
            # Call tool with appropriate arguments
            if isinstance(arguments, dict):
                if is_managed_agent:
                    output = await tool(**arguments)
                    return output
                else:
                    return tool(**arguments, sanitize_inputs_outputs=True)
            elif isinstance(arguments, str):
                if is_managed_agent:
                    output = await tool(arguments)
                    return output
                else:
                    # If the tool is not a managed agent, we need to sanitize the inputs and outputs
                    # before passing them to the tool.
                    # This is because the tool may not be able to handle raw strings.
                    return tool(arguments, sanitize_inputs_outputs=True)
            else:
                raise TypeError(f"Unsupported arguments type: {type(arguments)}")

        except TypeError as e:
            # Handle invalid arguments
            description = getattr(tool, "description", "No description")
            if is_managed_agent:
                error_msg = (
                    f"Invalid request to team member '{tool_name}' with arguments {json.dumps(arguments)}: {e}\n"
                    "You should call this team member with a valid request.\n"
                    f"Team member description: {description}"
                )
            else:
                error_msg = (
                    f"Invalid call to tool '{tool_name}' with arguments {json.dumps(arguments)}: {e}\n"
                    "You should call this tool with correct input arguments.\n"
                    f"Expected inputs: {json.dumps(tool.inputs)}\n"
                    f"Returns output type: {tool.output_type}\n"
                    f"Tool description: '{description}'"
                )
            raise AgentToolCallError(error_msg, self.logger) from e

        except Exception as e:
            # Handle execution errors
            if is_managed_agent:
                error_msg = (
                    f"Error executing request to team member '{tool_name}' with arguments {json.dumps(arguments)}: {e}\n"
                    "Please try again or request to another team member"
                )
            else:
                error_msg = (
                    f"Error executing tool '{tool_name}' with arguments {json.dumps(arguments)}: {type(e).__name__}: {e}\n"
                    "Please try again or use another tool"
                )
            raise AgentToolExecutionError(error_msg, self.logger) from e



class AsyncCodeAgent(AsyncMultiStepAgent):
    def __init__(
            self,
            tools: list[Tool],
            model: AsyncModel ,
            prompt_templates: PromptTemplates | None = None,
            grammar: dict[str, str] | None = None,
            additional_authorized_imports: list[str] | None = None,
            planning_interval: int | None = None,
            executor_type: str | None = "local",
            executor_kwargs: dict[str, Any] | None = None,
            max_print_outputs_length: int | None = None,
            stream_outputs: bool = False,
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
        self.stream_outputs = stream_outputs
        if self.stream_outputs and not hasattr(self.model, "generate_stream"):
            raise ValueError(
                "`stream_outputs` is set to True, but the model class implements no `generate_stream` method."
            )

        self.executor_type = executor_type or "local"
        self.executor_kwargs = executor_kwargs or {}
        self.python_executor = self.create_python_executor()

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
                return LocalPythonExecutor(
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

    async def _step_stream(self, memory_step: ActionStep) -> AsyncGenerator[ActionStep]:
        """
                Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
                Yields either None if the step is not final, or the final answer.
                """
        memory_messages = self.write_memory_to_messages()

        input_messages = memory_messages.copy()
        ### Generate model output ###
        memory_step.model_input_messages = input_messages
        try:
            additional_args = {"grammar": self.grammar} if self.grammar is not None else {}
            if self.stream_outputs:
                output_stream = self.model.generate_stream(
                    input_messages,
                    stop_sequences=["<end_code>", "Observation:", "Calling tools:"],
                    **additional_args,
                )
                output_text = ""
                with Live("", console=self.logger.console, vertical_overflow="visible") as live:
                    async for event in output_stream:
                        if event.content is not None:
                            output_text += event.content
                            live.update(Markdown(output_text))
                        yield event

                model_output = output_text
                chat_message = ChatMessage(role="assistant", content=model_output)
                memory_step.model_output_message = chat_message
                model_output = chat_message.content
            else:
                chat_message: ChatMessage = await self.model.generate(
                    input_messages,
                    stop_sequences=["<end_code>", "Observation:", "Calling tools:"],
                    **additional_args,
                )
                memory_step.model_output_message = chat_message
                model_output = chat_message.content
                await self.logger.log_markdown(
                    content=model_output,
                    title="Output message of the LLM:",
                    level=LogLevel.DEBUG,
                )

            # This adds <end_code> sequence to the history.
            # This will nudge ulterior LLM calls to finish with <end_code>, thus efficiently stopping generation.

            if model_output and model_output.strip().endswith("```"):
                model_output += "<end_code>"
                memory_step.model_output_message.content = model_output

            memory_step.model_output = model_output
        except Exception as e:
            raise AgentGenerationError(f"Error in generating model output:\n{e}", self.logger) from e

        ### Parse output ###
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

        ### Execute action ###
        await self.logger.log_code(title="Executing parsed code:", content=code_action, level=LogLevel.INFO)
        is_final_answer = False
        try:
            output, execution_logs, is_final_answer = self.python_executor(code_action)
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
        yield output if is_final_answer else None