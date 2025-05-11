# coding=utf-8
# Copyright 2024 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import io
import os
import tempfile
import uuid
from collections.abc import Generator
from contextlib import nullcontext as does_not_raise
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from huggingface_hub import (
    ChatCompletionOutputFunctionDefinition,
    ChatCompletionOutputMessage,
    ChatCompletionOutputToolCall,
)
from rich.console import Console

from smolagents import EMPTY_PROMPT_TEMPLATES
from smolagents.agent_types import AgentImage, AgentText
from smolagents.agents import (
    AgentError,
    AgentMaxStepsError,
    ToolCall,
    populate_template,
)
from smolagents.async_agents import AsyncToolCallingAgent, AsyncCodeAgent, AsyncMultiStepAgent
from smolagents.async_models import AsyncModel
from smolagents.default_tools import DuckDuckGoSearchTool, FinalAnswerTool, PythonInterpreterTool, VisitWebpageTool
from smolagents.memory import ActionStep, PlanningStep
from smolagents.models import (
    ChatMessage,
    ChatMessageToolCall,
    ChatMessageToolCallDefinition,
    MessageRole,
)

from smolagents.monitoring import AgentLogger, LogLevel
from smolagents.tools import Tool, tool
from smolagents.utils import BASE_BUILTIN_MODULES, AgentExecutionError, AgentGenerationError, AgentToolCallError


def get_new_path(suffix="") -> str:
    directory = tempfile.mkdtemp()
    return os.path.join(directory, str(uuid.uuid4()) + suffix)


@pytest.fixture
def agent_logger():
    return AgentLogger(
        LogLevel.DEBUG, console=Console(record=True, no_color=True, force_terminal=False, file=io.StringIO())
    )


class FakeToolCallModel(AsyncModel):
    async def generate(self, messages, tools_to_call_from=None, stop_sequences=None, grammar=None, **kwargs):
        if len(messages) < 3:
            yield ChatMessage(
                role="assistant",
                content="",
                tool_calls=[
                    ChatMessageToolCall(
                        id="call_0",
                        type="function",
                        function=ChatMessageToolCallDefinition(
                            name="python_interpreter", arguments={"code": "2*3.6452"}
                        ),
                    )
                ],
            )
        else:
            yield ChatMessage(
                role="assistant",
                content="",
                tool_calls=[
                    ChatMessageToolCall(
                        id="call_1",
                        type="function",
                        function=ChatMessageToolCallDefinition(name="final_answer", arguments={"answer": "7.2904"}),
                    )
                ],
            )


class FakeToolCallModelImage(AsyncModel):
    async def generate(self, messages, tools_to_call_from=None, stop_sequences=None, grammar=None, **kwargs):
        if len(messages) < 3:
            yield ChatMessage(
                role="assistant",
                content="",
                tool_calls=[
                    ChatMessageToolCall(
                        id="call_0",
                        type="function",
                        function=ChatMessageToolCallDefinition(
                            name="fake_image_generation_tool",
                            arguments={"prompt": "An image of a cat"},
                        ),
                    )
                ],
            )
        else:
            yield ChatMessage(
                role="assistant",
                content="",
                tool_calls=[
                    ChatMessageToolCall(
                        id="call_1",
                        type="function",
                        function=ChatMessageToolCallDefinition(name="final_answer", arguments="image.png"),
                    )
                ],
            )


class FakeToolCallModelVL(AsyncModel):
    async def generate(self, messages, tools_to_call_from=None, stop_sequences=None, grammar=None, **kwargs):
        if len(messages) < 3:
            yield ChatMessage(
                role="assistant",
                content="",
                tool_calls=[
                    ChatMessageToolCall(
                        id="call_0",
                        type="function",
                        function=ChatMessageToolCallDefinition(
                            name="fake_image_understanding_tool",
                            arguments={
                                "prompt": "What is in this image?",
                                "image": "image.png",
                            },
                        ),
                    )
                ],
            )
        else:
            yield ChatMessage(
                role="assistant",
                content="",
                tool_calls=[
                    ChatMessageToolCall(
                        id="call_1",
                        type="function",
                        function=ChatMessageToolCallDefinition(name="final_answer", arguments="The image is a cat."),
                    )
                ],
            )


class FakeCodeModel(AsyncModel):
    async def generate(self, messages, stop_sequences=None, grammar=None, **kwargs):
        prompt = str(messages)
        if "special_marker" not in prompt:
            yield ChatMessage(
                role="assistant",
                content="""
Thought: I should multiply 2 by 3.6452. special_marker
Code:
```py
result = 2**3.6452
```<end_code>
""",
            )
        else:  # We're at step 2
            yield ChatMessage(
                role="assistant",
                content="""
Thought: I can now answer the initial question
Code:
```py
final_answer(7.2904)
```<end_code>
""",
            )


class FakeCodeModelError(AsyncModel):
    async def generate(self, messages, stop_sequences=None):
        prompt = str(messages)
        if "special_marker" not in prompt:
            yield ChatMessage(
                role="assistant",
                content="""
Thought: I should multiply 2 by 3.6452. special_marker
Code:
```py
print("Flag!")
def error_function():
    raise ValueError("error")

error_function()
```<end_code>
""",
            )
        else:  # We're at step 2
            yield ChatMessage(
                role="assistant",
                content="""
Thought: I faced an error in the previous step.
Code:
```py
final_answer("got an error")
```<end_code>
""",
            )


class FakeCodeModelSyntaxError(AsyncModel):
    async def generate(self, messages, stop_sequences=None):
        prompt = str(messages)
        if "special_marker" not in prompt:
            yield ChatMessage(
                role="assistant",
                content="""
Thought: I should multiply 2 by 3.6452. special_marker
Code:
```py
a = 2
b = a * 2
    print("Failing due to unexpected indent")
print("Ok, calculation done!")
```<end_code>
""",
            )
        else:  # We're at step 2
            yield ChatMessage(
                role="assistant",
                content="""
Thought: I can now answer the initial question
Code:
```py
final_answer("got an error")
```<end_code>
""",
            )


class FakeCodeModelImport(AsyncModel):
    async def generate(self, messages, stop_sequences=None):
        yield ChatMessage(
            role="assistant",
            content="""
Thought: I can answer the question
Code:
```py
import numpy as np
final_answer("got an error")
```<end_code>
""",
        )


class FakeCodeModelFunctionDef(AsyncModel):
    async def generate(self, messages, stop_sequences=None):
        prompt = str(messages)
        if "special_marker" not in prompt:
            return ChatMessage(
                role="assistant",
                content="""
Thought: Let's define the function. special_marker
Code:
```py
import numpy as np

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w
```<end_code>
    """,
            )
        else:  # We're at step 2
            return ChatMessage(
                role="assistant",
                content="""
Thought: I can now answer the initial question
Code:
```py
x, w = [0, 1, 2, 3, 4, 5], 2
res = moving_average(x, w)
final_answer(res)
```<end_code>
""",
            )


class FakeCodeModelSingleStep(AsyncModel):
    async def generate(self, messages, stop_sequences=None, grammar=None, **kwargs):
        return ChatMessage(
            role="assistant",
            content="""
Thought: I should multiply 2 by 3.6452. special_marker
Code:
```py
result = python_interpreter(code="2*3.6452")
final_answer(result)
```
""",
        )


class FakeCodeModelNoReturn(AsyncModel):
    async def generate(self, messages, stop_sequences=None, grammar=None, **kwargs):
        return ChatMessage(
            role="assistant",
            content="""
Thought: I should multiply 2 by 3.6452. special_marker
Code:
```py
result = python_interpreter(code="2*3.6452")
print(result)
```
""",
        )


class TestAgent:
    async def test_fake_toolcalling_agent(self):
        agent = AsyncToolCallingAgent(tools=[PythonInterpreterTool()], model=FakeToolCallModel())
        output = await agent.run("What is 2 multiplied by 3.6452?")
        assert isinstance(output, str)
        assert "7.2904" in output
        assert agent.memory.steps[0].task == "What is 2 multiplied by 3.6452?"
        assert "7.2904" in agent.memory.steps[1].observations
        assert agent.memory.steps[2].model_output == "Called Tool: 'final_answer' with arguments: {'answer': '7.2904'}"

    async def test_toolcalling_agent_handles_image_tool_outputs(self, shared_datadir):
        import PIL.Image

        @tool
        def fake_image_generation_tool(prompt: str) -> PIL.Image.Image:
            """Tool that generates an image.

            Args:
                prompt: The prompt
            """

            import PIL.Image

            return PIL.Image.open(shared_datadir / "000000039769.png")

        agent = AsyncToolCallingAgent(tools=[fake_image_generation_tool], model=FakeToolCallModelImage())
        output = await agent.run("Make me an image.")
        assert isinstance(output, AgentImage)
        assert isinstance(agent.state["image.png"], PIL.Image.Image)

    async def test_toolcalling_agent_handles_image_inputs(self, shared_datadir):
        import PIL.Image

        image = PIL.Image.open(shared_datadir / "000000039769.png")  # dummy input

        @tool
        def fake_image_understanding_tool(prompt: str, image: PIL.Image.Image) -> str:
            """Tool that creates a caption for an image.

            Args:
                prompt: The prompt
                image: The image
            """
            return "The image is a cat."

        agent = AsyncToolCallingAgent(tools=[fake_image_understanding_tool], model=FakeToolCallModelVL())
        output = await agent.run("Caption this image.", images=[image])
        assert output == "The image is a cat."

    async def test_fake_code_agent(self):
        agent = AsyncCodeAgent(tools=[PythonInterpreterTool()], model=FakeCodeModel())
        output = await agent.run("What is 2 multiplied by 3.6452?")
        assert isinstance(output, float)
        assert output == 7.2904
        assert agent.memory.steps[0].task == "What is 2 multiplied by 3.6452?"
        assert agent.memory.steps[2].tool_calls == [
            ToolCall(name="python_interpreter", arguments="final_answer(7.2904)", id="call_2")
        ]

    async def test_additional_args_added_to_task(self):
        agent = AsyncCodeAgent(tools=[], model=FakeCodeModel())
        await agent.run(
            "What is 2 multiplied by 3.6452?",
            additional_args={"instruction": "Remember this."},
        )
        assert "Remember this" in agent.task

    async def test_reset_conversations(self):
        agent = AsyncCodeAgent(tools=[PythonInterpreterTool()], model=FakeCodeModel())
        output = await agent.run("What is 2 multiplied by 3.6452?", reset=True)
        assert output == 7.2904
        assert len(agent.memory.steps) == 3

        output = await agent.run("What is 2 multiplied by 3.6452?", reset=False)
        assert output == 7.2904
        assert len(agent.memory.steps) == 5

        output = await agent.run("What is 2 multiplied by 3.6452?", reset=True)
        assert output == 7.2904
        assert len(agent.memory.steps) == 3

    async def test_setup_agent_with_empty_toolbox(self):
        AsyncToolCallingAgent(model=FakeToolCallModel(), tools=[])

    async def test_fails_max_steps(self):
        agent = AsyncCodeAgent(
            tools=[PythonInterpreterTool()],
            model=FakeCodeModelNoReturn(),  # use this callable because it never ends
            max_steps=5,
        )
        answer = await agent.run("What is 2 multiplied by 3.6452?")
        assert len(agent.memory.steps) == 7  # Task step + 5 action steps + Final answer
        assert type(agent.memory.steps[-1].error) is AgentMaxStepsError
        assert isinstance(answer, str)

        agent = AsyncCodeAgent(
            tools=[PythonInterpreterTool()],
            model=FakeCodeModelNoReturn(),  # use this callable because it never ends
            max_steps=5,
        )
        answer = await agent.run("What is 2 multiplied by 3.6452?", max_steps=3)
        assert len(agent.memory.steps) == 5  # Task step + 3 action steps + Final answer
        assert type(agent.memory.steps[-1].error) is AgentMaxStepsError
        assert isinstance(answer, str)

    async def test_tool_descriptions_get_baked_in_system_prompt(self):
        tool = PythonInterpreterTool()
        tool.name = "fake_tool_name"
        tool.description = "fake_tool_description"
        agent = AsyncCodeAgent(tools=[tool], model=FakeCodeModel())
        await agent.run("Empty task")
        assert agent.system_prompt is not None
        assert f"def {tool.name}(" in agent.system_prompt
        assert f'"""{tool.description}' in agent.system_prompt

    async def test_module_imports_get_baked_in_system_prompt(self):
        agent = AsyncCodeAgent(tools=[], model=FakeCodeModel())
        await agent.run("Empty task")
        for module in BASE_BUILTIN_MODULES:
            assert module in agent.system_prompt

    async def test_init_agent_with_different_toolsets(self):
        toolset_1 = []
        agent = AsyncCodeAgent(tools=toolset_1, model=FakeCodeModel())
        assert len(agent.tools) == 1  # when no tools are provided, only the final_answer tool is added by default

        toolset_2 = [PythonInterpreterTool(), PythonInterpreterTool()]
        with pytest.raises(ValueError) as e:
            agent = AsyncCodeAgent(tools=toolset_2, model=FakeCodeModel())
        assert "Each tool or managed_agent should have a unique name!" in str(e)

        with pytest.raises(ValueError) as e:
            agent.name = "python_interpreter"
            agent.description = "empty"
            AsyncCodeAgent(tools=[PythonInterpreterTool()], model=FakeCodeModel(), managed_agents=[agent])
        assert "Each tool or managed_agent should have a unique name!" in str(e)

        # check that python_interpreter base tool does not get added to CodeAgent
        agent = AsyncCodeAgent(tools=[], model=FakeCodeModel(), add_base_tools=True)
        assert len(agent.tools) == 3  # added final_answer tool + search + visit_webpage

        # check that python_interpreter base tool gets added to ToolCallingAgent
        agent = AsyncToolCallingAgent(tools=[], model=FakeCodeModel(), add_base_tools=True)
        assert len(agent.tools) == 4  # added final_answer tool + search + visit_webpage

    async def test_function_persistence_across_steps(self):
        agent = AsyncCodeAgent(
            tools=[],
            model=FakeCodeModelFunctionDef(),
            max_steps=2,
            additional_authorized_imports=["numpy"],
        )
        res = await agent.run("ok")
        assert res[0] == 0.5

    async def test_init_managed_agent(self):
        agent = AsyncCodeAgent(tools=[], model=FakeCodeModelFunctionDef(), name="managed_agent", description="Empty")
        assert agent.name == "managed_agent"
        assert agent.description == "Empty"

    async def test_agent_description_gets_correctly_inserted_in_system_prompt(self):
        managed_agent = AsyncCodeAgent(
            tools=[], model=FakeCodeModelFunctionDef(), name="managed_agent", description="Empty"
        )
        manager_agent = AsyncCodeAgent(
            tools=[],
            model=FakeCodeModelFunctionDef(),
            managed_agents=[managed_agent],
        )
        assert "You can also give tasks to team members." not in managed_agent.system_prompt
        assert "{{managed_agents_descriptions}}" not in managed_agent.system_prompt
        assert "You can also give tasks to team members." in manager_agent.system_prompt

    async def test_replay_shows_logs(self, agent_logger):
        agent = AsyncCodeAgent(
            tools=[],
            model=FakeCodeModelImport(),
            verbosity_level=0,
            additional_authorized_imports=["numpy"],
            logger=agent_logger,
        )
        await agent.run("Count to 3")

        str_output = agent_logger.console.export_text()

        assert "New run" in str_output
        assert 'final_answer("got' in str_output
        assert "```<end_code>" in str_output

        agent = AsyncToolCallingAgent(tools=[PythonInterpreterTool()], model=FakeToolCallModel(), verbosity_level=0)
        agent.logger = agent_logger

        await agent.run("What is 2 multiplied by 3.6452?")
        agent.replay()

        str_output = agent_logger.console.export_text()
        assert "Called Tool" in str_output
        assert "arguments" in str_output

    async def test_code_nontrivial_final_answer_works(self):
        class FakeCodeModelFinalAnswer(AsyncModel):
            async def generate(self, messages, stop_sequences=None, grammar=None, **kwargs):
                return ChatMessage(
                    role="assistant",
                    content="""Code:
```py
def nested_answer():
    final_answer("Correct!")

nested_answer()
```<end_code>""",
                )

        agent = AsyncCodeAgent(tools=[], model=FakeCodeModelFinalAnswer())

        output = await agent.run("Count to 3")
        assert output == "Correct!"

    async def test_final_answer_checks(self):
        def check_always_fails(final_answer, agent_memory):
            assert False, "Error raised in check"

        agent = AsyncCodeAgent(model=FakeCodeModel(), tools=[], final_answer_checks=[check_always_fails])
        await agent.run("Dummy task.")
        assert "Error raised in check" in str(agent.write_memory_to_messages())

    async def test_generation_errors_are_raised(self):
        class FakeCodeModel(AsyncModel):
            async def generate(self, messages, stop_sequences=None, grammar=None, **kwargs):
                assert False, "Generation failed"

        agent = AsyncCodeAgent(model=FakeCodeModel(), tools=[])
        with pytest.raises(AgentGenerationError) as e:
            await agent.run("Dummy task.")
        assert len(agent.memory.steps) == 2
        assert "Generation failed" in str(e)


class CustomFinalAnswerTool(FinalAnswerTool):
    def forward(self, answer) -> str:
        return answer + "CUSTOM"


class MockTool(Tool):
    def __init__(self, name):
        self.name = name
        self.description = "Mock tool description"
        self.inputs = {}
        self.output_type = "string"

    def forward(self):
        return "Mock tool output"


class MockAgent:
    def __init__(self, name, tools, description="Mock agent description"):
        self.name = name
        self.tools = {t.name: t for t in tools}
        self.description = description


class DummyMultiStepAgent(AsyncMultiStepAgent):
    def step(self, memory_step: ActionStep) -> Generator[None]:
        yield None

    def initialize_system_prompt(self):
        pass


class TestMultiStepAgent:
    async def test_instantiation_disables_logging_to_terminal(self):
        fake_model = MagicMock()
        agent = DummyMultiStepAgent(tools=[], model=fake_model)
        assert agent.logger.level == -1, "logging to terminal should be disabled for testing using a fixture"

    async def test_instantiation_with_prompt_templates(self, prompt_templates):
        agent = DummyMultiStepAgent(tools=[], model=MagicMock(), prompt_templates=prompt_templates)
        assert agent.prompt_templates == prompt_templates
        assert agent.prompt_templates["system_prompt"] == "This is a test system prompt."
        assert "managed_agent" in agent.prompt_templates
        assert agent.prompt_templates["managed_agent"]["task"] == "Task for {{name}}: {{task}}"
        assert agent.prompt_templates["managed_agent"]["report"] == "Report for {{name}}: {{final_answer}}"

    @pytest.mark.parametrize(
        "tools, expected_final_answer_tool",
        [([], FinalAnswerTool), ([CustomFinalAnswerTool()], CustomFinalAnswerTool)],
    )
    async def test_instantiation_with_final_answer_tool(self, tools, expected_final_answer_tool):
        agent = DummyMultiStepAgent(tools=tools, model=MagicMock())
        assert "final_answer" in agent.tools
        assert isinstance(agent.tools["final_answer"], expected_final_answer_tool)

    async def test_logs_display_thoughts_even_if_error(self):
        class FakeJsonModelNoCall(AsyncModel):
            async def generate(self, messages, stop_sequences=None, tools_to_call_from=None):
                return ChatMessage(
                    role="assistant",
                    content="""I don't want to call tools today""",
                    tool_calls=None,
                    raw="""I don't want to call tools today""",
                )

        agent_toolcalling = AsyncToolCallingAgent(model=FakeJsonModelNoCall(), tools=[], max_steps=1, verbosity_level=10)
        with agent_toolcalling.logger.console.capture() as capture:
            agent_toolcalling.run("Dummy task")
        assert "don't" in capture.get() and "want" in capture.get()

        class FakeCodeModelNoCall(AsyncModel):
            async def generate(self, messages, stop_sequences=None):
                return ChatMessage(
                    role="assistant",
                    content="""I don't want to write an action today""",
                )

        agent_code = AsyncCodeAgent(model=FakeCodeModelNoCall(), tools=[], max_steps=1, verbosity_level=10)
        with agent_code.logger.console.capture() as capture:
            agent_code.run("Dummy task")
        assert "don't" in capture.get() and "want" in capture.get()

    async def test_step_number(self):
        fake_model = MagicMock()
        fake_model.last_input_token_count = 10
        fake_model.last_output_token_count = 20
        max_steps = 2
        agent = AsyncCodeAgent(tools=[], model=fake_model, max_steps=max_steps)
        assert hasattr(agent, "step_number"), "step_number attribute should be defined"
        assert agent.step_number == 0, "step_number should be initialized to 0"
        await agent.run("Test task")
        assert hasattr(agent, "step_number"), "step_number attribute should be defined"
        assert agent.step_number == max_steps + 1, "step_number should be max_steps + 1 after run method is called"

    @pytest.mark.parametrize(
        "step, expected_messages_list",
        [
            (
                1,
                [
                    [{"role": MessageRole.USER, "content": [{"type": "text", "text": "INITIAL_PLAN_USER_PROMPT"}]}],
                ],
            ),
            (
                2,
                [
                    [
                        {
                            "role": MessageRole.SYSTEM,
                            "content": [{"type": "text", "text": "UPDATE_PLAN_SYSTEM_PROMPT"}],
                        },
                        {"role": MessageRole.USER, "content": [{"type": "text", "text": "UPDATE_PLAN_USER_PROMPT"}]},
                    ],
                ],
            ),
        ],
    )
    async def test_planning_step(self, step, expected_messages_list):
        fake_model = MagicMock()
        agent = AsyncCodeAgent(
            tools=[],
            model=fake_model,
        )
        task = "Test task"

        planning_step = list(agent._generate_planning_step(task, is_first_step=(step == 1), step=step))[-1]
        expected_message_texts = {
            "INITIAL_PLAN_USER_PROMPT": populate_template(
                agent.prompt_templates["planning"]["initial_plan"],
                variables=dict(
                    task=task,
                    tools=agent.tools,
                    managed_agents=agent.managed_agents,
                    answer_facts=planning_step.model_output_message.content,
                ),
            ),
            "UPDATE_PLAN_SYSTEM_PROMPT": populate_template(
                agent.prompt_templates["planning"]["update_plan_pre_messages"], variables=dict(task=task)
            ),
            "UPDATE_PLAN_USER_PROMPT": populate_template(
                agent.prompt_templates["planning"]["update_plan_post_messages"],
                variables=dict(
                    task=task,
                    tools=agent.tools,
                    managed_agents=agent.managed_agents,
                    facts_update=planning_step.model_output_message.content,
                    remaining_steps=agent.max_steps - step,
                ),
            ),
        }
        for expected_messages in expected_messages_list:
            for expected_message in expected_messages:
                for expected_content in expected_message["content"]:
                    expected_content["text"] = expected_message_texts[expected_content["text"]]
        assert isinstance(planning_step, PlanningStep)
        expected_model_input_messages = expected_messages_list[0]
        model_input_messages = planning_step.model_input_messages
        assert isinstance(model_input_messages, list)
        assert len(model_input_messages) == len(expected_model_input_messages)  # 2
        for message, expected_message in zip(model_input_messages, expected_model_input_messages):
            assert isinstance(message, dict)
            assert "role" in message
            assert "content" in message
            assert message["role"] in MessageRole.__members__.values()
            assert message["role"] == expected_message["role"]
            assert isinstance(message["content"], list)
            assert len(message["content"]) == 1
            for content, expected_content in zip(message["content"], expected_message["content"]):
                assert content == expected_content
        # Test calls to model
        assert len(fake_model.generate.call_args_list) == 1
        for call_args, expected_messages in zip(fake_model.generate.call_args_list, expected_messages_list):
            assert len(call_args.args) == 1
            messages = call_args.args[0]
            assert isinstance(messages, list)
            assert len(messages) == len(expected_messages)
            for message, expected_message in zip(messages, expected_messages):
                assert isinstance(message, dict)
                assert "role" in message
                assert "content" in message
                assert message["role"] in MessageRole.__members__.values()
                assert message["role"] == expected_message["role"]
                assert isinstance(message["content"], list)
                assert len(message["content"]) == 1
                for content, expected_content in zip(message["content"], expected_message["content"]):
                    assert content == expected_content

    @pytest.mark.parametrize(
        "images, expected_messages_list",
        [
            (
                None,
                [
                    [
                        {
                            "role": MessageRole.SYSTEM,
                            "content": [{"type": "text", "text": "FINAL_ANSWER_SYSTEM_PROMPT"}],
                        },
                        {"role": MessageRole.USER, "content": [{"type": "text", "text": "FINAL_ANSWER_USER_PROMPT"}]},
                    ]
                ],
            ),
            (
                ["image1.png"],
                [
                    [
                        {
                            "role": MessageRole.SYSTEM,
                            "content": [{"type": "text", "text": "FINAL_ANSWER_SYSTEM_PROMPT"}, {"type": "image"}],
                        },
                        {"role": MessageRole.USER, "content": [{"type": "text", "text": "FINAL_ANSWER_USER_PROMPT"}]},
                    ]
                ],
            ),
        ],
    )
    async def test_provide_final_answer(self, images, expected_messages_list):
        fake_model = MagicMock()
        fake_model.return_value.content = "Final answer."
        agent = AsyncCodeAgent(
            tools=[],
            model=fake_model,
        )
        task = "Test task"
        final_answer = agent.provide_final_answer(task, images=images)
        expected_message_texts = {
            "FINAL_ANSWER_SYSTEM_PROMPT": agent.prompt_templates["final_answer"]["pre_messages"],
            "FINAL_ANSWER_USER_PROMPT": populate_template(
                agent.prompt_templates["final_answer"]["post_messages"], variables=dict(task=task)
            ),
        }
        for expected_messages in expected_messages_list:
            for expected_message in expected_messages:
                for expected_content in expected_message["content"]:
                    if "text" in expected_content:
                        expected_content["text"] = expected_message_texts[expected_content["text"]]
        assert final_answer == "Final answer."
        # Test calls to model
        assert len(fake_model.call_args_list) == 1
        for call_args, expected_messages in zip(fake_model.call_args_list, expected_messages_list):
            assert len(call_args.args) == 1
            messages = call_args.args[0]
            assert isinstance(messages, list)
            assert len(messages) == len(expected_messages)
            for message, expected_message in zip(messages, expected_messages):
                assert isinstance(message, dict)
                assert "role" in message
                assert "content" in message
                assert message["role"] in MessageRole.__members__.values()
                assert message["role"] == expected_message["role"]
                assert isinstance(message["content"], list)
                assert len(message["content"]) == len(expected_message["content"])
                for content, expected_content in zip(message["content"], expected_message["content"]):
                    assert content == expected_content

    async def test_interrupt(self):
        fake_model = MagicMock()
        fake_model.return_value.content = "Model output."
        fake_model.last_input_token_count = None

        def interrupt_callback(memory_step, agent):
            agent.interrupt()

        agent = AsyncCodeAgent(
            tools=[],
            model=fake_model,
            step_callbacks=[interrupt_callback],
        )
        with pytest.raises(AgentError) as e:
            await agent.run("Test task")
        assert "Agent interrupted" in str(e)

    @pytest.mark.parametrize(
        "tools, managed_agents, name, expectation",
        [
            # Valid case: no duplicates
            (
                [MockTool("tool1"), MockTool("tool2")],
                [MockAgent("agent1", [MockTool("tool3")])],
                "test_agent",
                does_not_raise(),
            ),
            # Invalid case: duplicate tool names
            ([MockTool("tool1"), MockTool("tool1")], [], "test_agent", pytest.raises(ValueError)),
            # Invalid case: tool name same as managed agent name
            (
                [MockTool("tool1")],
                [MockAgent("tool1", [MockTool("final_answer")])],
                "test_agent",
                pytest.raises(ValueError),
            ),
            # Valid case: tool name same as managed agent's tool name
            ([MockTool("tool1")], [MockAgent("agent1", [MockTool("tool1")])], "test_agent", does_not_raise()),
            # Invalid case: duplicate managed agent name and managed agent tool name
            ([MockTool("tool1")], [], "tool1", pytest.raises(ValueError)),
            # Valid case: duplicate tool names across managed agents
            (
                [MockTool("tool1")],
                [
                    MockAgent("agent1", [MockTool("tool2"), MockTool("final_answer")]),
                    MockAgent("agent2", [MockTool("tool2"), MockTool("final_answer")]),
                ],
                "test_agent",
                does_not_raise(),
            ),
        ],
    )
    async def test_validate_tools_and_managed_agents(self, tools, managed_agents, name, expectation):
        fake_model = MagicMock()
        with expectation:
            DummyMultiStepAgent(
                tools=tools,
                model=fake_model,
                name=name,
                managed_agents=managed_agents,
            )

    async def test_from_dict(self):
        # Create a test agent dictionary
        agent_dict = {
            "model": {"class": "TransformersModel", "data": {"model_id": "test/model"}},
            "tools": [
                {
                    "name": "valid_tool_function",
                    "code": 'from smolagents import Tool\nfrom typing import Any, Optional\n\nclass SimpleTool(Tool):\n    name = "valid_tool_function"\n    description = "A valid tool function."\n    inputs = {"input":{"type":"string","description":"Input string."}}\n    output_type = "string"\n\n    def forward(self, input: str) -> str:\n        """A valid tool function.\n\n        Args:\n            input (str): Input string.\n        """\n        return input.upper()',
                    "requirements": {"smolagents"},
                }
            ],
            "managed_agents": {},
            "prompt_templates": EMPTY_PROMPT_TEMPLATES,
            "max_steps": 15,
            "verbosity_level": 2,
            "grammar": {"test": "grammar"},
            "planning_interval": 3,
            "name": "test_agent",
            "description": "Test agent description",
        }

        # Call from_dict
        with patch("smolagents.models.TransformersModel") as mock_model_class:
            mock_model_instance = mock_model_class.from_dict.return_value
            agent = DummyMultiStepAgent.from_dict(agent_dict)

        # Verify the agent was created correctly
        assert agent.model == mock_model_instance
        assert mock_model_class.from_dict.call_args.args[0] == {"model_id": "test/model"}
        assert agent.max_steps == 15
        assert agent.logger.level == 2
        assert agent.grammar == {"test": "grammar"}
        assert agent.planning_interval == 3
        assert agent.name == "test_agent"
        assert agent.description == "Test agent description"
        # Verify the tool was created correctly
        assert sorted(agent.tools.keys()) == ["final_answer", "valid_tool_function"]
        assert agent.tools["valid_tool_function"].name == "valid_tool_function"
        assert agent.tools["valid_tool_function"].description == "A valid tool function."
        assert agent.tools["valid_tool_function"].inputs == {
            "input": {"type": "string", "description": "Input string."}
        }
        assert agent.tools["valid_tool_function"].output_type == "string"
        assert agent.tools["valid_tool_function"]("test") == "TEST"

        # Test overriding with kwargs
        with patch("smolagents.models.TransformersModel") as mock_model_class:
            agent = DummyMultiStepAgent.from_dict(agent_dict, max_steps=30)
        assert agent.max_steps == 30


class TestToolCallingAgent:


    async def test_change_tools_after_init(self):
        from smolagents import tool

        @tool
        def fake_tool_1() -> str:
            """Fake tool"""
            return "1"

        @tool
        def fake_tool_2() -> str:
            """Fake tool"""
            return "2"

        class FakeCodeModel(AsyncModel):
            async def generate(self, messages, tools_to_call_from=None, stop_sequences=None, grammar=None, **kwargs):
                if len(messages) < 3:
                    return ChatMessage(
                        role="assistant",
                        content="",
                        tool_calls=[
                            ChatMessageToolCall(
                                id="call_0",
                                type="function",
                                function=ChatMessageToolCallDefinition(name="fake_tool_1", arguments={}),
                            )
                        ],
                    )
                else:
                    tool_result = messages[-1]["content"][0]["text"].removeprefix("Observation:\n")
                    return ChatMessage(
                        role="assistant",
                        content="",
                        tool_calls=[
                            ChatMessageToolCall(
                                id="call_1",
                                type="function",
                                function=ChatMessageToolCallDefinition(
                                    name="final_answer", arguments={"answer": tool_result}
                                ),
                            )
                        ],
                    )

        agent = AsyncToolCallingAgent(tools=[fake_tool_1], model=FakeCodeModel())

        agent.tools["final_answer"] = CustomFinalAnswerTool()
        agent.tools["fake_tool_1"] = fake_tool_2

        answer = await agent.run("Fake task.")
        assert answer == "2CUSTOM"


class TestCodeAgent:
    @pytest.mark.parametrize("provide_run_summary", [False, True])
    async def test_call_with_provide_run_summary(self, provide_run_summary):
        agent = AsyncCodeAgent(tools=[], model=MagicMock(), provide_run_summary=provide_run_summary)
        assert agent.provide_run_summary is provide_run_summary
        agent.managed_agent_prompt = "Task: {task}"
        agent.name = "test_agent"
        agent.run = MagicMock(return_value="Test output")
        agent.write_memory_to_messages = MagicMock(return_value=[{"content": "Test summary"}])

        result = agent("Test request")
        expected_summary = "Here is the final answer from your managed agent 'test_agent':\nTest output"
        if provide_run_summary:
            expected_summary += (
                "\n\nFor more detail, find below a summary of this agent's work:\n"
                "<summary_of_work>\n\nTest summary\n---\n</summary_of_work>"
            )
        assert result == expected_summary

    async def test_errors_logging(self):
        class FakeCodeModel(AsyncModel):
            async def generate(self, messages, stop_sequences=None, grammar=None, **kwargs):
                return ChatMessage(role="assistant", content="Code:\n```py\nsecret=3;['1', '2'][secret]\n```")

        agent = AsyncCodeAgent(tools=[], model=FakeCodeModel(), verbosity_level=1)

        with agent.logger.console.capture() as capture:
            await agent.run("Test request")
        assert "secret\\\\" in repr(capture.get())

    async def test_missing_import_triggers_advice_in_error_log(self):
        # Set explicit verbosity level to 1 to override the default verbosity level of -1 set in CI fixture
        agent = AsyncCodeAgent(tools=[], model=FakeCodeModelImport(), verbosity_level=1)

        with agent.logger.console.capture() as capture:
            await agent.run("Count to 3")
        str_output = capture.get()
        assert "`additional_authorized_imports`" in str_output.replace("\n", "")

    async def test_errors_show_offending_line_and_error(self):
        agent = AsyncCodeAgent(tools=[PythonInterpreterTool()], model=FakeCodeModelError())
        output = await agent.run("What is 2 multiplied by 3.6452?")
        assert isinstance(output, AgentText)
        assert output == "got an error"
        assert "Code execution failed at line 'error_function()'" in str(agent.memory.steps[1].error)
        assert "ValueError" in str(agent.memory.steps)

    async def test_error_saves_previous_print_outputs(self):
        agent = AsyncCodeAgent(tools=[PythonInterpreterTool()], model=FakeCodeModelError(), verbosity_level=10)
        await agent.run("What is 2 multiplied by 3.6452?")
        assert "Flag!" in str(agent.memory.steps[1].observations)

    async def test_syntax_error_show_offending_lines(self):
        agent = AsyncCodeAgent(tools=[PythonInterpreterTool()], model=FakeCodeModelSyntaxError())
        output = await agent.run("What is 2 multiplied by 3.6452?")
        assert isinstance(output, AgentText)
        assert output == "got an error"
        assert '    print("Failing due to unexpected indent")' in str(agent.memory.steps)

    async def test_end_code_appending(self):
        # Checking original output message
        orig_output = await FakeCodeModelNoReturn().generate([])
        assert not orig_output.content.endswith("<end_code>")

        # Checking the step output
        agent = AsyncCodeAgent(
            tools=[PythonInterpreterTool()],
            model=FakeCodeModelNoReturn(),
            max_steps=1,
        )
        answer = await agent.run("What is 2 multiplied by 3.6452?")
        assert answer

        memory_steps = agent.memory.steps
        actions_steps = [s for s in memory_steps if isinstance(s, ActionStep)]

        outputs = [s.model_output for s in actions_steps if s.model_output]
        assert outputs
        assert all(o.endswith("<end_code>") for o in outputs)

        messages = [s.model_output_message for s in actions_steps if s.model_output_message]
        assert messages
        assert all(m.content.endswith("<end_code>") for m in messages)

    async def test_change_tools_after_init(self):
        from smolagents import tool

        @tool
        def fake_tool_1() -> str:
            """Fake tool"""
            return "1"

        @tool
        def fake_tool_2() -> str:
            """Fake tool"""
            return "2"

        class FakeCodeModel(AsyncModel):
            async def generate(self, messages, stop_sequences=None, grammar=None, **kwargs):
                return ChatMessage(role="assistant", content="Code:\n```py\nfinal_answer(fake_tool_1())\n```")

        agent = AsyncCodeAgent(tools=[fake_tool_1], model=FakeCodeModel())

        agent.tools["final_answer"] = CustomFinalAnswerTool()
        agent.tools["fake_tool_1"] = fake_tool_2

        answer = await agent.run("Fake task.")
        assert answer == "2CUSTOM"

    async def test_local_python_executor_with_custom_functions(self):
        model = MagicMock()
        model.last_input_token_count = 10
        model.last_output_token_count = 5
        agent = AsyncCodeAgent(tools=[], model=model)
        agent.python_executor.additional_functions = {"open": open}
        await agent.run("Test run")
        assert "open" in agent.python_executor.static_tools



class TestMultiAgents:
    async def test_multiagents(self):
        class FakeModelMultiagentsManagerAgent(AsyncModel):
            model_id = "fake_model"

            async def generate(
                self,
                messages,
                stop_sequences=None,
                grammar=None,
                tools_to_call_from=None,
                **kwargs,
            ):
                if tools_to_call_from is not None:
                    if len(messages) < 3:
                        return ChatMessage(
                            role="assistant",
                            content="",
                            tool_calls=[
                                ChatMessageToolCall(
                                    id="call_0",
                                    type="function",
                                    function=ChatMessageToolCallDefinition(
                                        name="search_agent",
                                        arguments="Who is the current US president?",
                                    ),
                                )
                            ],
                        )
                    else:
                        assert "Report on the current US president" in str(messages)
                        return ChatMessage(
                            role="assistant",
                            content="",
                            tool_calls=[
                                ChatMessageToolCall(
                                    id="call_0",
                                    type="function",
                                    function=ChatMessageToolCallDefinition(
                                        name="final_answer", arguments="Final report."
                                    ),
                                )
                            ],
                        )
                else:
                    if len(messages) < 3:
                        return ChatMessage(
                            role="assistant",
                            content="""
Thought: Let's call our search agent.
Code:
```py
result = search_agent("Who is the current US president?")
```<end_code>
""",
                        )
                    else:
                        assert "Report on the current US president" in str(messages)
                        return ChatMessage(
                            role="assistant",
                            content="""
Thought: Let's return the report.
Code:
```py
final_answer("Final report.")
```<end_code>
""",
                        )

        manager_model = FakeModelMultiagentsManagerAgent()

        class FakeModelMultiagentsManagedAgent(AsyncModel):
            model_id = "fake_model"

            async def generate(
                self,
                messages,
                tools_to_call_from=None,
                stop_sequences=None,
                grammar=None,
                **kwargs,
            ):
                return ChatMessage(
                    role="assistant",
                    content="Here is the secret content: FLAG1",
                    tool_calls=[
                        ChatMessageToolCall(
                            id="call_0",
                            type="function",
                            function=ChatMessageToolCallDefinition(
                                name="final_answer",
                                arguments="Report on the current US president",
                            ),
                        )
                    ],
                )

        managed_model = FakeModelMultiagentsManagedAgent()

        web_agent = AsyncToolCallingAgent(
            tools=[],
            model=managed_model,
            max_steps=10,
            name="search_agent",
            description="Runs web searches for you. Give it your request as an argument. Make the request as detailed as needed, you can ask for thorough reports",
            verbosity_level=2,
        )

        manager_code_agent = AsyncCodeAgent(
            tools=[],
            model=manager_model,
            managed_agents=[web_agent],
            additional_authorized_imports=["time", "numpy", "pandas"],
        )

        report = await manager_code_agent.run("Fake question.")
        assert report == "Final report."

        manager_toolcalling_agent = AsyncToolCallingAgent(
            tools=[],
            model=manager_model,
            managed_agents=[web_agent],
        )

        with web_agent.logger.console.capture() as capture:
            report = await manager_toolcalling_agent.run("Fake question.")
        assert report == "Final report."
        assert "FLAG1" in capture.get()  # Check that managed agent's output is properly logged

        # Test that visualization works
        with manager_toolcalling_agent.logger.console.capture() as capture:
            manager_toolcalling_agent.visualize()
        assert "├──" in capture.get()


@pytest.fixture
def prompt_templates():
    return {
        "system_prompt": "This is a test system prompt.",
        "managed_agent": {"task": "Task for {{name}}: {{task}}", "report": "Report for {{name}}: {{final_answer}}"},
        "planning": {
            "initial_plan": "The plan.",
            "update_plan_pre_messages": "custom",
            "update_plan_post_messages": "custom",
        },
        "final_answer": {"pre_messages": "custom", "post_messages": "custom"},
    }


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"arg": "bar"},
        {None: None},
        [1, 2, 3],
    ],
)

async def test_tool_calling_agents_raises_tool_call_error_being_invoked_with_wrong_arguments(arguments):
    @tool
    def _sample_tool(prompt: str) -> str:
        """Tool that returns same string

        Args:
            prompt: The string to return
        Returns:
            The same string
        """

        return prompt

    agent = AsyncToolCallingAgent(model=FakeToolCallModel(), tools=[_sample_tool])
    with pytest.raises(AgentToolCallError):
        agent.execute_tool_call(_sample_tool.name, arguments)


async def test_tool_calling_agents_raises_agent_execution_error_when_tool_raises():
    @tool
    def _sample_tool(_: str) -> float:
        """Tool that fails

        Args:
            _: The pointless string
        Returns:
            Some number
        """

        return 1 / 0

    agent = AsyncToolCallingAgent(model=FakeToolCallModel(), tools=[_sample_tool])
    with pytest.raises(AgentExecutionError):
        agent.execute_tool_call(_sample_tool.name, "sample")
