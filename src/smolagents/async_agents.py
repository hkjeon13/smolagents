
from typing import List, Optional, Dict, Generator, Union, AsyncGenerator
from .memory import ActionStep
from .agents import MultiStepAgent
from .agent_types import AgentType


class AsyncMultiStepAgentBase:
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
    ) -> AsyncGenerator[ActionStep|AgentType, None, None]:
        raise NotImplementedError

    async _execute_step(self, task:str, memory_step: ActionStep) -> Union[None, Any]:
        raise NotImplementedError

    async def _finalize_step(self, memory_step: ActionStep, step_start_time: float):
        raise NotImplementedError

    async def _handle_max_steps_reached(self, task: str, images: List[str], step_start_time: float) -> Any:
        raise NotImplementedError

    async planning_step(self, task, is_first_step: bool, step: int) -> None:
        raise NotImplementedError

    async def _generate_initial_plan(
        task: str
    ) -> tuple[list[dict[str, list[dict[str, str]] | MessageRole]], ChatMessage, ChatMessage]:
        raise NotImplementedError

    async def _generate_updated_plan(
            self,
            task: str,
            step: int
    ) -> tuple[list[dict[str, str] | dict[str, list[dict[str, str]] | MessageRole]], ChatMessage, ChatMessage]:
        raise NotImplementedError

    async def provide_final_answer(self, task: str, images: Optional[list[str]]) -> str:
        raise NotImplementedError

    async def step(self, memory_step: ActionStep) -> Union[None, Any]:
        raise NotImplementedError

    async def __call__(task:str, **kwargs):
        raise NotImplementedError


class AsyncMultiStepAgent(AsyncMultiStepAgentBase, MultiStepAgent):
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

        self.logger.log_task(
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




