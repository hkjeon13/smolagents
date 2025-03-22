#!/usr/bin/env python
# coding=utf-8

import asyncio
import json
from enum import IntEnum
from typing import List, Optional

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.tree import Tree
from .utils import escape_code_brackets

__all__ = ["AsyncAgentLogger", "LogLevel", "AsyncMonitor"]

YELLOW_HEX = "#d4b702"


class LogLevel(IntEnum):
    OFF = -1  # No output
    ERROR = 0  # Only errors
    INFO = 1  # Normal output (default)
    DEBUG = 2  # Detailed output


class AsyncMonitor:
    def __init__(self, tracked_model, logger):
        self.step_durations = []
        self.tracked_model = tracked_model
        self.logger = logger
        if getattr(self.tracked_model, "last_input_token_count", "Not found") != "Not found":
            self.total_input_token_count = 0
            self.total_output_token_count = 0

    def get_total_token_counts(self):
        return {
            "input": self.total_input_token_count,
            "output": self.total_output_token_count,
        }

    def reset(self):
        self.step_durations = []
        self.total_input_token_count = 0
        self.total_output_token_count = 0

    async def update_metrics(self, step_log):
        """비동기로 모니터의 메트릭을 업데이트합니다.

        Args:
            step_log: Step log로, duration, input/output token 정보 포함.
        """
        step_duration = step_log.duration
        self.step_durations.append(step_duration)
        console_outputs = f"[Step {len(self.step_durations)}: Duration {step_duration:.2f} seconds"
        if getattr(self.tracked_model, "last_input_token_count", None) is not None:
            self.total_input_token_count += self.tracked_model.last_input_token_count
            self.total_output_token_count += self.tracked_model.last_output_token_count
            console_outputs += (
                f" | Input tokens: {self.total_input_token_count:,} | Output tokens: {self.total_output_token_count:,}"
            )
        console_outputs += "]"
        await self.logger.log(Text(console_outputs, style="dim"), level=LogLevel.INFO)


class AsyncAgentLogger:
    def __init__(self, level: LogLevel = LogLevel.INFO):
        self.level = level
        self.console = Console()
        self.log_queue = asyncio.Queue()
        # 백그라운드로 로그를 출력하는 작업을 시작합니다.
        self.worker_task = asyncio.create_task(self._log_worker())

    async def _log_worker(self):
        while True:
            args, kwargs = await self.log_queue.get()
            self.console.print(*args, **kwargs)
            self.log_queue.task_done()

    async def log(self, *args, level: str | LogLevel = LogLevel.INFO, **kwargs) -> None:
        """비동기 방식으로 메시지를 로그에 기록합니다.

        Args:
            level (LogLevel, optional): 로그 레벨. 기본값은 LogLevel.INFO.
        """
        if isinstance(level, str):
            level = LogLevel[level.upper()]
        if level <= self.level:
            await self.log_queue.put((args, kwargs))

    async def log_error(self, error_message: str) -> None:
        await self.log(escape_code_brackets(error_message), style="bold red", level=LogLevel.ERROR)

    async def log_markdown(
            self, content: str, title: Optional[str] = None, level=LogLevel.INFO, style=YELLOW_HEX
    ) -> None:
        markdown_content = Syntax(
            content,
            lexer="markdown",
            theme="github-dark",
            word_wrap=True,
        )
        if title:
            await self.log(
                Group(
                    Rule("[bold italic]" + title, align="left", style=style),
                    markdown_content,
                ),
                level=level,
            )
        else:
            await self.log(markdown_content, level=level)

    async def log_code(self, title: str, content: str, level: int = LogLevel.INFO) -> None:
        await self.log(
            Panel(
                Syntax(
                    content,
                    lexer="python",
                    theme="monokai",
                    word_wrap=True,
                ),
                title="[bold]" + title,
                title_align="left",
                box=box.HORIZONTALS,
            ),
            level=level,
        )

    async def log_rule(self, title: str, level: int = LogLevel.INFO) -> None:
        await self.log(
            Rule(
                "[bold]" + title,
                characters="━",
                style=YELLOW_HEX,
            ),
            level=level,
        )

    async def log_task(
            self, content: str, subtitle: str, title: Optional[str] = None, level: int = LogLevel.INFO
    ) -> None:
        await self.log(
            Panel(
                f"\n[bold]{escape_code_brackets(content)}\n",
                title="[bold]New run" + (f" - {title}" if title else ""),
                subtitle=subtitle,
                border_style=YELLOW_HEX,
                subtitle_align="left",
            ),
            level=level,
        )

    async def log_messages(self, messages: List) -> None:
        messages_as_string = "\n".join([json.dumps(dict(message), indent=4) for message in messages])
        await self.log(
            Syntax(
                messages_as_string,
                lexer="markdown",
                theme="github-dark",
                word_wrap=True,
            )
        )

    async def visualize_agent_tree(self, agent):
        def create_tools_section(tools_dict):
            table = Table(show_header=True, header_style="bold")
            table.add_column("Name", style="#1E90FF")
            table.add_column("Description")
            table.add_column("Arguments")
            for name, tool in tools_dict.items():
                args = [
                    f"{arg_name} (`{info.get('type', 'Any')}`{', optional' if info.get('optional') else ''}): {info.get('description', '')}"
                    for arg_name, info in getattr(tool, "inputs", {}).items()
                ]
                table.add_row(name, getattr(tool, "description", str(tool)), "\n".join(args))
            return Group("🛠️ [italic #1E90FF]Tools:[/italic #1E90FF]", table)

        def get_agent_headline(agent, name: Optional[str] = None):
            name_headline = f"{name} | " if name else ""
            return f"[bold {YELLOW_HEX}]{name_headline}{agent.__class__.__name__} | {agent.model.model_id}"

        def build_agent_tree(parent_tree, agent_obj):
            parent_tree.add(create_tools_section(agent_obj.tools))
            if agent_obj.managed_agents:
                agents_branch = parent_tree.add("🤖 [italic #1E90FF]Managed agents:")
                for name, managed_agent in agent_obj.managed_agents.items():
                    agent_tree = agents_branch.add(get_agent_headline(managed_agent, name))
                    if managed_agent.__class__.__name__ == "CodeAgent":
                        agent_tree.add(
                            f"✅ [italic #1E90FF]Authorized imports:[/italic #1E90FF] {managed_agent.additional_authorized_imports}"
                        )
                    agent_tree.add(f"📝 [italic #1E90FF]Description:[/italic #1E90FF] {managed_agent.description}")
                    build_agent_tree(agent_tree, managed_agent)

        main_tree = Tree(get_agent_headline(agent))
        if agent.__class__.__name__ == "CodeAgent":
            main_tree.add(
                f"✅ [italic #1E90FF]Authorized imports:[/italic #1E90FF] {agent.additional_authorized_imports}"
            )
        build_agent_tree(main_tree, agent)
        await self.log(main_tree)
