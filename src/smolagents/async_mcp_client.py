#!/usr/bin/env python
# coding=utf-8

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import asyncio
import jsonref  # type: ignore
import keyword
import logging
import re
from inspect import iscoroutinefunction
from types import TracebackType
from typing import TYPE_CHECKING, Any, Callable, Coroutine

import mcp
import smolagents  # type: ignore
from mcpadapt.core import MCPAdapt, ToolAdapter
from smolagents.tools import Tool

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from mcpadapt.core import StdioServerParameters


def _sanitize_function_name(name: str) -> str:
    name = name.replace("-", "_")
    name = re.sub(r"[^\w_]", "", name)
    if name[0].isdigit():
        name = f"_{name}"
    if keyword.iskeyword(name):
        name = f"{name}_"
    return name


class SmolAgentsAdapter(ToolAdapter):
    """Adapter for the `smolagents` framework.

    Note that the `smolagents` framework does not support async tools directly,
    so this adapter only implements the sync `adapt` method.
    """

    def adapt(
        self,
        func: Callable[[dict | None], mcp.types.CallToolResult],
        mcp_tool: mcp.types.Tool,
    ) -> smolagents.Tool:
        class MCPAdaptTool(smolagents.Tool):
            def __init__(
                self,
                name: str,
                description: str,
                inputs: dict[str, dict[str, str]],
                output_type: str,
            ):
                self.name = _sanitize_function_name(name)
                self.description = description
                self.inputs = inputs
                self.output_type = output_type
                self.is_initialized = True
                self.skip_forward_signature_validation = True

            def forward(self, *args, **kwargs) -> str:
                if len(args) > 0:
                    if len(args) == 1 and isinstance(args[0], dict) and not kwargs:
                        if iscoroutinefunction(func):
                            future = asyncio.run_coroutine_threadsafe(func(args[0]), asyncio.get_event_loop())
                            mcp_output = future.result()
                        else:
                            mcp_output = func(args[0])
                    else:
                        raise ValueError(f"tool {self.name} does not support multiple positional arguments or combined positional and keyword arguments")
                else:
                    if iscoroutinefunction(func):
                        future = asyncio.run_coroutine_threadsafe(func(kwargs), asyncio.get_event_loop())
                        mcp_output = future.result()
                    else:
                        mcp_output = func(kwargs)

                if len(mcp_output.content) == 0:
                    raise ValueError(f"tool {self.name} returned empty content")

                if len(mcp_output.content) > 1:
                    logger.warning(f"tool {self.name} returned multiple content items, using the first")

                if not isinstance(mcp_output.content[0], mcp.types.TextContent):
                    raise ValueError(f"tool {self.name} returned a non-text content: `{type(mcp_output.content[0])}`")

                return mcp_output.content[0].text  # type: ignore

        input_schema = {
            k: v for k, v in jsonref.replace_refs(mcp_tool.inputSchema).items() if k != "$defs"
        }

        for k, v in input_schema["properties"].items():
            v.setdefault("description", "see tool description")
            v.setdefault("type", "string")

        return MCPAdaptTool(
            name=mcp_tool.name,
            description=mcp_tool.description or "",
            inputs=input_schema["properties"],
            output_type="string",
        )

    async def async_adapt(
        self,
        afunc: Callable[[dict | None], Coroutine[Any, Any, mcp.types.CallToolResult]],
        mcp_tool: mcp.types.Tool,
    ) -> smolagents.Tool:
        return self.adapt(afunc, mcp_tool)


class AsyncMCPClient:
    """Manages connection to an MCP server and exposes tools for SmolAgents."""

    def __init__(
        self,
        server_parameters: "StdioServerParameters" | dict[str, Any] | list["StdioServerParameters" | dict[str, Any]],
    ):
        self._adapter = MCPAdapt(server_parameters, SmolAgentsAdapter())
        self._tools: list[Tool] | None = None

    def connect(self):
        self._tools = self._adapter.__enter__()

    def disconnect(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        exc_traceback: TracebackType | None = None,
    ):
        self._adapter.__exit__(exc_type, exc_value, exc_traceback)

    async def aconnect(self):
        self._tools = await self._adapter.__aenter__()

    async def adisconnect(self, exc_type=None, exc_value=None, exc_traceback=None):
        await self._adapter.__aexit__(exc_type, exc_value, exc_traceback)

    def get_tools(self) -> list[Tool]:
        if self._tools is None:
            raise ValueError("Run `connect()` or `aconnect()` before accessing tools.")
        return self._tools

    def __enter__(self) -> list[Tool]:
        self.connect()
        return self._tools

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.disconnect(exc_type, exc_value, exc_traceback)

    async def __aenter__(self) -> list[Tool]:
        await self.aconnect()
        return self._tools

    async def __aexit__(self, exc_type, exc_value, exc_traceback):
        await self.adisconnect(exc_type, exc_value, exc_traceback)