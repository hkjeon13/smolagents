#!/usr/bin/env python
# coding=utf-8

import asyncio
import aiohttp

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .local_python_executor import (
    BASE_BUILTIN_MODULES,
    BASE_PYTHON_TOOLS,
    evaluate_python_code,
)
from .async_tools import AsyncPipelineTool, AsyncTool


@dataclass
class PreTool:
    name: str
    inputs: Dict[str, str]
    output_type: type
    task: str
    description: str
    repo_id: str


class AsyncPythonInterpreterTool(AsyncTool):
    name = "async_python_interpreter"
    description = "This is a tool that evaluates python code. It can be used to perform calculations."
    inputs = {
        "code": {
            "type": "string",
            "description": "The python code to run in interpreter",
        }
    }
    output_type = "string"

    def __init__(self, *args, authorized_imports=None, **kwargs):
        if authorized_imports is None:
            self.authorized_imports = list(set(BASE_BUILTIN_MODULES))
        else:
            self.authorized_imports = list(set(BASE_BUILTIN_MODULES) | set(authorized_imports))
        self.inputs = {
            "code": {
                "type": "string",
                "description": (
                    "The code snippet to evaluate. All variables used in this snippet must be defined in this same snippet, "
                    f"else you will get an error. This code can only import the following python libraries: {authorized_imports}."
                ),
            }
        }
        self.base_python_tools = BASE_PYTHON_TOOLS
        self.python_evaluator = evaluate_python_code
        super().__init__(*args, **kwargs)

    async def forward(self, code: str) -> str:
        state = {}
        loop = asyncio.get_running_loop()
        # 블로킹 evaluator 호출을 별도 스레드에서 실행하여 비동기로 처리
        result = await loop.run_in_executor(
            None,
            self.python_evaluator,
            code,
            state,
            self.base_python_tools,
            self.authorized_imports,
        )
        output = str(result[0])  # 두 번째 요소는 is_final_answer (불린)임
        return f"Stdout:\n{str(state.get('_print_outputs', []))}\nOutput: {output}"


class AsyncFinalAnswerTool(AsyncTool):
    name = "async_final_answer"
    description = "Provides a final answer to the given problem."
    inputs = {"answer": {"type": "any", "description": "The final answer to the problem"}}
    output_type = "any"

    async def forward(self, answer: Any) -> Any:
        return answer


class AsyncUserInputTool(AsyncTool):
    name = "user_input"
    description = "Asks for user's input on a specific question"
    inputs = {"question": {"type": "string", "description": "The question to ask the user"}}
    output_type = "string"

    async def forward(self, question):
        loop = asyncio.get_running_loop()
        # input()는 기본적으로 블로킹이므로 run_in_executor로 감싸 비동기 처리
        user_input = await loop.run_in_executor(None, input, f"{question} => Type your answer here:")
        return user_input


class AsyncDuckDuckGoSearchTool(AsyncTool):
    name = "async_web_search"
    description = (
        "Performs a duckduckgo web search based on your query (think a Google search) then returns the top search results."
    )
    inputs = {"query": {"type": "string", "description": "The search query to perform."}}
    output_type = "string"

    def __init__(self, max_results=10, **kwargs):
        super().__init__()
        self.max_results = max_results
        try:
            from duckduckgo_search import DDGS
        except ImportError as e:
            raise ImportError(
                "You must install package `duckduckgo_search` to run this tool: for instance run `pip install duckduckgo-search`."
            ) from e
        self.ddgs = DDGS(**kwargs)

    async def forward(self, query: str) -> str:
        loop = asyncio.get_running_loop()
        # ddgs.text()를 별도 스레드에서 실행하여 비동기로 처리
        results = await loop.run_in_executor(None, self.ddgs.text, query, max_results=self.max_results)
        if len(results) == 0:
            raise Exception("No results found! Try a less restrictive/shorter query.")
        postprocessed_results = [f"[{result['title']}]({result['href']})\n{result['body']}" for result in results]
        return "## Search Results\n\n" + "\n\n".join(postprocessed_results)


class AsyncGoogleSearchTool(AsyncTool):
    name = "async_web_search"
    description = "Performs a google web search for your query then returns a string of the top search results."
    inputs = {
        "query": {"type": "string", "description": "The search query to perform."},
        "filter_year": {
            "type": "integer",
            "description": "Optionally restrict results to a certain year",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, provider: str = "serpapi"):
        super().__init__()
        import os

        self.provider = provider
        if provider == "serpapi":
            self.organic_key = "organic_results"
            api_key_env_name = "SERPAPI_API_KEY"
        else:
            self.organic_key = "organic"
            api_key_env_name = "SERPER_API_KEY"
        self.api_key = os.getenv(api_key_env_name)
        if self.api_key is None:
            raise ValueError(f"Missing API key. Make sure you have '{api_key_env_name}' in your env variables.")

    async def forward(self, query: str, filter_year: Optional[int] = None) -> str:
        if self.provider == "serpapi":
            params = {
                "q": query,
                "api_key": self.api_key,
                "engine": "google",
                "google_domain": "google.com",
            }
            base_url = "https://serpapi.com/search.json"
        else:
            params = {
                "q": query,
                "api_key": self.api_key,
            }
            base_url = "https://google.serper.dev/search"
        if filter_year is not None:
            params["tbs"] = f"cdr:1,cd_min:01/01/{filter_year},cd_max:12/31/{filter_year}"

        async with aiohttp.ClientSession() as session:
            async with session.get(base_url, params=params) as response:
                if response.status == 200:
                    results = await response.json()
                else:
                    error = await response.text()
                    raise ValueError(error)

        if self.organic_key not in results.keys():
            if filter_year is not None:
                raise Exception(
                    f"No results found for query: '{query}' with filtering on year={filter_year}. Use a less restrictive query or do not filter on year."
                )
            else:
                raise Exception(f"No results found for query: '{query}'. Use a less restrictive query.")
        if len(results[self.organic_key]) == 0:
            year_filter_message = f" with filter year={filter_year}" if filter_year is not None else ""
            return f"No results found for '{query}'{year_filter_message}. Try with a more general query, or remove the year filter."

        web_snippets = []
        if self.organic_key in results:
            for idx, page in enumerate(results[self.organic_key]):
                date_published = ""
                if "date" in page:
                    date_published = "\nDate published: " + page["date"]

                source = ""
                if "source" in page:
                    source = "\nSource: " + page["source"]

                snippet = ""
                if "snippet" in page:
                    snippet = "\n" + page["snippet"]

                redacted_version = f"{idx}. [{page['title']}]({page['link']}){date_published}{source}\n{snippet}"
                web_snippets.append(redacted_version)

        return "## Search Results\n" + "\n\n".join(web_snippets)


class AsyncVisitWebpageTool(AsyncTool):
    name = "async_visit_webpage"
    description = (
        "Visits a webpage at the given url and reads its content as a markdown string. Use this to browse webpages."
    )
    inputs = {
        "url": {
            "type": "string",
            "description": "The url of the webpage to visit.",
        }
    }
    output_type = "string"

    async def forward(self, url: str) -> str:
        try:
            import re
            from markdownify import markdownify
            from smolagents.utils import truncate_content
        except ImportError as e:
            raise ImportError(
                "You must install packages `markdownify` and `aiohttp` to run this tool: for instance run `pip install markdownify aiohttp`."
            ) from e
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=20) as response:
                    response.raise_for_status()
                    text = await response.text()
            markdown_content = markdownify(text).strip()
            markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)
            return truncate_content(markdown_content, 10000)
        except asyncio.TimeoutError:
            return "The request timed out. Please try again later or check the URL."
        except Exception as e:
            return f"Error fetching the webpage: {str(e)}"


class AsyncSpeechToTextTool(AsyncPipelineTool):
    default_checkpoint = "openai/whisper-large-v3-turbo"
    description = "This is a tool that transcribes an audio into text. It returns the transcribed text."
    name = "async_transcriber"
    inputs = {
        "audio": {
            "type": "audio",
            "description": "The audio to transcribe. Can be a local path, an url, or a tensor.",
        }
    }
    output_type = "string"

    def __new__(cls, *args, **kwargs):
        from transformers.models.whisper import (
            WhisperForConditionalGeneration,
            WhisperProcessor,
        )

        cls.pre_processor_class = WhisperProcessor
        cls.model_class = WhisperForConditionalGeneration
        return super().__new__(cls, *args, **kwargs)

    def encode(self, audio):
        from .agent_types import AgentAudio

        audio = AgentAudio(audio).to_raw()
        return self.pre_processor(audio, return_tensors="pt")

    async def forward(self, inputs):
        loop = asyncio.get_running_loop()
        outputs = await loop.run_in_executor(None, self.model.generate, inputs["input_features"])
        return outputs

    def decode(self, outputs):
        return self.pre_processor.batch_decode(outputs, skip_special_tokens=True)[0]


ASYNC_TOOL_MAPPING = {
    tool_class.name: tool_class
    for tool_class in [
        AsyncPythonInterpreterTool,
        AsyncDuckDuckGoSearchTool,
        AsyncVisitWebpageTool,
    ]
}

__all__ = [
    "AsyncPythonInterpreterTool",
    "AsyncFinalAnswerTool",
    "AsyncUserInputTool",
    "AsyncDuckDuckGoSearchTool",
    "AsyncGoogleSearchTool",
    "AsyncVisitWebpageTool",
    "AsyncSpeechToTextTool",
]
