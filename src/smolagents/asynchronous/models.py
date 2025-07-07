from collections.abc import AsyncGenerator
from typing import Any

from smolagents import Tool, ChatMessage
from smolagents.models import Model, ChatMessageStreamDelta
from smolagents.monitoring import TokenUsage


class AsyncModel(Model):
    """
    Base class for async models.
    """

    async def generate(
            self,
            messages: list[dict[str, str | list[dict]]],
            stop_sequences: list[str] | None = None,
            grammar: str | None = None,
            tools_to_call_from: list[Tool] | None = None,
            **kwargs,
    ) -> ChatMessage:
        """
        Generate a response from the model.

        Args:
            messages (list[dict[str, str | list[dict]]]): The messages to send to the model.
            stop_sequences (list[str] | None): The stop sequences to use.
            grammar (str | None): The grammar to use.
            tools_to_call_from (list[Tool] | None): The tools to call from.
            **kwargs: Additional arguments to pass to the model.

        Returns:
            ChatMessage: The generated response.
        """
        raise NotImplementedError("Subclasses must implement this method")

    async def __call__(self, *args, **kwargs):
        return await self.generate(*args, **kwargs)

    async def generate_stream(
            self,
            messages: list[dict[str, str | list[dict]]],
            stop_sequences: list[str] | None = None,
            grammar: str | None = None,
            tools_to_call_from: list[Tool] | None = None,
            **kwargs,
    ) -> AsyncGenerator[ChatMessageStreamDelta]:
        """
        Generate a response from the model in a streaming manner.

        Args:
            messages (list[dict[str, str | list[dict]]]): The messages to send to the model.
            stop_sequences (list[str] | None): The stop sequences to use.
            grammar (str | None): The grammar to use.
            tools_to_call_from (list[Tool] | None): The tools to call from.
            **kwargs: Additional arguments to pass to the model.

        Yields:
            ChatMessageStreamDelta: The generated response in a streaming manner.
        """
        yield "Not implemented yet"


class AsyncApiModel(AsyncModel):
    """
        Base class for API-based language models.

        This class serves as a foundation for implementing models that interact with
        external APIs. It handles the common functionality for managing model IDs,
        custom role mappings, and API client connections.

        Parameters:
            model_id (`str`):
                The identifier for the model to be used with the API.
            custom_role_conversions (`dict[str, str`], **optional**):
                Mapping to convert  between internal role names and API-specific role names. Defaults to None.
            client (`Any`, **optional**):
                Pre-configured API client instance. If not provided, a default client will be created. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the parent class.
        """

    def __init__(
            self, model_id: str, custom_role_conversions: dict[str, str] | None = None, client: Any | None = None,
            **kwargs
    ):
        super().__init__(model_id=model_id, **kwargs)
        self.custom_role_conversions = custom_role_conversions or {}
        self.client = client or self.create_client()

    def create_client(self):
        """Create the API client for the specific service."""
        raise NotImplementedError("Subclasses must implement this method to create a client")


class AsyncOpenAIServerModel(AsyncApiModel):
    """This model connects to an OpenAI-compatible API server.

    Parameters:
        model_id (`str`):
            The model identifier to use on the server (e.g. "gpt-3.5-turbo").
        api_base (`str`, *optional*):
            The base URL of the OpenAI-compatible API server.
        api_key (`str`, *optional*):
            The API key to use for authentication.
        organization (`str`, *optional*):
            The organization to use for the API request.
        project (`str`, *optional*):
            The project to use for the API request.
        client_kwargs (`dict[str, Any]`, *optional*):
            Additional keyword arguments to pass to the OpenAI client (like organization, project, max_retries etc.).
        custom_role_conversions (`dict[str, str]`, *optional*):
            Custom role conversion mapping to convert message roles in others.
            Useful for specific models that do not support specific message roles like "system".
        flatten_messages_as_text (`bool`, default `False`):
            Whether to flatten messages as text.
        **kwargs:
            Additional keyword arguments to pass to the OpenAI API.
    """

    def __init__(
            self,
            model_id: str,
            api_base: str | None = None,
            api_key: str | None = None,
            organization: str | None = None,
            project: str | None = None,
            client_kwargs: dict[str, Any] | None = None,
            custom_role_conversions: dict[str, str] | None = None,
            flatten_messages_as_text: bool = False,
            **kwargs,
    ):
        self.client_kwargs = {
            **(client_kwargs or {}),
            "api_key": api_key,
            "base_url": api_base,
            "organization": organization,
            "project": project,
        }
        super().__init__(
            model_id=model_id,
            custom_role_conversions=custom_role_conversions,
            flatten_messages_as_text=flatten_messages_as_text,
            **kwargs,
        )

    def create_client(self):
        try:
            import openai
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Please install 'openai' extra to use OpenAIServerModel: `pip install 'smolagents[openai]'`"
            ) from e

        return openai.AsyncOpenAI(**self.client_kwargs)

    async def generate_stream(
            self,
            messages: list[ChatMessage],
            stop_sequences: list[str] | None = None,
            response_format: dict[str, str] | None = None,
            tools_to_call_from: list[Tool] | None = None,
            **kwargs,
    ) -> AsyncGenerator[ChatMessageStreamDelta]:
        if tools_to_call_from:
            raise NotImplementedError("Streaming is not yet supported for tool calling")

        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            response_format=response_format,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            **kwargs,
        )

        async for event in self.client.chat.completions.create(
                **completion_kwargs, stream=True, stream_options={"include_usage": True}
        ):
            if event.choices:
                if event.choices[0].delta is None:
                    if not getattr(event.choices[0], "finish_reason", None):
                        raise ValueError(f"No content or tool calls in event: {event}")
                else:
                    yield ChatMessageStreamDelta(
                        content=event.choices[0].delta.content,
                    )
            if getattr(event, "usage", None):
                self.last_input_token_count = event.usage.prompt_tokens
                self.last_output_token_count = event.usage.completion_tokens

    async def generate(
            self,
            messages: list[dict[str, str | list[dict]]],
            stop_sequences: list[str] | None = None,
            grammar: str | None = None,
            tools_to_call_from: list[Tool] | None = None,
            **kwargs,
    ) -> ChatMessage:
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            grammar=grammar,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            **kwargs,
        )
        completion_kwargs.pop("grammar", None)
        response = await self.client.chat.completions.create(**completion_kwargs)
        self._last_input_token_count = getattr(response.usage, "prompt_tokens", 0)
        self._last_output_token_count = getattr(response.usage, "completion_tokens", 0)

        return ChatMessage.from_dict(
            response.choices[0].message.model_dump(include={"role", "content", "tool_calls"}),
            raw=response,
            token_usage=TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            ),
        )


class AsyncAzureOpenAIServerModel(AsyncOpenAIServerModel):
    """This model connects to an Azure OpenAI deployment.

    Parameters:
        model_id (`str`):
            The model deployment name to use when connecting (e.g. "gpt-4o-mini").
        azure_endpoint (`str`, *optional*):
            The Azure endpoint, including the resource, e.g. `https://example-resource.azure.openai.com/`. If not provided, it will be inferred from the `AZURE_OPENAI_ENDPOINT` environment variable.
        api_key (`str`, *optional*):
            The API key to use for authentication. If not provided, it will be inferred from the `AZURE_OPENAI_API_KEY` environment variable.
        api_version (`str`, *optional*):
            The API version to use. If not provided, it will be inferred from the `OPENAI_API_VERSION` environment variable.
        client_kwargs (`dict[str, Any]`, *optional*):
            Additional keyword arguments to pass to the AzureOpenAI client (like organization, project, max_retries etc.).
        custom_role_conversions (`dict[str, str]`, *optional*):
            Custom role conversion mapping to convert message roles in others.
            Useful for specific models that do not support specific message roles like "system".
        **kwargs:
            Additional keyword arguments to pass to the Azure OpenAI API.
    """

    def __init__(
            self,
            model_id: str,
            azure_endpoint: str | None = None,
            api_key: str | None = None,
            api_version: str | None = None,
            client_kwargs: dict[str, Any] | None = None,
            custom_role_conversions: dict[str, str] | None = None,
            **kwargs,
    ):
        client_kwargs = client_kwargs or {}
        client_kwargs.update(
            {
                "api_version": api_version,
                "azure_endpoint": azure_endpoint,
            }
        )
        super().__init__(
            model_id=model_id,
            api_key=api_key,
            client_kwargs=client_kwargs,
            custom_role_conversions=custom_role_conversions,
            **kwargs,
        )

    def create_client(self):
        try:
            import openai
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Please install 'openai' extra to use AzureOpenAIServerModel: `pip install 'smolagents[openai]'`"
            ) from e

        return openai.AsyncAzureOpenAI(**self.client_kwargs)



if __name__ == "__main__":
    import asyncio
    import os
    async def main():
        model = AsyncOpenAIServerModel(model_id="gpt-4.1", api_key=os.getenv("OPENAI_API_KEY"))
        response = await model([ChatMessage(role="user", content="Hello, how are you?")])
        print(response)

    asyncio.run(main())