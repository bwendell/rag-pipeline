"""Ollama LLM provider implementation.

Provides integration with local Ollama server for text generation.
Supports both streaming and non-streaming generation with full
ChatLLMProvider protocol compliance.

Example:
    >>> from rag_pipeline.llm_providers import OllamaProvider
    >>>
    >>> async with OllamaProvider() as llm:
    ...     response = await llm.generate("Explain RAG pipelines")
    ...     print(response)
    >>>
    >>> # Streaming
    >>> async with OllamaProvider() as llm:
    ...     async for chunk in llm.generate_stream("Write a poem"):
    ...         print(chunk, end="", flush=True)
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator  # noqa: TC003
from dataclasses import dataclass
from typing import Any

import httpx
import structlog

from rag_pipeline.core.base_llm_provider import ChatLLMProvider, LLMProvider
from rag_pipeline.core.exceptions import LLMProviderError, RetryableError

logger = structlog.get_logger(__name__)

_PROTOCOLS: tuple[type[object], ...] = (ChatLLMProvider, LLMProvider)


@dataclass
class OllamaConfig:
    """Configuration for Ollama provider.

    Attributes:
        base_url: Ollama API URL.
        model: Model name to use.
        timeout: Request timeout in seconds.
        max_retries: Maximum number of retry attempts.
        retry_delay: Initial delay between retries (exponential backoff).
    """

    base_url: str = "http://localhost:11434"
    model: str = "mistral"
    timeout: float = 120.0
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass
class ChatMessage:
    """A message in a chat conversation."""

    role: str  # "system", "user", or "assistant"
    content: str

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary for API request."""

        return {"role": self.role, "content": self.content}


class OllamaProvider:
    """Ollama LLM provider with streaming support.

    Connects to a local Ollama server and provides text generation
    capabilities. Implements both LLMProvider and ChatLLMProvider
    protocols for maximum flexibility.

    The provider uses httpx for async HTTP requests and supports:
    - Non-streaming generation (returns complete response)
    - Streaming generation (yields chunks as they arrive)
    - Multi-turn chat conversations
    - Health checking for service availability

    Example:
        >>> provider = OllamaProvider(config=OllamaConfig(model="codellama"))
        >>>
        >>> # Context manager usage (recommended)
        >>> async with provider:
        ...     response = await provider.generate("Hello!")
        >>>
        >>> # Manual lifecycle
        >>> await provider.start()
        >>> response = await provider.generate("Hello!")
        >>> await provider.stop()

    Attributes:
        config: Provider configuration.
    """

    # Default RAG system prompt
    DEFAULT_RAG_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

Instructions:
1. Answer the question using ONLY the information in the context below.
2. If the context doesn't contain the answer, say "I don't have enough information to answer that."
3. Cite the source numbers [1], [2], etc. when referencing information.
4. Be concise but thorough."""

    def __init__(self, config: OllamaConfig | None = None) -> None:
        """Initialize the Ollama provider.

        Args:
            config: Provider configuration. Uses defaults if None.
        """

        self.config = config or OllamaConfig()
        self._client: httpx.AsyncClient | None = None
        self._started = False

    async def __aenter__(self) -> OllamaProvider:
        """Async context manager entry."""

        await self.start()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""

        await self.stop()

    async def start(self) -> None:
        """Initialize the HTTP client.

        Creates a connection pool for efficient API requests.
        """

        if self._started:
            return

        self._client = httpx.AsyncClient(
            base_url=self.config.base_url,
            timeout=httpx.Timeout(self.config.timeout),
        )
        self._started = True
        logger.info(
            "ollama_provider_started",
            base_url=self.config.base_url,
            model=self.config.model,
        )

    async def stop(self) -> None:
        """Close the HTTP client.

        Releases connection pool resources.
        """

        if self._client:
            await self._client.aclose()
            self._client = None
        self._started = False
        logger.info("ollama_provider_stopped")

    def _ensure_started(self) -> httpx.AsyncClient:
        """Ensure client is initialized.

        Returns:
            The HTTP client instance.

        Raises:
            LLMProviderError: If the provider hasn't been started.
        """

        if not self._client:
            raise LLMProviderError(
                "Provider not started. Use 'async with' or call start() first.",
                details={"provider": "ollama"},
            )
        return self._client

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate a response from Ollama.

        Sends a prompt to the Ollama API and returns the complete
        response. Uses the chat endpoint for better results.

        Args:
            prompt: The user prompt to send.
            system_prompt: Optional system prompt for context.
            temperature: Sampling temperature (0.0-1.0).
            max_tokens: Maximum tokens to generate.
            stop_sequences: Sequences that stop generation.
            **kwargs: Additional Ollama-specific options.

        Returns:
            The generated text response.

        Raises:
            LLMProviderError: If generation fails.
            RetryableError: If the error is transient.
        """

        messages: list[ChatMessage] = []
        if system_prompt:
            messages.append(ChatMessage(role="system", content=system_prompt))
        messages.append(ChatMessage(role="user", content=prompt))

        return await self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
            **kwargs,
        )

    async def generate_stream(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Generate a streaming response from Ollama.

        Sends a prompt to Ollama and yields response chunks as they
        are generated. Useful for real-time display.

        Args:
            prompt: The user prompt to send.
            system_prompt: Optional system prompt for context.
            temperature: Sampling temperature (0.0-1.0).
            max_tokens: Maximum tokens to generate.
            stop_sequences: Sequences that stop generation.
            **kwargs: Additional Ollama-specific options.

        Yields:
            String chunks of the response.

        Raises:
            LLMProviderError: If generation fails.
        """

        messages: list[ChatMessage] = []
        if system_prompt:
            messages.append(ChatMessage(role="system", content=system_prompt))
        messages.append(ChatMessage(role="user", content=prompt))

        async for chunk in self.chat_stream(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
            **kwargs,
        ):
            yield chunk

    async def generate_with_context(
        self,
        query: str,
        context: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.3,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate a response using RAG context.

        Formats the prompt with retrieved context for RAG use cases.
        Uses a lower temperature by default for more focused answers.

        Args:
            query: The user's question.
            context: Retrieved context from vector search.
            system_prompt: Optional system prompt override.
            temperature: Sampling temperature (default: 0.3).
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional provider-specific parameters.

        Returns:
            The generated response grounded in context.
        """

        effective_system_prompt = system_prompt or self.DEFAULT_RAG_SYSTEM_PROMPT

        formatted_prompt = f"""Context:
{context}

Question: {query}

Answer:"""

        return await self.generate(
            prompt=formatted_prompt,
            system_prompt=effective_system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate a response from a conversation.

        Args:
            messages: List of chat messages.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            stop_sequences: Sequences that stop generation.
            **kwargs: Additional options.

        Returns:
            The assistant's response.
        """

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [message.to_dict() for message in messages],
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        if stop_sequences:
            payload["options"]["stop"] = stop_sequences

        if kwargs:
            payload["options"].update(kwargs)

        response = await self._request_with_retry(
            "POST",
            "/api/chat",
            json=payload,
        )

        return response["message"]["content"]

    async def chat_stream(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a response from a conversation.

        Args:
            messages: List of chat messages.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            stop_sequences: Sequences that stop generation.
            **kwargs: Additional options.

        Yields:
            String chunks of the response.
        """

        client = self._ensure_started()

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [message.to_dict() for message in messages],
            "stream": True,
            "options": {
                "temperature": temperature,
            },
        }

        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        if stop_sequences:
            payload["options"]["stop"] = stop_sequences

        if kwargs:
            payload["options"].update(kwargs)

        try:
            async with client.stream("POST", "/api/chat", json=payload) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning("invalid_json_in_stream", line=line)
                        continue

                    message = data.get("message")
                    if isinstance(message, dict):
                        content = message.get("content")
                        if content:
                            yield content

                    if data.get("done", False):
                        break

        except httpx.HTTPStatusError as exc:
            raise LLMProviderError(
                f"Ollama API error: {exc.response.status_code}",
                details={
                    "provider": "ollama",
                    "status_code": exc.response.status_code,
                    "model": self.config.model,
                },
            ) from exc
        except httpx.ConnectError as exc:
            raise LLMProviderError(
                "Failed to connect to Ollama server",
                details={
                    "provider": "ollama",
                    "base_url": self.config.base_url,
                },
            ) from exc

    async def _request_with_retry(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute an HTTP request with retry logic.

        Uses exponential backoff for transient failures.

        Args:
            method: HTTP method.
            url: Request URL (relative to base).
            **kwargs: Request arguments.

        Returns:
            Parsed JSON response.

        Raises:
            LLMProviderError: If all retries fail.
            RetryableError: If the error is transient.
        """

        client = self._ensure_started()
        last_error: Exception | None = None

        for attempt in range(self.config.max_retries):
            try:
                response = await client.request(method, url, **kwargs)
                response.raise_for_status()
                return response.json()
            except httpx.ConnectError as exc:
                last_error = exc
                logger.warning(
                    "ollama_connection_error",
                    attempt=attempt + 1,
                    max_retries=self.config.max_retries,
                    error=str(exc),
                )
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code in (429, 503, 504):
                    last_error = exc
                    logger.warning(
                        "ollama_retryable_error",
                        attempt=attempt + 1,
                        status_code=exc.response.status_code,
                    )
                else:
                    raise LLMProviderError(
                        f"Ollama API error: {exc.response.status_code}",
                        details={
                            "provider": "ollama",
                            "status_code": exc.response.status_code,
                            "response": exc.response.text[:500],
                        },
                    ) from exc
            except httpx.TimeoutException as exc:
                last_error = exc
                logger.warning(
                    "ollama_timeout",
                    attempt=attempt + 1,
                    timeout=self.config.timeout,
                )

            if attempt < self.config.max_retries - 1:
                delay = self.config.retry_delay * (2**attempt)
                await asyncio.sleep(delay)

        raise RetryableError(
            "All retry attempts failed",
            details={
                "provider": "ollama",
                "max_retries": self.config.max_retries,
                "last_error": str(last_error),
            },
            retry_after=self.config.retry_delay * (2**self.config.max_retries),
        )

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the current model.

        Returns:
            Dictionary with model information.
        """

        return {
            "name": self.config.model,
            "provider": "ollama",
            "base_url": self.config.base_url,
            "context_length": 4096,
            "capabilities": ["chat", "generate", "stream"],
        }

    async def health_check(self) -> bool:
        """Check if Ollama is healthy and accessible.

        Returns:
            True if Ollama is running and the model is available.
        """

        try:
            client = self._ensure_started()
            response = await client.get("/api/tags")
            response.raise_for_status()

            data = response.json()
            models = [model.get("name", "").split(":")[0] for model in data.get("models", [])]

            if self.config.model not in models:
                base_model = self.config.model.split(":")[0]
                if base_model not in models:
                    logger.warning(
                        "model_not_found",
                        model=self.config.model,
                        available=models,
                    )
                    return False

            return True
        except Exception as exc:
            logger.warning(
                "health_check_failed",
                error=str(exc),
            )
            return False

    async def list_models(self) -> list[str]:
        """List available models on the Ollama server.

        Returns:
            List of model names.

        Raises:
            LLMProviderError: If the request fails.
        """

        client = self._ensure_started()

        try:
            response = await client.get("/api/tags")
            response.raise_for_status()
            data = response.json()
            return [model.get("name", "") for model in data.get("models", [])]
        except httpx.HTTPError as exc:
            raise LLMProviderError(
                "Failed to list models",
                details={"provider": "ollama", "error": str(exc)},
            ) from exc
