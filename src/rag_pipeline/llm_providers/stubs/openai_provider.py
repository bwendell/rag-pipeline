"""OpenAI LLM provider stub.

This is a placeholder for future OpenAI API integration.
Raises NotImplementedError when used except for specific methods.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rag_pipeline.core.base_llm_provider import ChatMessage


@dataclass
class OpenAIConfig:
    """Configuration for OpenAI provider.

    Attributes:
        api_key: OpenAI API key.
        model: Model identifier (e.g., "gpt-4", "gpt-3.5-turbo").
        organization: Optional organization ID.
        base_url: Optional custom base URL for the API.
        timeout: Request timeout in seconds.
    """

    api_key: str
    model: str
    organization: str | None = None
    base_url: str | None = None
    timeout: float = 30.0


class OpenAIProvider:
    """Stub for OpenAI LLM provider.

    This provider will integrate with OpenAI's API for chat and
    text generation. Currently not implemented.

    Raises:
        NotImplementedError: Most methods raise this until implemented.
    """

    def __init__(self, config: OpenAIConfig | None = None) -> None:
        """Initialize OpenAI provider stub.

        Args:
            config: Configuration for OpenAI provider.
        """
        self.config = config

    async def __aenter__(self) -> OpenAIProvider:
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.stop()

    async def start(self) -> None:
        """Initialize the provider connection.

        Raises:
            NotImplementedError: This is a stub implementation.
        """
        raise NotImplementedError(
            "OpenAIProvider is a stub. OpenAI integration will be implemented in a future phase."
        )

    async def stop(self) -> None:
        """Cleanup the provider connection."""
        pass

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
        """Generate a response from the LLM.

        Raises:
            NotImplementedError: This is a stub implementation.
        """
        raise NotImplementedError(
            "OpenAIProvider is a stub. OpenAI integration will be implemented in a future phase."
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
        """Generate a streaming response from the LLM.

        Raises:
            NotImplementedError: This is a stub implementation.
        """
        raise NotImplementedError(
            "OpenAIProvider is a stub. OpenAI integration will be implemented in a future phase."
        )
        yield ""  # pragma: no cover

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

        Raises:
            NotImplementedError: This is a stub implementation.
        """
        raise NotImplementedError(
            "OpenAIProvider is a stub. OpenAI integration will be implemented in a future phase."
        )

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate a response from a conversation history.

        Raises:
            NotImplementedError: This is a stub implementation.
        """
        raise NotImplementedError(
            "OpenAIProvider is a stub. OpenAI integration will be implemented in a future phase."
        )

    async def chat_stream(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a response from a conversation history.

        Raises:
            NotImplementedError: This is a stub implementation.
        """
        raise NotImplementedError(
            "OpenAIProvider is a stub. OpenAI integration will be implemented in a future phase."
        )
        yield ""  # pragma: no cover

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the current model.

        Returns:
            Dictionary containing model information.
        """
        return {
            "name": self.config.model if self.config else "unknown",
            "provider": "openai",
            "status": "stub",
        }

    async def health_check(self) -> bool:
        """Check if the LLM provider is healthy and accessible.

        Returns:
            False as this is a stub implementation.
        """
        return False
