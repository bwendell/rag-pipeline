"""Anthropic LLM provider stub.

This is a placeholder for future Anthropic API integration.
Raises NotImplementedError when used except for specific methods.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rag_pipeline.core.base_llm_provider import ChatMessage


@dataclass
class AnthropicConfig:
    """Configuration for Anthropic provider.

    Attributes:
        api_key: Anthropic API key.
        model: Model identifier (e.g., "claude-3-opus", "claude-3-sonnet").
        max_tokens: Maximum tokens to generate in responses.
        timeout: Request timeout in seconds.
    """

    api_key: str
    model: str
    max_tokens: int = 4096
    timeout: float = 30.0


class AnthropicProvider:
    """Stub for Anthropic LLM provider.

    This provider will integrate with Anthropic's API for chat and
    text generation using Claude models. Currently not implemented.

    Raises:
        NotImplementedError: Most methods raise this until implemented.
    """

    def __init__(self, config: AnthropicConfig | None = None) -> None:
        """Initialize Anthropic provider stub.

        Args:
            config: Configuration for Anthropic provider.
        """
        self.config = config

    async def __aenter__(self) -> AnthropicProvider:
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
            "AnthropicProvider is a stub. "
            "Anthropic integration will be implemented in a future phase."
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
            "AnthropicProvider is a stub. "
            "Anthropic integration will be implemented in a future phase."
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
            "AnthropicProvider is a stub. "
            "Anthropic integration will be implemented in a future phase."
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
            "AnthropicProvider is a stub. "
            "Anthropic integration will be implemented in a future phase."
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
            "AnthropicProvider is a stub. "
            "Anthropic integration will be implemented in a future phase."
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
            "AnthropicProvider is a stub. "
            "Anthropic integration will be implemented in a future phase."
        )
        yield ""  # pragma: no cover

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the current model.

        Returns:
            Dictionary containing model information.
        """
        return {
            "name": self.config.model if self.config else "unknown",
            "provider": "anthropic",
            "status": "stub",
        }

    async def health_check(self) -> bool:
        """Check if the LLM provider is healthy and accessible.

        Returns:
            False as this is a stub implementation.
        """
        return False
