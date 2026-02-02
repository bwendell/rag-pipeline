"""OCI Generative AI LLM provider stub.

This is a placeholder for future OCI GenAI Service integration.
Raises NotImplementedError when used except for specific methods.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rag_pipeline.core.base_llm_provider import ChatMessage


@dataclass
class OCIGenAIConfig:
    """Configuration for OCI Generative AI provider.

    Attributes:
        compartment_id: OCI compartment OCID.
        model_id: OCI GenAI model identifier.
        region: OCI region (e.g., "us-phoenix-1").
        endpoint: Optional custom endpoint URL.
        config_file: Path to OCI config file.
        profile: OCI config profile to use.
    """

    compartment_id: str = ""
    model_id: str = "cohere.command"
    region: str = "us-ashburn-1"
    endpoint: str | None = None
    config_file: str | None = None
    profile: str = "DEFAULT"


class OCIGenAIProvider:
    """Stub for OCI Generative AI LLM provider.

    This provider will integrate with Oracle Cloud Infrastructure's
    Generative AI service. Currently not implemented.

    Raises:
        NotImplementedError: Most methods raise this until implemented.
    """

    def __init__(self, config: OCIGenAIConfig | None = None) -> None:
        """Initialize OCI GenAI provider stub.

        Args:
            config: Configuration for OCI GenAI provider.
        """
        self.config = config

    async def __aenter__(self) -> OCIGenAIProvider:
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
            "OCIGenAIProvider is a stub. "
            "OCI GenAI integration will be implemented in a future phase."
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
            "OCIGenAIProvider is a stub. "
            "OCI GenAI integration will be implemented in a future phase."
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
            "OCIGenAIProvider is a stub. "
            "OCI GenAI integration will be implemented in a future phase."
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
            "OCIGenAIProvider is a stub. "
            "OCI GenAI integration will be implemented in a future phase."
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
            "OCIGenAIProvider is a stub. "
            "OCI GenAI integration will be implemented in a future phase."
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
            "OCIGenAIProvider is a stub. "
            "OCI GenAI integration will be implemented in a future phase."
        )
        yield ""  # pragma: no cover

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the current model.

        Returns:
            Dictionary containing model information.
        """
        return {
            "name": self.config.model_id if self.config else "unknown",
            "provider": "oci-genai",
            "status": "stub",
            "region": self.config.region if self.config else "unknown",
        }

    async def health_check(self) -> bool:
        """Check if the LLM provider is healthy and accessible.

        Returns:
            False as this is a stub implementation.
        """
        return False
