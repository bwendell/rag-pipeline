"""OpenAI embedding provider stub.

This is a placeholder for future OpenAI API integration.
Raises NotImplementedError when used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag_pipeline.embedding_providers.base import AbstractEmbeddingProvider

if TYPE_CHECKING:
    from rag_pipeline.core.types import EmbeddingVector


class OpenAIEmbeddingProvider(AbstractEmbeddingProvider):
    """Stub for OpenAI embeddings API.

    This provider will integrate with OpenAI's embedding API.
    Currently not implemented.

    Attributes:
        model_name: OpenAI model name (e.g., "text-embedding-ada-002")
        api_key: OpenAI API key

    Raises:
        NotImplementedError: All methods raise this until implemented.
    """

    def __init__(
        self,
        model_name: str = "text-embedding-ada-002",
        api_key: str | None = None,
        dimension: int = 1536,
    ) -> None:
        """Initialize OpenAI embedding provider stub.

        Args:
            model_name: OpenAI model name.
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
            dimension: Embedding dimension for the model.
        """
        super().__init__(model_name=model_name, dimension=dimension)
        self.api_key = api_key

    def _get_provider_name(self) -> str:
        return "openai"

    async def embed(self, text: str) -> EmbeddingVector:
        """Not implemented - raises NotImplementedError."""
        raise NotImplementedError(
            "OpenAIEmbeddingProvider is a stub. "
            "OpenAI integration will be implemented in a future phase."
        )

    async def embed_batch(
        self,
        texts: list[str],
        *,
        batch_size: int | None = None,
        show_progress: bool = False,
    ) -> list[EmbeddingVector]:
        """Not implemented - raises NotImplementedError."""
        raise NotImplementedError(
            "OpenAIEmbeddingProvider is a stub. "
            "OpenAI integration will be implemented in a future phase."
        )
