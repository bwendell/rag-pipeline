"""OCI Generative AI embedding provider stub.

This is a placeholder for future OCI GenAI Service integration.
Raises NotImplementedError when used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag_pipeline.embedding_providers.base import AbstractEmbeddingProvider

if TYPE_CHECKING:
    from rag_pipeline.core.types import EmbeddingVector


class OCIEmbeddingProvider(AbstractEmbeddingProvider):
    """Stub for OCI Generative AI embeddings.

    This provider will integrate with Oracle Cloud Infrastructure's
    Generative AI service for embeddings. Currently not implemented.

    Attributes:
        model_name: OCI model identifier (e.g., "cohere.embed-english-v3.0")
        compartment_id: OCI compartment OCID

    Raises:
        NotImplementedError: All methods raise this until implemented.
    """

    def __init__(
        self,
        model_name: str = "cohere.embed-english-v3.0",
        compartment_id: str | None = None,
        dimension: int = 1024,
    ) -> None:
        """Initialize OCI embedding provider stub.

        Args:
            model_name: OCI GenAI model name.
            compartment_id: OCI compartment OCID.
            dimension: Embedding dimension for the model.
        """
        super().__init__(model_name=model_name, dimension=dimension)
        self.compartment_id = compartment_id

    def _get_provider_name(self) -> str:
        return "oci-genai"

    async def embed(self, text: str) -> EmbeddingVector:
        """Not implemented - raises NotImplementedError."""
        raise NotImplementedError(
            "OCIEmbeddingProvider is a stub. "
            "OCI GenAI integration will be implemented in a future phase."
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
            "OCIEmbeddingProvider is a stub. "
            "OCI GenAI integration will be implemented in a future phase."
        )
