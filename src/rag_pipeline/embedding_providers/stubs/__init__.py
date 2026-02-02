"""Stub embedding providers for future integrations.

These are placeholders that raise NotImplementedError.
"""

from rag_pipeline.embedding_providers.stubs.oci_embedding_provider import OCIEmbeddingProvider
from rag_pipeline.embedding_providers.stubs.openai_embedding_provider import OpenAIEmbeddingProvider

__all__ = ["OCIEmbeddingProvider", "OpenAIEmbeddingProvider"]
