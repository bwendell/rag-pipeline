"""Stub vector stores for future integrations.

These are placeholders that raise NotImplementedError when used.
They serve as examples of the VectorStore interface and can be
extended for production use in future phases.

Available stubs:
- OCIVectorStore: Oracle Database 23ai Vector Search
- QdrantStore: Qdrant vector database
"""

from rag_pipeline.vector_stores.stubs.oci_store import OCIVectorStore
from rag_pipeline.vector_stores.stubs.qdrant_store import QdrantStore

__all__ = ["OCIVectorStore", "QdrantStore"]
