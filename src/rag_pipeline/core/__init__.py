"""Core module for RAG pipeline abstractions.

This module exports all Protocol definitions, data types, and exceptions
used throughout the RAG pipeline. Import from here for a clean public API.

Example:
    >>> from rag_pipeline.core import (
    ...     Document, Chunk, SearchResult,
    ...     VectorStore, LLMProvider, EmbeddingProvider,
    ...     RAGPipelineError, ConfigurationError,
    ... )
"""

from rag_pipeline.core.base_chunker import (
    Chunker,
    ChunkingConfig,
    ConfigurableChunker,
)
from rag_pipeline.core.base_document_source import (
    DocumentChangeCallback,
    DocumentChangeEvent,
    DocumentSource,
    WatchableDocumentSource,
    WatchHandle,
)
from rag_pipeline.core.base_embedding_provider import (
    CachedEmbeddingProvider,
    EmbeddingProvider,
)
from rag_pipeline.core.base_llm_provider import (
    ChatLLMProvider,
    ChatMessage,
    LLMProvider,
    Message,
    MessageRole,
)
from rag_pipeline.core.base_trigger import (
    QueryCallback,
    QueryRequest,
    QueryResponse,
    StreamingCallback,
    StreamingTrigger,
    TokenHandler,
    Trigger,
)
from rag_pipeline.core.base_vector_store import VectorStore
from rag_pipeline.core.exceptions import (
    ChunkingError,
    ConfigurationError,
    DocumentSourceError,
    EmbeddingError,
    GenerationError,
    IngestionError,
    LLMProviderError,
    NotImplementedSourceError,
    QueryError,
    RAGPipelineError,
    RetrievalError,
    RetryableError,
    TriggerError,
    VectorStoreError,
)
from rag_pipeline.core.types import (
    Chunk,
    ChunkID,
    Document,
    DocumentID,
    DocumentType,
    EmbeddingVector,
    Metadata,
    QueryContext,
    SearchResult,
    SourceType,
)

__all__ = [  # noqa: RUF022 - Grouped by category for maintainability
    # Types
    "Chunk",
    "ChunkID",
    "Document",
    "DocumentID",
    "DocumentType",
    "EmbeddingVector",
    "Metadata",
    "QueryContext",
    "SearchResult",
    "SourceType",
    # Protocols - Vector Store
    "VectorStore",
    # Protocols - LLM
    "ChatLLMProvider",
    "ChatMessage",
    "LLMProvider",
    "Message",
    "MessageRole",
    # Protocols - Embedding
    "CachedEmbeddingProvider",
    "EmbeddingProvider",
    # Protocols - Document Source
    "DocumentChangeCallback",
    "DocumentChangeEvent",
    "DocumentSource",
    "WatchableDocumentSource",
    "WatchHandle",
    # Protocols - Chunker
    "Chunker",
    "ChunkingConfig",
    "ConfigurableChunker",
    # Protocols - Trigger
    "QueryCallback",
    "QueryRequest",
    "QueryResponse",
    "StreamingCallback",
    "StreamingTrigger",
    "TokenHandler",
    "Trigger",
    # Exceptions
    "ChunkingError",
    "ConfigurationError",
    "DocumentSourceError",
    "EmbeddingError",
    "GenerationError",
    "IngestionError",
    "LLMProviderError",
    "NotImplementedSourceError",
    "QueryError",
    "RAGPipelineError",
    "RetrievalError",
    "RetryableError",
    "TriggerError",
    "VectorStoreError",
]
