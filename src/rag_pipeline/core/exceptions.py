"""Custom exception hierarchy for the RAG pipeline.

This module defines a structured exception hierarchy that allows
callers to catch specific error types or broad categories of errors.

Exception Hierarchy:
    RAGPipelineError (base)
    ├── ConfigurationError
    ├── IngestionError
    │   └── DocumentSourceError
    │   └── ChunkingError
    ├── EmbeddingError
    ├── VectorStoreError
    ├── LLMProviderError
    └── QueryError

Example:
    >>> try:
    ...     await vector_store.add(chunks)
    ... except VectorStoreError as e:
    ...     logger.error(f"Failed to store chunks: {e}")
    ... except RAGPipelineError as e:
    ...     logger.error(f"Pipeline error: {e}")
"""

from typing import Any


class RAGPipelineError(Exception):
    """Base exception for all RAG pipeline errors.

    All custom exceptions in the pipeline inherit from this class,
    allowing callers to catch any pipeline-related error.

    Attributes:
        message: Human-readable error message.
        details: Optional dictionary with additional error context.

    Example:
        >>> raise RAGPipelineError("Something went wrong", details={"id": "123"})
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error message.
            details: Optional dictionary with additional context.
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        """Return string representation of the error."""
        if self.details:
            details_str = ", ".join(f"{k}={v!r}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message

    def __repr__(self) -> str:
        """Return detailed representation for debugging."""
        return f"{self.__class__.__name__}(message={self.message!r}, details={self.details!r})"


class ConfigurationError(RAGPipelineError):
    """Raised when there's a configuration problem.

    This includes invalid settings, missing required configuration,
    or incompatible configuration combinations.

    Example:
        >>> raise ConfigurationError(
        ...     "Invalid vector store type",
        ...     details={"provided": "invalid", "valid": ["chroma", "qdrant"]}
        ... )
    """

    pass


class IngestionError(RAGPipelineError):
    """Raised when document ingestion fails.

    This is a broad category covering failures during the ingestion
    pipeline, from loading documents to storing embeddings.

    Example:
        >>> raise IngestionError(
        ...     "Failed to ingest documents",
        ...     details={"failed_count": 5, "total": 100}
        ... )
    """

    pass


class DocumentSourceError(IngestionError):
    """Raised when loading documents from a source fails.

    This includes file not found, permission errors, network errors
    for remote sources, or malformed document content.

    Example:
        >>> raise DocumentSourceError(
        ...     "Failed to read file",
        ...     details={"path": "/path/to/file.py", "error": "Permission denied"}
        ... )
    """

    pass


class ChunkingError(IngestionError):
    """Raised when chunking a document fails.

    This includes parsing errors (e.g., invalid syntax for code),
    or issues with the chunking strategy.

    Example:
        >>> raise ChunkingError(
        ...     "Failed to parse Python file",
        ...     details={"path": "broken.py", "line": 42}
        ... )
    """

    pass


class EmbeddingError(RAGPipelineError):
    """Raised when embedding generation fails.

    This includes model loading failures, inference errors,
    or issues with the embedding service.

    Example:
        >>> raise EmbeddingError(
        ...     "Failed to generate embeddings",
        ...     details={"model": "all-MiniLM-L6-v2", "batch_size": 32}
        ... )
    """

    pass


class VectorStoreError(RAGPipelineError):
    """Raised when vector store operations fail.

    This includes connection errors, storage errors, or query failures.

    Example:
        >>> raise VectorStoreError(
        ...     "Failed to connect to vector store",
        ...     details={"store_type": "chroma", "path": "/data/vectors"}
        ... )
    """

    pass


class LLMProviderError(RAGPipelineError):
    """Raised when LLM operations fail.

    This includes connection errors to the LLM service, generation
    failures, or rate limiting issues.

    Example:
        >>> raise LLMProviderError(
        ...     "LLM request failed",
        ...     details={"provider": "ollama", "model": "mistral", "status": 503}
        ... )
    """

    pass


class QueryError(RAGPipelineError):
    """Raised when processing a query fails.

    This covers errors in the query pipeline, from embedding the query
    to generating the final response.

    Example:
        >>> raise QueryError(
        ...     "Failed to process query",
        ...     details={"query": "How does auth work?", "stage": "retrieval"}
        ... )
    """

    pass


class TriggerError(RAGPipelineError):
    """Raised when a trigger (HTTP, Slack, etc.) encounters an error.

    This includes startup failures, webhook processing errors,
    or message formatting issues.

    Example:
        >>> raise TriggerError(
        ...     "Slack webhook failed",
        ...     details={"trigger": "slack", "channel": "#general"}
        ... )
    """

    pass


class RetryableError(RAGPipelineError):
    """Raised for errors that may succeed on retry.

    This is a marker exception indicating that the operation
    could be retried with a reasonable chance of success.

    Attributes:
        retry_after: Suggested seconds to wait before retry.

    Example:
        >>> raise RetryableError(
        ...     "Service temporarily unavailable",
        ...     details={"service": "ollama"},
        ...     retry_after=5.0
        ... )
    """

    def __init__(
        self, message: str, details: dict[str, Any] | None = None, retry_after: float = 1.0
    ) -> None:
        """Initialize the retryable error.

        Args:
            message: Human-readable error message.
            details: Optional dictionary with additional context.
            retry_after: Suggested seconds to wait before retry.
        """
        super().__init__(message, details)
        self.retry_after = retry_after


class RetrievalError(RAGPipelineError):
    """Raised when retrieval operations fail.

    This includes query embedding failures, vector search errors,
    or result processing issues.

    Example:
        >>> raise RetrievalError(
        ...     "Vector search failed",
        ...     details={"query": "How does auth work?", "store": "chroma"}
        ... )
    """

    pass


class GenerationError(RAGPipelineError):
    """Raised when response generation fails.

    This includes context assembly failures, prompt construction issues,
    or LLM invocation errors during the generation phase.

    Example:
        >>> raise GenerationError(
        ...     "Failed to generate response",
        ...     details={"model": "mistral", "context_chunks": 5}
        ... )
    """

    pass


class NotImplementedSourceError(RAGPipelineError):
    """Raised when using a stub document source.

    Indicates that a document source is defined but not yet implemented.
    Provides guidance on alternatives and future implementation.

    Example:
        >>> raise NotImplementedSourceError(
        ...     "Document source not yet implemented",
        ...     details={"source_type": "confluence", "available": ["local_files", "git"]}
        ... )
    """

    pass
