"""Shared data types for the RAG pipeline.

This module defines the core data structures used throughout the pipeline:
- Document: Raw document before chunking
- Chunk: A piece of content with metadata and embedding
- SearchResult: Result from vector search with relevance score
- QueryContext: Context assembled for LLM generation

All types use dataclasses with comprehensive type hints for IDE support
and runtime validation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from hashlib import sha256
from typing import Any
from uuid import uuid4


class DocumentType(str, Enum):
    """Enumeration of supported document types.

    Used by chunkers to determine the appropriate chunking strategy.
    """

    CODE = "code"
    MARKDOWN = "markdown"
    TEXT = "text"
    RUNBOOK = "runbook"
    UNKNOWN = "unknown"


class SourceType(str, Enum):
    """Enumeration of document source types.

    Used to track where documents originated from.
    """

    FILESYSTEM = "filesystem"
    GIT = "git"
    CONFLUENCE = "confluence"
    S3 = "s3"
    OCI_OBJECT_STORAGE = "oci_object_storage"
    UNKNOWN = "unknown"


@dataclass
class Metadata:
    """Flexible metadata container for documents and chunks.

    Provides a structured way to store both common metadata fields
    and arbitrary custom fields.

    Attributes:
        source_path: Original file path or URL of the document.
        source_type: Type of source (filesystem, git, etc.).
        language: Programming language for code documents.
        file_extension: File extension (e.g., ".py", ".md").
        created_at: When the document was created/discovered.
        updated_at: When the document was last modified.
        custom: Dictionary for arbitrary additional metadata.

    Example:
        >>> meta = Metadata(
        ...     source_path="/path/to/file.py",
        ...     source_type=SourceType.FILESYSTEM,
        ...     language="python",
        ...     file_extension=".py"
        ... )
        >>> meta.custom["author"] = "John Doe"
    """

    source_path: str = ""
    source_type: SourceType = SourceType.UNKNOWN
    language: str | None = None
    file_extension: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    custom: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to a flat dictionary for storage.

        Returns:
            Dictionary with all metadata fields, suitable for
            vector store metadata filtering.
        """
        result = {
            "source_path": self.source_path,
            "source_type": self.source_type.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
        if self.language:
            result["language"] = self.language
        if self.file_extension:
            result["file_extension"] = self.file_extension
        result.update(self.custom)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Metadata":
        """Create Metadata from a dictionary.

        Args:
            data: Dictionary containing metadata fields.

        Returns:
            New Metadata instance.
        """
        known_keys = {
            "source_path",
            "source_type",
            "language",
            "file_extension",
            "created_at",
            "updated_at",
        }
        custom = {k: v for k, v in data.items() if k not in known_keys}

        return cls(
            source_path=data.get("source_path", ""),
            source_type=SourceType(data.get("source_type", "unknown")),
            language=data.get("language"),
            file_extension=data.get("file_extension"),
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if isinstance(data.get("created_at"), str)
                else data.get("created_at", datetime.now())
            ),
            updated_at=(
                datetime.fromisoformat(data["updated_at"])
                if isinstance(data.get("updated_at"), str)
                else data.get("updated_at", datetime.now())
            ),
            custom=custom,
        )


@dataclass
class Document:
    """A raw document before chunking.

    Represents a complete document loaded from a source, before it
    has been split into chunks for embedding.

    Attributes:
        id: Unique identifier for the document.
        content: The full text content of the document.
        doc_type: Type of document (code, markdown, etc.).
        metadata: Associated metadata.

    Example:
        >>> doc = Document(
        ...     content="def hello(): print('world')",
        ...     doc_type=DocumentType.CODE,
        ...     metadata=Metadata(source_path="hello.py", language="python")
        ... )
    """

    content: str
    doc_type: DocumentType = DocumentType.UNKNOWN
    metadata: Metadata = field(default_factory=Metadata)
    id: str = field(default_factory=lambda: str(uuid4()))

    def __post_init__(self) -> None:
        """Validate document after initialization."""
        if not self.content:
            raise ValueError("Document content cannot be empty")

    @property
    def content_hash(self) -> str:
        """Get a hash of the document content for change detection."""
        return sha256(self.content.encode("utf-8")).hexdigest()[:16]


@dataclass
class Chunk:
    """A piece of content with metadata, ready for embedding.

    Represents a segment of a document that has been chunked using
    an appropriate strategy. Chunks are the unit of storage in the
    vector database.

    Attributes:
        id: Unique identifier for the chunk.
        content: The text content of the chunk.
        document_id: ID of the parent document.
        metadata: Associated metadata (inherited and chunk-specific).
        embedding: Vector embedding (populated after embedding).
        start_index: Character offset in the original document.
        end_index: End character offset in the original document.
        chunk_index: Position of this chunk within the document.

    Example:
        >>> chunk = Chunk(
        ...     content="def hello(): print('world')",
        ...     document_id="doc-123",
        ...     metadata=Metadata(source_path="hello.py"),
        ...     start_index=0,
        ...     end_index=27
        ... )
    """

    content: str
    document_id: str
    metadata: Metadata = field(default_factory=Metadata)
    id: str = field(default_factory=lambda: str(uuid4()))
    embedding: list[float] | None = None
    start_index: int = 0
    end_index: int = 0
    chunk_index: int = 0

    def __post_init__(self) -> None:
        """Validate chunk after initialization."""
        if not self.content:
            raise ValueError("Chunk content cannot be empty")

    @property
    def has_embedding(self) -> bool:
        """Check if the chunk has an embedding vector."""
        return self.embedding is not None and len(self.embedding) > 0


@dataclass
class SearchResult:
    """Result from a vector similarity search.

    Contains a chunk along with its similarity score and any
    additional search metadata.

    Attributes:
        chunk: The matched chunk.
        score: Similarity score (higher is more similar, typically 0-1).
        rank: Position in the result list (1-indexed).

    Example:
        >>> result = SearchResult(
        ...     chunk=chunk,
        ...     score=0.95,
        ...     rank=1
        ... )
    """

    chunk: Chunk
    score: float
    rank: int = 1

    def __post_init__(self) -> None:
        """Validate search result after initialization."""
        if self.score < 0 or self.score > 1:
            # Some vector stores return scores > 1, so just warn conceptually
            pass  # Allow any score, different stores have different scales


@dataclass
class QueryContext:
    """Context assembled for LLM generation.

    Contains the user's query along with retrieved context chunks
    and any additional information needed for generation.

    Attributes:
        query: The original user query.
        chunks: Retrieved context chunks, ordered by relevance.
        max_tokens: Maximum tokens for context (for truncation).

    Example:
        >>> context = QueryContext(
        ...     query="How does authentication work?",
        ...     chunks=[result1.chunk, result2.chunk],
        ... )
    """

    query: str
    chunks: list[Chunk] = field(default_factory=list)
    max_tokens: int = 4000

    def format_context(self) -> str:
        """Format chunks into a context string for the LLM.

        Returns:
            Formatted string with all chunks and their sources.
        """
        if not self.chunks:
            return "No relevant context found."

        parts = []
        for i, chunk in enumerate(self.chunks, 1):
            source = chunk.metadata.source_path or "Unknown source"
            parts.append(f"[{i}] Source: {source}\n{chunk.content}")

        return "\n\n---\n\n".join(parts)

    @property
    def total_chunks(self) -> int:
        """Get the number of context chunks."""
        return len(self.chunks)


# Type aliases for common patterns
EmbeddingVector = list[float]
"""Type alias for embedding vectors (list of floats)."""

ChunkID = str
"""Type alias for chunk identifiers."""

DocumentID = str
"""Type alias for document identifiers."""
