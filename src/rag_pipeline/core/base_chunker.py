"""Chunker Protocol definition.

This module defines the abstract interface for chunker implementations.
All chunkers (code, markdown, generic text) must conform to this protocol.

Chunking is the process of splitting documents into smaller pieces
that can be embedded and stored in the vector database. Different
content types benefit from different chunking strategies.

Example usage:
    >>> from rag_pipeline.core import Chunker
    >>>
    >>> class MyChunker:
    ...     def chunk(self, document: Document) -> list[Chunk]:
    ...         # Split document into chunks
    ...         ...
    >>>
    >>> # Type checking: isinstance works at runtime
    >>> assert isinstance(MyChunker(), Chunker)
"""

from typing import Any, Protocol, runtime_checkable

from rag_pipeline.core.types import Chunk, Document


@runtime_checkable
class Chunker(Protocol):
    """Protocol for chunker implementations.

    Chunkers are responsible for:
    1. Splitting documents into meaningful chunks
    2. Preserving semantic boundaries (functions, paragraphs, etc.)
    3. Adding chunk-specific metadata (line numbers, headers, etc.)
    4. Handling overlap between chunks for context continuity

    Chunking is synchronous because it's CPU-bound work that
    benefits from simple, sequential processing.

    Implementations:
        - CodeChunker: AST-aware code chunking with tree-sitter
        - MarkdownChunker: Header-based markdown chunking
        - GenericChunker: Recursive text splitting
    """

    def chunk(self, document: Document) -> list[Chunk]:
        """Split a document into chunks.

        Analyzes the document content and splits it into chunks
        that preserve semantic meaning as much as possible.

        Args:
            document: The document to chunk. Must have non-empty content.

        Returns:
            List of Chunk objects. Each chunk contains:
                - content: The chunk text
                - document_id: Reference to parent document
                - metadata: Inherited + chunk-specific metadata
                - start_index, end_index: Position in original
                - chunk_index: Order within document

        Raises:
            ChunkingError: If chunking fails (e.g., parse error).

        Example:
            >>> doc = Document(content="def foo(): pass", doc_type=DocumentType.CODE)
            >>> chunks = chunker.chunk(doc)
            >>> print(f"Created {len(chunks)} chunks")
        """
        ...

    def get_supported_types(self) -> list[str]:
        """Get document types this chunker supports.

        Returns:
            List of DocumentType values this chunker handles.

        Example:
            >>> types = chunker.get_supported_types()
            >>> print(types)  # ["code"]
        """
        ...

    def get_config(self) -> dict[str, Any]:
        """Get the chunker's configuration.

        Returns:
            Dictionary containing configuration:
                - chunk_size: Target chunk size in characters
                - chunk_overlap: Overlap between chunks
                - min_chunk_size: Minimum chunk size
                - strategy: Chunking strategy name

        Example:
            >>> config = chunker.get_config()
            >>> print(f"Chunk size: {config['chunk_size']}")
        """
        ...


@runtime_checkable
class ConfigurableChunker(Chunker, Protocol):
    """Extended protocol for chunkers with runtime configuration.

    Adds the ability to modify chunker behavior at runtime.
    """

    def set_chunk_size(self, size: int) -> None:
        """Set the target chunk size.

        Args:
            size: Target size in characters.
        """
        ...

    def set_chunk_overlap(self, overlap: int) -> None:
        """Set the chunk overlap.

        Args:
            overlap: Overlap size in characters.
        """
        ...


class ChunkingConfig:
    """Configuration for chunking operations.

    This class holds configuration that can be shared across
    chunker implementations.

    Attributes:
        chunk_size: Target size for chunks in characters.
        chunk_overlap: Number of characters to overlap between chunks.
        min_chunk_size: Minimum chunk size (smaller pieces are merged).
        max_chunk_size: Maximum chunk size (larger pieces are split).
        preserve_sentences: Whether to avoid splitting mid-sentence.
        preserve_code_blocks: Whether to keep code blocks intact.

    Example:
        >>> config = ChunkingConfig(chunk_size=1000, chunk_overlap=100)
        >>> chunker = GenericChunker(config)
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
        max_chunk_size: int | None = None,
        preserve_sentences: bool = True,
        preserve_code_blocks: bool = True,
    ) -> None:
        """Initialize chunking configuration.

        Args:
            chunk_size: Target chunk size in characters.
            chunk_overlap: Overlap between consecutive chunks.
            min_chunk_size: Minimum chunk size.
            max_chunk_size: Maximum chunk size (None = 2x chunk_size).
            preserve_sentences: Avoid mid-sentence splits.
            preserve_code_blocks: Keep code blocks intact.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size or (chunk_size * 2)
        self.preserve_sentences = preserve_sentences
        self.preserve_code_blocks = preserve_code_blocks

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "min_chunk_size": self.min_chunk_size,
            "max_chunk_size": self.max_chunk_size,
            "preserve_sentences": self.preserve_sentences,
            "preserve_code_blocks": self.preserve_code_blocks,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChunkingConfig":
        """Create configuration from dictionary."""
        return cls(
            chunk_size=data.get("chunk_size", 1000),
            chunk_overlap=data.get("chunk_overlap", 200),
            min_chunk_size=data.get("min_chunk_size", 100),
            max_chunk_size=data.get("max_chunk_size"),
            preserve_sentences=data.get("preserve_sentences", True),
            preserve_code_blocks=data.get("preserve_code_blocks", True),
        )
