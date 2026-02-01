"""Base chunker implementation with common functionality.

This module provides the AbstractChunker base class that implements
shared functionality for all chunker implementations, including:
- Configuration validation
- Deterministic chunk ID generation
- Chunk creation with metadata inheritance
- Helper functions for text position calculations

Example usage:
    >>> from rag_pipeline.chunkers.base import AbstractChunker
    >>>
    >>> class MyCustomChunker(AbstractChunker):
    ...     def chunk(self, document: Document) -> list[Chunk]:
    ...         # Implementation
    ...         pass
    ...
    ...     def get_supported_types(self) -> list[str]:
    ...         return ["text"]
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Any

from rag_pipeline.core.base_chunker import ChunkingConfig
from rag_pipeline.core.exceptions import ChunkingError
from rag_pipeline.core.types import Chunk, Document, Metadata


def byte_offset_to_char_offset(content: str, byte_offset: int) -> int:
    """Convert tree-sitter byte offset to character offset.

    Tree-sitter uses byte offsets into the UTF-8 encoded content, but Python
    strings are Unicode codepoints. This function converts byte offsets to
    character offsets for correct position mapping.

    Args:
        content: The original document content.
        byte_offset: The byte offset from tree-sitter (position in UTF-8 bytes).

    Returns:
        The equivalent character offset in the Unicode string.

    Raises:
        ValueError: If byte_offset is negative or exceeds content length.

    Example:
        >>> content = "Hello 世界"  # '世' is 3 bytes in UTF-8
        >>> byte_offset_to_char_offset(content, 6)
        4  # Points to '世' which is at character index 4
    """
    if byte_offset < 0:
        raise ValueError(f"byte_offset must be non-negative, got {byte_offset}")

    content_bytes = content.encode("utf-8")

    if byte_offset > len(content_bytes):
        raise ValueError(f"byte_offset {byte_offset} exceeds content length {len(content_bytes)}")

    byte_prefix = content_bytes[:byte_offset]
    return len(byte_prefix.decode("utf-8"))


def char_offset_to_byte_offset(content: str, char_offset: int) -> int:
    """Convert character offset to tree-sitter byte offset.

    The inverse of byte_offset_to_char_offset. Converts a character position
    in a Unicode string to the corresponding byte position in UTF-8 encoding.

    Args:
        content: The original document content.
        char_offset: The character offset in the Unicode string.

    Returns:
        The equivalent byte offset in the UTF-8 encoded content.

    Raises:
        ValueError: If char_offset is negative or exceeds content length.

    Example:
        >>> content = "Hello 世界"
        >>> char_offset_to_byte_offset(content, 4)
        6  # '世' starts at byte offset 6
    """
    if char_offset < 0:
        raise ValueError(f"char_offset must be non-negative, got {char_offset}")

    if char_offset > len(content):
        raise ValueError(f"char_offset {char_offset} exceeds content length {len(content)}")

    char_prefix = content[:char_offset]
    return len(char_prefix.encode("utf-8"))


def calculate_line_numbers(content: str, start_index: int, end_index: int) -> tuple[int, int]:
    """Calculate 1-indexed line numbers for a text range.

    Determines the starting and ending line numbers for a substring within
    the content, using 1-indexed line numbers as is standard for editors
    and error reporting.

    Args:
        content: The full document content.
        start_index: The starting character offset (0-indexed).
        end_index: The ending character offset (0-indexed, exclusive).

    Returns:
        A tuple of (start_line, end_line) where both are 1-indexed.

    Raises:
        ValueError: If indices are out of bounds or start_index > end_index.

    Example:
        >>> content = "Line 1\\nLine 2\\nLine 3"
        >>> calculate_line_numbers(content, 7, 13)
        (2, 2)  # "Line 2" spans lines 2 to 2
    """
    if start_index < 0 or end_index < 0:
        raise ValueError("Indices must be non-negative")

    if start_index > end_index:
        raise ValueError(f"start_index {start_index} must be <= end_index {end_index}")

    if start_index > len(content) or end_index > len(content):
        raise ValueError("Indices exceed content length")

    # Count newlines before start_index to determine start_line
    start_line = content.count("\n", 0, start_index) + 1

    # Count newlines before end_index to determine end_line
    end_line = content.count("\n", 0, end_index) + 1

    return (start_line, end_line)


class AbstractChunker(ABC):
    """Abstract base class for chunker implementations.

    Provides common functionality shared across all chunkers, including
    configuration validation, chunk ID generation, and chunk creation with
    proper metadata inheritance. Concrete implementations must override
    the abstract methods `chunk()` and `get_supported_types()`.

    Attributes:
        config: The chunking configuration controlling behavior.

    Example:
        >>> config = ChunkingConfig(chunk_size=1000, chunk_overlap=200)
        >>> chunker = MyConcreteChunker(config)
        >>> chunks = chunker.chunk(document)
    """

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        """Initialize the chunker with configuration.

        Args:
            config: Chunking configuration. If None, uses default config.

        Raises:
            ChunkingError: If configuration validation fails.
        """
        self.config = config or ChunkingConfig()
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate chunking configuration parameters.

        Ensures that:
        - chunk_size is positive
        - chunk_overlap is non-negative
        - chunk_overlap is less than chunk_size
        - min_chunk_size is positive
        - max_chunk_size is positive and >= min_chunk_size

        Raises:
            ChunkingError: If any validation fails.
        """
        if self.config.chunk_size <= 0:
            raise ChunkingError(
                "chunk_size must be positive", details={"chunk_size": self.config.chunk_size}
            )

        if self.config.chunk_overlap < 0:
            raise ChunkingError(
                "chunk_overlap must be non-negative",
                details={"chunk_overlap": self.config.chunk_overlap},
            )

        if self.config.chunk_overlap >= self.config.chunk_size:
            raise ChunkingError(
                "chunk_overlap must be less than chunk_size",
                details={
                    "chunk_overlap": self.config.chunk_overlap,
                    "chunk_size": self.config.chunk_size,
                },
            )

        if self.config.min_chunk_size <= 0:
            raise ChunkingError(
                "min_chunk_size must be positive",
                details={"min_chunk_size": self.config.min_chunk_size},
            )

        if self.config.max_chunk_size <= 0:
            raise ChunkingError(
                "max_chunk_size must be positive",
                details={"max_chunk_size": self.config.max_chunk_size},
            )

        if self.config.max_chunk_size < self.config.min_chunk_size:
            raise ChunkingError(
                "max_chunk_size must be >= min_chunk_size",
                details={
                    "max_chunk_size": self.config.max_chunk_size,
                    "min_chunk_size": self.config.min_chunk_size,
                },
            )

    def _generate_chunk_id(self, document_id: str, chunk_index: int, content: str) -> str:
        """Generate a deterministic chunk ID.

        Creates a unique identifier for a chunk based on its document ID,
        index within the document, and content (first 100 characters).
        This ensures consistent IDs across re-runs while maintaining
        uniqueness.

        Args:
            document_id: The parent document's unique identifier.
            chunk_index: The position of this chunk within the document.
            content: The chunk content (used for uniqueness).

        Returns:
            A 16-character hexadecimal string representing the chunk ID.

        Example:
            >>> chunk_id = chunker._generate_chunk_id("doc-123", 0, "def hello():")
            >>> print(chunk_id)  # e.g., "a1b2c3d4e5f6789a"
        """
        hash_input = f"{document_id}:{chunk_index}:{content[:100]}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    def _create_chunk(
        self,
        content: str,
        document: Document,
        chunk_index: int,
        start_index: int,
        end_index: int,
        extra_metadata: dict[str, Any] | None = None,
    ) -> Chunk | None:
        """Create a Chunk with inherited metadata from the parent document.

        Creates a properly initialized Chunk instance with:
        - Deterministic ID generation
        - Metadata inheritance from parent document
        - Optional additional metadata
        - Proper positioning information

        Args:
            content: The chunk's text content.
            document: The parent document this chunk belongs to.
            chunk_index: The position of this chunk within the document.
            start_index: The starting character offset in the original document.
            end_index: The ending character offset in the original document.
            extra_metadata: Additional metadata to include (optional).

        Returns:
            A new Chunk instance, or None if content is empty or whitespace-only.

        Example:
            >>> chunk = chunker._create_chunk(
            ...     content="def hello():",
            ...     document=doc,
            ...     chunk_index=0,
            ...     start_index=0,
            ...     end_index=11,
            ...     extra_metadata={"function_name": "hello"}
            ... )
        """
        # Return None for empty or whitespace-only content
        if not content or not content.strip():
            return None

        # Generate deterministic chunk ID
        chunk_id = self._generate_chunk_id(document.id, chunk_index, content)

        # Create chunk metadata by inheriting from document
        chunk_metadata = Metadata(
            source_path=document.metadata.source_path,
            source_type=document.metadata.source_type,
            language=document.metadata.language,
            file_extension=document.metadata.file_extension,
            created_at=document.metadata.created_at,
            updated_at=document.metadata.updated_at,
        )

        # Copy custom metadata from document
        chunk_metadata.custom.update(document.metadata.custom)

        # Add chunk-specific metadata
        chunk_metadata.custom["chunk_index"] = chunk_index
        chunk_metadata.custom["start_index"] = start_index
        chunk_metadata.custom["end_index"] = end_index

        # Calculate line numbers if content is available
        if document.content:
            start_line, end_line = calculate_line_numbers(document.content, start_index, end_index)
            chunk_metadata.custom["start_line"] = start_line
            chunk_metadata.custom["end_line"] = end_line

        # Merge any extra metadata
        if extra_metadata:
            chunk_metadata.custom.update(extra_metadata)

        return Chunk(
            id=chunk_id,
            content=content,
            document_id=document.id,
            metadata=chunk_metadata,
            start_index=start_index,
            end_index=end_index,
            chunk_index=chunk_index,
        )

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Split a document into chunks.

        Concrete implementations must override this method to provide
        document-specific chunking logic.

        Args:
            document: The document to chunk. Must have non-empty content.

        Returns:
            List of Chunk objects representing the split document.

        Raises:
            ChunkingError: If chunking fails (e.g., parse error).
        """
        ...

    @abstractmethod
    def get_supported_types(self) -> list[str]:
        """Get the document types this chunker supports.

        Returns:
            List of DocumentType values this chunker can handle.
        """
        ...

    def get_config(self) -> dict[str, Any]:
        """Get the chunker's configuration as a dictionary.

        Returns:
            Dictionary containing configuration parameters.
        """
        config_dict: dict[str, Any] = self.config.to_dict()
        return config_dict

    def set_chunk_size(self, size: int) -> None:
        """Set the target chunk size.

        Args:
            size: Target size in characters.

        Raises:
            ChunkingError: If the new size is invalid.
        """
        self.config.chunk_size = size
        self._validate_config()

    def set_chunk_overlap(self, overlap: int) -> None:
        """Set the chunk overlap.

        Args:
            overlap: Overlap size in characters.

        Raises:
            ChunkingError: If the new overlap is invalid.
        """
        self.config.chunk_overlap = overlap
        self._validate_config()
