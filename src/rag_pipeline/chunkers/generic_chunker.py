"""Generic chunker implementation using recursive text splitting.

This module provides the GenericChunker class that implements a recursive
text splitting strategy for chunking generic text documents. It uses a
hierarchical separator list to intelligently split text at natural boundaries,
then merges small chunks and adds overlap for context continuity.

Example usage:
    >>> from rag_pipeline.chunkers.generic_chunker import GenericChunker
    >>> from rag_pipeline.core.base_chunker import ChunkingConfig
    >>> from rag_pipeline.core.types import Document, DocumentType
    >>>
    >>> config = ChunkingConfig(chunk_size=500, chunk_overlap=50)
    >>> chunker = GenericChunker(config)
    >>> doc = Document(
    ...     content="This is a long text...",
    ...     doc_type=DocumentType.TEXT
    ... )
    >>> chunks = chunker.chunk(doc)
    >>> print(f"Created {len(chunks)} chunks")
"""

from __future__ import annotations

from typing import ClassVar

from rag_pipeline.chunkers.base import AbstractChunker
from rag_pipeline.core.base_chunker import ChunkingConfig  # noqa: TC001
from rag_pipeline.core.types import Chunk, Document  # noqa: TC001


class GenericChunker(AbstractChunker):
    """Generic text chunker using recursive text splitting.

        This chunker implements a hierarchical splitting strategy that attempts
    to split text at natural boundaries first (paragraphs), then progressively
    at smaller boundaries (sentences, phrases, words). It then merges small
    chunks and adds overlap between chunks to preserve context.

        The chunking algorithm:
        1. Split text recursively using separators in order of preference
        2. Merge chunks smaller than min_chunk_size with adjacent chunks
        3. Add overlap between consecutive chunks
        4. Create Chunk objects with proper metadata

        Attributes:
            config: The chunking configuration controlling behavior.
            SEPARATORS: Ordered list of separators for recursive splitting.

        Example:
            >>> config = ChunkingConfig(chunk_size=1000, chunk_overlap=200)
            >>> chunker = GenericChunker(config)
            >>> chunks = chunker.chunk(document)
    """

    # Default separators in order of preference (largest to smallest)
    SEPARATORS: ClassVar[list[str]] = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        """Initialize the generic chunker with configuration.

        Args:
            config: Chunking configuration. If None, uses default config.

        Raises:
            ChunkingError: If configuration validation fails.
        """
        super().__init__(config)

    def get_supported_types(self) -> list[str]:
        """Get the document types this chunker supports.

        Returns:
            List of DocumentType values this chunker can handle.
        """
        return ["text", "unknown"]

    def chunk(self, document: Document) -> list[Chunk]:
        """Split a document into chunks using recursive text splitting.

        The chunking process:
        1. Split text recursively using hierarchical separators
        2. Merge chunks smaller than min_chunk_size
        3. Add overlap between consecutive chunks
        4. Create Chunk objects with inherited metadata

        Args:
            document: The document to chunk. Must have non-empty content.

        Returns:
            List of Chunk objects representing the split document.

        Raises:
            ChunkingError: If chunking fails.
        """
        if not document.content:
            return []

        # Step 1: Split recursively using separators
        splits = self._recursive_split(document.content, self.SEPARATORS)

        # Step 2: Merge small chunks
        merged = self._merge_small_chunks(splits)

        # Step 3: Create Chunk objects with overlap
        # We track positions using merged chunks (without overlap)
        # but add overlap to the content for context continuity
        chunks: list[Chunk] = []
        current_index = 0

        for chunk_index, content_without_overlap in enumerate(merged):
            # Find the position of this chunk in the original document
            start_index = document.content.find(content_without_overlap, current_index)
            if start_index == -1:
                # Fallback: approximate position
                start_index = current_index
            end_index = start_index + len(content_without_overlap)

            # Add overlap from previous chunk if not the first chunk
            if chunk_index > 0 and self.config.chunk_overlap > 0:
                prev_content = merged[chunk_index - 1]
                overlap_text = prev_content[-self.config.chunk_overlap :]
                content_with_overlap = overlap_text + content_without_overlap
            else:
                content_with_overlap = content_without_overlap

            chunk = self._create_chunk(
                content=content_with_overlap,
                document=document,
                chunk_index=chunk_index,
                start_index=start_index,
                end_index=end_index,
                extra_metadata={"chunking_strategy": "generic"},
            )

            if chunk is not None:
                chunks.append(chunk)
                current_index = end_index

        return chunks

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        """Split text recursively using hierarchical separators.

        This method attempts to split text at natural boundaries first,
        using larger separators (paragraphs), then progressively smaller
        separators (sentences, phrases) until the target chunk size is reached.

        Args:
            text: The text to split.
            separators: Ordered list of separators to use for splitting.

        Returns:
            List of text pieces after recursive splitting.
        """
        # Base case: no separators left or text is small enough
        if not separators or len(text) <= self.config.chunk_size:
            return [text] if text else []

        separator = separators[0]
        remaining_separators = separators[1:]

        # Split by current separator
        if separator == " ":
            # Handle space separator specially to preserve it
            parts = text.split(separator)
            # Add space back to all but the last part
            splits = [part + separator for part in parts[:-1] if part]
            if parts[-1]:
                splits.append(parts[-1])
        else:
            # For other separators, split and keep the separator with the preceding part
            parts = text.split(separator)
            splits = []
            for i, part in enumerate(parts):
                if part:
                    if i < len(parts) - 1:
                        # Add separator back (except for the last part)
                        splits.append(part + separator)
                    else:
                        splits.append(part)

        # Recursively split pieces that are still too large
        result: list[str] = []
        for piece in splits:
            if len(piece) > self.config.chunk_size and remaining_separators:
                # Recursively split this piece with remaining separators
                sub_splits = self._recursive_split(piece, remaining_separators)
                result.extend(sub_splits)
            else:
                result.append(piece)

        return result

    def _merge_small_chunks(self, chunks: list[str]) -> list[str]:
        """Merge chunks that are smaller than the minimum chunk size.

                Small chunks are merged with adjacent chunks to avoid creating
        too many tiny chunks that would be inefficient for embedding and retrieval.

                Args:
                    chunks: List of text chunks to potentially merge.

                Returns:
                    List of merged chunks where all chunks meet the minimum size.
        """
        if not chunks:
            return []

        merged: list[str] = []
        current_merge = chunks[0]

        for i in range(1, len(chunks)):
            next_chunk = chunks[i]

            if len(current_merge) < self.config.min_chunk_size or (
                len(next_chunk) < self.config.min_chunk_size
                and len(current_merge) + len(next_chunk) <= self.config.max_chunk_size
            ):
                current_merge += next_chunk
            else:
                # Current merge is good, start a new one
                merged.append(current_merge)
                current_merge = next_chunk

        # Don't forget the last chunk
        merged.append(current_merge)

        return merged

    def _add_overlap(self, chunks: list[str]) -> list[str]:
        """Add overlap between consecutive chunks.

        Overlap helps preserve context across chunk boundaries by including
        the end of the previous chunk at the start of the next chunk.

        Note: This method is provided for API compliance. The main chunk()
        method handles overlap inline to correctly track document positions.

        Args:
            chunks: List of text chunks to add overlap to.

        Returns:
            List of chunks with overlap applied.
        """
        if not chunks or self.config.chunk_overlap <= 0:
            return chunks

        result: list[str] = [chunks[0]]

        for i in range(1, len(chunks)):
            current_chunk = chunks[i]
            previous_chunk = chunks[i - 1]

            # Extract overlap from the end of the previous chunk
            overlap_text = previous_chunk[-self.config.chunk_overlap :]

            # Add overlap to the start of current chunk
            result.append(overlap_text + current_chunk)

        return result
