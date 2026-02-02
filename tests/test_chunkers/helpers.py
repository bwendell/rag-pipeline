"""Test helper functions for chunker tests.

This module provides utility functions to simplify chunker testing,
including validation helpers and factory functions.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_pipeline.core.types import Chunk, Document, DocumentType


def assert_chunk_valid(chunk: Chunk) -> None:
    """Validate that a chunk has a valid structure.

        Performs comprehensive validation of chunk fields to ensure
    database storage requirements are met.

        Args:
            chunk: The chunk to validate.

        Raises:
            AssertionError: If any validation check fails.

        Example:
            >>> chunks = chunker.chunk(document)
            >>> for chunk in chunks:
            ...     assert_chunk_valid(chunk)
    """
    # ID validation
    assert chunk.id, "Chunk ID cannot be empty"
    assert isinstance(chunk.id, str), f"Chunk ID must be string, got {type(chunk.id)}"

    # Content validation
    assert chunk.content, "Chunk content cannot be empty"
    assert isinstance(chunk.content, str), (
        f"Chunk content must be string, got {type(chunk.content)}"
    )
    assert chunk.content.strip(), "Chunk content cannot be whitespace-only"

    # Document ID validation
    assert chunk.document_id, "Chunk document_id cannot be empty"
    assert isinstance(chunk.document_id, str), (
        f"Chunk document_id must be string, got {type(chunk.document_id)}"
    )

    # Index validation
    assert isinstance(chunk.start_index, int), (
        f"Chunk start_index must be int, got {type(chunk.start_index)}"
    )
    assert isinstance(chunk.end_index, int), (
        f"Chunk end_index must be int, got {type(chunk.end_index)}"
    )
    assert chunk.start_index >= 0, f"Chunk start_index must be >= 0, got {chunk.start_index}"
    assert chunk.end_index >= chunk.start_index, (
        f"Chunk end_index ({chunk.end_index}) must be >= start_index ({chunk.start_index})"
    )

    # Chunk index validation
    assert isinstance(chunk.chunk_index, int), (
        f"Chunk chunk_index must be int, got {type(chunk.chunk_index)}"
    )
    assert chunk.chunk_index >= 0, f"Chunk chunk_index must be >= 0, got {chunk.chunk_index}"

    # Metadata validation
    assert chunk.metadata is not None, "Chunk metadata cannot be None"
    assert hasattr(chunk.metadata, "source_path"), "Chunk metadata must have source_path attribute"

    # Embedding validation (if present)
    if chunk.embedding is not None:
        assert isinstance(chunk.embedding, list), (
            f"Chunk embedding must be list, got {type(chunk.embedding)}"
        )
        assert len(chunk.embedding) > 0, "Chunk embedding cannot be empty list"
        assert all(isinstance(x, (int, float)) for x in chunk.embedding), (
            "Chunk embedding must contain only numbers"
        )


def assert_chunks_cover_document(chunks: list[Chunk], document: Document) -> None:
    """Verify that chunks provide complete coverage of the document.

    Checks that:
    1. All chunks reference the correct document
    2. Chunks are ordered by position
    3. No gaps exist between chunks (accounting for overlap)
    4. The entire document is covered

    Args:
        chunks: List of chunks to validate.
        document: The original document.

    Raises:
        AssertionError: If coverage validation fails.

    Example:
        >>> chunks = chunker.chunk(document)
        >>> assert_chunks_cover_document(chunks, document)
    """
    if not chunks:
        raise AssertionError("No chunks provided for coverage validation")

    # All chunks must reference the same document
    for chunk in chunks:
        assert chunk.document_id == document.id, (
            f"Chunk document_id ({chunk.document_id}) doesn't match document id ({document.id})"
        )

    # Chunks must be ordered by start_index
    for i in range(len(chunks) - 1):
        assert chunks[i].start_index <= chunks[i + 1].start_index, (
            f"Chunks are not ordered: chunk {i} starts at {chunks[i].start_index}, "
            f"chunk {i + 1} starts at {chunks[i + 1].start_index}"
        )

    # First chunk should start at or near the beginning
    assert chunks[0].start_index <= 10, (
        f"First chunk starts too far into document: {chunks[0].start_index}"
    )

    # Last chunk should end at or near the document end
    doc_length = len(document.content)
    last_chunk = chunks[-1]
    assert last_chunk.end_index >= doc_length - 10, (
        f"Last chunk ({last_chunk.end_index}) doesn't reach document end ({doc_length})"
    )

    # Verify chunk indices are sequential
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i, (
            f"Chunk index mismatch at position {i}: expected {i}, got {chunk.chunk_index}"
        )


def assert_chunk_content_matches_indices(chunk: Chunk, document: Document) -> None:
    """Verify that chunk content matches the expected slice of document content.

    Args:
        chunk: The chunk to validate.
        document: The original document.

    Raises:
        AssertionError: If content doesn't match the expected slice.
    """
    expected_content = document.content[chunk.start_index : chunk.end_index]
    assert chunk.content == expected_content, (
        f"Chunk content doesn't match expected slice from document.\n"
        f"Expected ({len(expected_content)} chars): {expected_content[:100]!r}...\n"
        f"Got ({len(chunk.content)} chars): {chunk.content[:100]!r}..."
    )


def assert_no_overlapping_chunks(chunks: list[Chunk], allow_overlap: bool = True) -> None:
    """Verify that chunks don't improperly overlap.

    With allow_overlap=True, allows expected overlap at boundaries.
    With allow_overlap=False, requires strictly non-overlapping chunks.

    Args:
        chunks: List of chunks to validate.
        allow_overlap: Whether to allow expected overlap (default: True).

    Raises:
        AssertionError: If overlap validation fails.
    """
    if len(chunks) < 2:
        return

    for i in range(len(chunks) - 1):
        current_end = chunks[i].end_index
        next_start = chunks[i + 1].start_index

        if allow_overlap:
            # Next chunk can start before or at current chunk end (overlap allowed)
            assert next_start <= current_end, (
                f"Gap between chunks {i} and {i + 1}: "
                f"chunk {i} ends at {current_end}, chunk {i + 1} starts at {next_start}"
            )
        else:
            # Next chunk must start at or after current chunk end (no overlap)
            assert next_start >= current_end, (
                f"Unexpected overlap between chunks {i} and {i + 1}: "
                f"chunk {i} ends at {current_end}, chunk {i + 1} starts at {next_start}"
            )


def create_test_document(
    content: str,
    doc_type: DocumentType,
    source_path: str = "/test/document.txt",
    language: str | None = None,
    file_extension: str | None = None,
) -> Document:
    """Create a test document with the specified parameters.

    Factory function for creating documents in tests without importing
    all the dependencies directly.

    Args:
        content: The document content.
        doc_type: The document type (CODE, MARKDOWN, TEXT, etc.).
        source_path: Path for the document source (default: /test/document.txt).
        language: Programming language for code documents.
        file_extension: File extension override (default: inferred from source_path).

    Returns:
        A new Document instance.

    Example:
        >>> doc = create_test_document(
        ...     content="def foo(): pass",
        ...     doc_type=DocumentType.CODE,
        ...     language="python"
        ... )
    """
    from rag_pipeline.core.types import Document, Metadata, SourceType

    if file_extension is None:
        file_extension = Path(source_path).suffix

    metadata = Metadata(
        source_path=source_path,
        source_type=SourceType.FILESYSTEM,
        language=language,
        file_extension=file_extension,
    )

    return Document(
        content=content,
        doc_type=doc_type,
        metadata=metadata,
    )


def load_fixture(filename: str, fixtures_dir: Path | None = None) -> str:
    """Load test fixture file content.

    Looks for fixture files in the tests/fixtures directory.

    Args:
        filename: Name of the fixture file to load.
        fixtures_dir: Optional path to fixtures directory. If not provided,
                     uses tests/fixtures relative to project root.

    Returns:
        Content of the fixture file as a string.

    Raises:
        FileNotFoundError: If the fixture file doesn't exist.

    Example:
        >>> content = load_fixture("sample_code.py")
        >>> doc = create_test_document(content, DocumentType.CODE)
    """
    if fixtures_dir is None:
        # Calculate project root from this file's location
        project_root = Path(__file__).parent.parent.parent
        fixtures_dir = project_root / "tests" / "fixtures"

    fixture_path = fixtures_dir / filename

    if not fixture_path.exists():
        raise FileNotFoundError(
            f"Fixture file not found: {fixture_path}\nCreate it at: {fixture_path.absolute()}"
        )

    return fixture_path.read_text(encoding="utf-8")


def count_newlines(content: str) -> dict[str, int]:
    """Count different types of newlines in content.

    Useful for testing line ending handling in chunkers.

    Args:
        content: The text content to analyze.

    Returns:
        Dictionary with counts of LF, CRLF, and CR line endings.

    Example:
        >>> counts = count_newlines("Line1\nLine2\r\nLine3")
        >>> print(counts)
        {'lf': 1, 'crlf': 1, 'cr': 0}
    """
    return {
        "lf": content.count("\n"),
        "crlf": content.count("\r\n"),
        "cr": content.count("\r") - content.count("\r\n"),
    }


def calculate_expected_chunks(document_length: int, chunk_size: int, chunk_overlap: int) -> int:
    """Calculate the expected number of chunks for a document.

    Simple estimation that assumes even chunking without boundary preservation.
    Actual chunk count may vary based on content and chunker implementation.

    Args:
        document_length: Length of the document in characters.
        chunk_size: Target chunk size.
        chunk_overlap: Overlap between chunks.

    Returns:
        Expected number of chunks (minimum 1).

    Example:
        >>> expected = calculate_expected_chunks(1000, 300, 50)
        >>> print(f"Expect approximately {expected} chunks")
    """
    if document_length <= chunk_size:
        return 1

    effective_chunk_size = chunk_size - chunk_overlap
    remaining = document_length - chunk_size
    additional_chunks = (remaining + effective_chunk_size - 1) // effective_chunk_size

    return 1 + additional_chunks


def get_chunk_sizes(chunks: list[Chunk]) -> list[int]:
    """Get the sizes of all chunks in characters.

    Args:
        chunks: List of chunks.

    Returns:
        List of chunk content lengths.

    Example:
        >>> sizes = get_chunk_sizes(chunks)
        >>> print(f"Average chunk size: {sum(sizes) / len(sizes)}")
    """
    return [len(chunk.content) for chunk in chunks]
