"""Pytest fixtures for chunker tests.

This module provides chunker-specific fixtures that are automatically
loaded when running tests in the test_chunkers directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from rag_pipeline.core.base_chunker import ChunkingConfig
from rag_pipeline.core.types import Document, DocumentType, Metadata, SourceType

if TYPE_CHECKING:
    from collections.abc import Callable


# =============================================================================
# Chunking Configuration Fixtures
# =============================================================================


@pytest.fixture
def chunking_config() -> ChunkingConfig:
    """Return standard chunking configuration for tests.

    Default configuration suitable for most chunking tests.
    """
    return ChunkingConfig(
        chunk_size=500,
        chunk_overlap=50,
        min_chunk_size=50,
        max_chunk_size=1000,
        preserve_sentences=True,
        preserve_code_blocks=True,
    )


@pytest.fixture
def small_chunking_config() -> ChunkingConfig:
    """Return small chunking configuration for testing splits.

    Small chunk sizes force documents to be split into multiple chunks,
    useful for testing overlap and boundary handling.
    """
    return ChunkingConfig(
        chunk_size=100,
        chunk_overlap=20,
        min_chunk_size=20,
        max_chunk_size=200,
        preserve_sentences=True,
        preserve_code_blocks=True,
    )


@pytest.fixture
def large_chunking_config() -> ChunkingConfig:
    """Return large chunking configuration for testing single-chunk documents.

    Large chunk sizes ensure most documents fit in a single chunk,
    useful for testing basic chunking behavior.
    """
    return ChunkingConfig(
        chunk_size=5000,
        chunk_overlap=200,
        min_chunk_size=100,
        max_chunk_size=10000,
        preserve_sentences=True,
        preserve_code_blocks=True,
    )


@pytest.fixture
def overlap_disabled_config() -> ChunkingConfig:
    """Return chunking configuration with overlap disabled.

    Useful for testing chunk boundary calculations without overlap complexity.
    """
    return ChunkingConfig(
        chunk_size=500,
        chunk_overlap=0,
        min_chunk_size=50,
        max_chunk_size=1000,
        preserve_sentences=True,
        preserve_code_blocks=True,
    )


# =============================================================================
# Document Fixtures
# =============================================================================


@pytest.fixture
def code_document(sample_python_code: str) -> Document:
    """Return a Python code document for testing.

    Uses the sample_python_code fixture from parent conftest.py.
    """
    return Document(
        content=sample_python_code,
        doc_type=DocumentType.CODE,
        metadata=Metadata(
            source_path="/test/calculator.py",
            source_type=SourceType.FILESYSTEM,
            language="python",
            file_extension=".py",
        ),
    )


@pytest.fixture
def markdown_document(sample_markdown: str) -> Document:
    """Return a Markdown document for testing.

    Uses the sample_markdown fixture from parent conftest.py.
    """
    return Document(
        content=sample_markdown,
        doc_type=DocumentType.MARKDOWN,
        metadata=Metadata(
            source_path="/test/documentation.md",
            source_type=SourceType.FILESYSTEM,
            file_extension=".md",
        ),
    )


@pytest.fixture
def runbook_document(sample_runbook: str) -> Document:
    """Return a runbook document for testing.

    Uses the sample_runbook fixture from parent conftest.py.
    """
    return Document(
        content=sample_runbook,
        doc_type=DocumentType.RUNBOOK,
        metadata=Metadata(
            source_path="/test/incident_response.md",
            source_type=SourceType.FILESYSTEM,
            file_extension=".md",
        ),
    )


@pytest.fixture
def text_document() -> Document:
    """Return a plain text document for testing.

    Simple text content with multiple paragraphs.
    """
    content = """This is a simple text document.

It contains multiple paragraphs of plain text without any special formatting.
The purpose is to test basic text chunking functionality.

Another paragraph here with more content to ensure proper chunking behavior.
This should be split into appropriate chunks based on size constraints."""

    return Document(
        content=content,
        doc_type=DocumentType.TEXT,
        metadata=Metadata(
            source_path="/test/sample.txt",
            source_type=SourceType.FILESYSTEM,
            file_extension=".txt",
        ),
    )


@pytest.fixture
def empty_document() -> Document:
    """Return an empty document for edge case testing.

    Should raise ValueError during initialization.
    """
    return Document(
        content="",
        doc_type=DocumentType.TEXT,
        metadata=Metadata(
            source_path="/test/empty.txt",
            source_type=SourceType.FILESYSTEM,
            file_extension=".txt",
        ),
    )


@pytest.fixture
def unicode_document() -> Document:
    """Return a document with Unicode content for testing encoding.

    Contains various Unicode characters including multi-byte UTF-8 sequences.
    """
    content = """Unicode test document.

Here are some multi-byte characters:
- Emoji: 🚀 🎉 💻 🐍
- Chinese: 你好世界 测试文档
- Japanese: こんにちは テスト
- Russian: Привет мир тестирование
- Math: ∑ ∏ ∫ √ ∞ ≠
- Arrows: ← ↑ → ↓ ↔ ↕
- Special: € £ ¥ © ® ™

End of Unicode content."""

    return Document(
        content=content,
        doc_type=DocumentType.TEXT,
        metadata=Metadata(
            source_path="/test/unicode.txt",
            source_type=SourceType.FILESYSTEM,
            file_extension=".txt",
        ),
    )


@pytest.fixture
def mixed_line_endings_document() -> Document:
    """Return a document with mixed line endings for testing.

    Contains both LF (\n) and CRLF (\r\n) line endings.
    """
    content = "Line 1 with LF\nLine 2 with LF\r\nLine 3 with CRLF\r\nLine 4 with LF\n"

    return Document(
        content=content,
        doc_type=DocumentType.TEXT,
        metadata=Metadata(
            source_path="/test/mixed_endings.txt",
            source_type=SourceType.FILESYSTEM,
            file_extension=".txt",
        ),
    )


# =============================================================================
# Factory Fixtures
# =============================================================================


@pytest.fixture
def mock_document_factory() -> Callable[..., Document]:
    """Return a factory function for creating test documents.

    The factory allows customization of content, type, and metadata.

    Example:
        >>> doc = mock_document_factory(
        ...     content="custom content",
        ...     doc_type=DocumentType.CODE,
        ...     language="python"
        ... )
    """

    def factory(
        content: str = "Default test content.",
        doc_type: DocumentType = DocumentType.TEXT,
        source_path: str = "/test/default.txt",
        language: str | None = None,
        file_extension: str | None = None,
        custom_metadata: dict[str, Any] | None = None,
    ) -> Document:
        metadata = Metadata(
            source_path=source_path,
            source_type=SourceType.FILESYSTEM,
            language=language,
            file_extension=file_extension or Path(source_path).suffix,
        )
        if custom_metadata:
            metadata.custom.update(custom_metadata)

        return Document(
            content=content,
            doc_type=doc_type,
            metadata=metadata,
        )

    return factory


@pytest.fixture
def document_with_custom_metadata(mock_document_factory: Callable[..., Document]) -> Document:
    """Return a document with custom metadata fields."""
    return mock_document_factory(
        content="Document with custom metadata.",
        doc_type=DocumentType.MARKDOWN,
        source_path="/test/custom.md",
        custom_metadata={
            "author": "Test Author",
            "version": "1.0.0",
            "tags": ["test", "chunking"],
        },
    )


# =============================================================================
# Content Fixtures
# =============================================================================


@pytest.fixture
def long_text_content() -> str:
    """Return long text content that will definitely be split into multiple chunks.

    Approximately 2000 characters to ensure multiple chunks with standard config.
    """
    paragraphs = []
    for i in range(20):
        paragraphs.append(
            f"This is paragraph {i + 1} of a long document. "
            f"It contains enough text to ensure that when chunked with standard "
            f"settings, it will be split into multiple chunks. "
            f"Each paragraph adds more content to make the document longer."
        )
    return "\n\n".join(paragraphs)


@pytest.fixture
def single_sentence_document() -> Document:
    """Return a document with only one sentence.

    Useful for testing minimum chunk size handling.
    """
    return Document(
        content="This is a single sentence.",
        doc_type=DocumentType.TEXT,
        metadata=Metadata(
            source_path="/test/short.txt",
            source_type=SourceType.FILESYSTEM,
            file_extension=".txt",
        ),
    )


@pytest.fixture
def whitespace_only_document() -> Document:
    """Return a document with only whitespace for edge case testing.

    Contains spaces, tabs, and newlines but no meaningful content.
    """
    return Document(
        content="   \n\t  \n   ",
        doc_type=DocumentType.TEXT,
        metadata=Metadata(
            source_path="/test/whitespace.txt",
            source_type=SourceType.FILESYSTEM,
            file_extension=".txt",
        ),
    )
