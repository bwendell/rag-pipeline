"""Tests for MarkdownChunker implementation.

This module provides comprehensive tests for the MarkdownChunker class,
covering header-based splitting, hierarchy preservation, code block handling,
and edge cases.
"""

from __future__ import annotations

import pytest

from rag_pipeline.chunkers import MarkdownChunker
from rag_pipeline.core.base_chunker import ChunkingConfig
from rag_pipeline.core.types import Document, DocumentType, Metadata, SourceType
from tests.test_chunkers.helpers import (
    assert_chunk_valid,
    assert_no_overlapping_chunks,
    create_test_document,
)

# =============================================================================
# Supported Types Tests
# =============================================================================


class TestMarkdownChunkerTypes:
    """Tests for get_supported_types method."""

    def test_supports_markdown(self) -> None:
        """Test that chunker supports markdown documents."""
        chunker = MarkdownChunker()
        supported = chunker.get_supported_types()

        assert "markdown" in supported

    def test_supports_runbook(self) -> None:
        """Test that chunker supports runbook documents."""
        chunker = MarkdownChunker()
        supported = chunker.get_supported_types()

        assert "runbook" in supported

    def test_returns_list(self) -> None:
        """Test that get_supported_types returns a list."""
        chunker = MarkdownChunker()
        supported = chunker.get_supported_types()

        assert isinstance(supported, list)
        assert len(supported) == 2


# =============================================================================
# Empty Document Tests
# =============================================================================


class TestEmptyDocument:
    """Tests for handling empty documents."""

    def test_whitespace_only_document_returns_empty_list(self) -> None:
        """Test that whitespace-only document returns empty chunk list."""
        doc = Document(
            content="   \n\t  \n   ",
            doc_type=DocumentType.MARKDOWN,
            metadata=Metadata(
                source_path="/test/whitespace.md",
                source_type=SourceType.FILESYSTEM,
                file_extension=".md",
            ),
        )
        chunker = MarkdownChunker()

        chunks = chunker.chunk(doc)

        assert chunks == []


# =============================================================================
# No Headers Tests
# =============================================================================


class TestNoHeaders:
    """Tests for documents without any headers."""

    def test_no_headers_returns_single_chunk(self) -> None:
        """Test that document without headers becomes single chunk."""
        content = "This is plain text content without any markdown headers."
        doc = create_test_document(content, DocumentType.MARKDOWN)
        chunker = MarkdownChunker()

        chunks = chunker.chunk(doc)

        assert len(chunks) == 1
        assert chunks[0].content == content

    def test_no_headers_chunk_has_zero_level(self) -> None:
        """Test that no-header chunk has header_level of 0."""
        content = "Plain text without headers."
        doc = create_test_document(content, DocumentType.MARKDOWN)
        chunker = MarkdownChunker()

        chunks = chunker.chunk(doc)

        assert chunks[0].metadata.custom["header_level"] == 0

    def test_no_headers_chunk_has_no_title(self) -> None:
        """Test that no-header chunk has no header_title."""
        content = "Plain text without headers."
        doc = create_test_document(content, DocumentType.MARKDOWN)
        chunker = MarkdownChunker()

        chunks = chunker.chunk(doc)

        assert "header_title" not in chunks[0].metadata.custom

    def test_no_headers_with_code_blocks(self) -> None:
        """Test plain text document with code blocks but no headers."""
        content = """Here is some code:

```python
def hello():
    print("world")
```

That was the code."""
        doc = create_test_document(content, DocumentType.MARKDOWN)
        chunker = MarkdownChunker()

        chunks = chunker.chunk(doc)

        assert len(chunks) == 1
        assert "def hello()" in chunks[0].content


# =============================================================================
# Single Header Tests
# =============================================================================


class TestSingleHeader:
    """Tests for documents with a single header."""

    def test_h1_header_creates_chunk(self) -> None:
        """Test that H1 header creates a chunk."""
        content = """# Main Title

This is the content under the main title."""
        doc = create_test_document(content, DocumentType.MARKDOWN)
        chunker = MarkdownChunker()

        chunks = chunker.chunk(doc)

        assert len(chunks) == 1
        assert "# Main Title" in chunks[0].content
        assert chunks[0].metadata.custom["header_title"] == "Main Title"
        assert chunks[0].metadata.custom["header_level"] == 1

    def test_h2_header_creates_chunk(self) -> None:
        """Test that H2 header creates a chunk."""
        content = """## Section Title

Content for this section."""
        doc = create_test_document(content, DocumentType.MARKDOWN)
        chunker = MarkdownChunker()

        chunks = chunker.chunk(doc)

        assert len(chunks) == 1
        assert chunks[0].metadata.custom["header_level"] == 2
        assert chunks[0].metadata.custom["header_title"] == "Section Title"

    @pytest.mark.parametrize(
        "level,hashes", [(1, "#"), (2, "##"), (3, "###"), (4, "####"), (5, "#####"), (6, "######")]
    )
    def test_all_header_levels(self, level: int, hashes: str) -> None:
        """Test that all header levels (H1-H6) are recognized."""
        content = f"{hashes} Title Level {level}\n\nContent here."
        doc = create_test_document(content, DocumentType.MARKDOWN)
        chunker = MarkdownChunker()

        chunks = chunker.chunk(doc)

        assert len(chunks) == 1
        assert chunks[0].metadata.custom["header_level"] == level
        assert chunks[0].metadata.custom["header_title"] == f"Title Level {level}"


# =============================================================================
# Multiple Headers Tests
# =============================================================================


class TestMultipleHeaders:
    """Tests for documents with multiple headers."""

    def test_two_h1_headers_create_two_chunks(self) -> None:
        """Test that two H1 headers create two chunks."""
        content = """# First Section

Content 1

# Second Section

Content 2"""
        doc = create_test_document(content, DocumentType.MARKDOWN)
        chunker = MarkdownChunker()

        chunks = chunker.chunk(doc)

        assert len(chunks) == 2
        assert chunks[0].metadata.custom["header_title"] == "First Section"
        assert chunks[1].metadata.custom["header_title"] == "Second Section"

    def test_same_level_headers_create_separate_chunks(self) -> None:
        """Test that headers at same level create separate chunks."""
        content = """## Part A

Content A

## Part B

Content B

## Part C

Content C"""
        doc = create_test_document(content, DocumentType.MARKDOWN)
        chunker = MarkdownChunker()

        chunks = chunker.chunk(doc)

        assert len(chunks) == 3
        assert chunks[0].metadata.custom["header_title"] == "Part A"
        assert chunks[1].metadata.custom["header_title"] == "Part B"
        assert chunks[2].metadata.custom["header_title"] == "Part C"

    def test_headers_are_ordered_by_appearance(self) -> None:
        """Test that chunks are ordered by header appearance."""
        content = """## Second

Content 2

## First

Content 1

## Third

Content 3"""
        doc = create_test_document(content, DocumentType.MARKDOWN)
        chunker = MarkdownChunker()

        chunks = chunker.chunk(doc)

        assert chunks[0].metadata.custom["header_title"] == "Second"
        assert chunks[1].metadata.custom["header_title"] == "First"
        assert chunks[2].metadata.custom["header_title"] == "Third"


# =============================================================================
# Pre-Header Content Tests
# =============================================================================


class TestPreHeaderContent:
    """Tests for content before the first header."""

    def test_content_before_first_header_becomes_chunk(self) -> None:
        """Test that content before first header becomes its own chunk."""
        content = """This is introduction text.

It goes on for a bit.

# Main Section

Content under header."""
        doc = create_test_document(content, DocumentType.MARKDOWN)
        chunker = MarkdownChunker()

        chunks = chunker.chunk(doc)

        assert len(chunks) == 2
        assert chunks[0].content.startswith("This is introduction")
        assert chunks[0].metadata.custom["header_level"] == 0
        assert "header_title" not in chunks[0].metadata.custom

    def test_pre_header_content_chunk_first(self) -> None:
        """Test that pre-header content chunk comes first."""
        content = """Preamble text.

# Main Title

Main content."""
        doc = create_test_document(content, DocumentType.MARKDOWN)
        chunker = MarkdownChunker()

        chunks = chunker.chunk(doc)

        assert chunks[0].content.startswith("Preamble")
        assert "# Main Title" in chunks[1].content

    def test_no_pre_header_content_with_immediate_header(self) -> None:
        """Test that immediate header produces no pre-header chunk."""
        content = """# Start

Content here."""
        doc = create_test_document(content, DocumentType.MARKDOWN)
        chunker = MarkdownChunker()

        chunks = chunker.chunk(doc)

        assert len(chunks) == 1
        assert "# Start" in chunks[0].content

    def test_whitespace_only_pre_header_ignored(self) -> None:
        """Test that whitespace-only pre-header content is ignored."""
        content = """

# Header

Content."""
        doc = create_test_document(content, DocumentType.MARKDOWN)
        chunker = MarkdownChunker()

        chunks = chunker.chunk(doc)

        assert len(chunks) == 1
        assert chunks[0].metadata.custom["header_title"] == "Header"


# =============================================================================
# Header Hierarchy Tests
# =============================================================================


class TestHeaderHierarchy:
    """Tests for header hierarchy preservation in metadata."""

    def test_h1_has_no_parent(self) -> None:
        """Test that H1 header has no parent."""
        content = """# Top Level

Content."""
        doc = create_test_document(content, DocumentType.MARKDOWN)
        chunker = MarkdownChunker()

        chunks = chunker.chunk(doc)

        assert "parent_headers" not in chunks[0].metadata.custom
        assert "header_path" not in chunks[0].metadata.custom

    def test_h2_under_h1_has_h1_parent(self) -> None:
        """Test that H2 under H1 records H1 as parent."""
        content = """# Chapter 1

## Section 1.1

Content."""
        doc = create_test_document(content, DocumentType.MARKDOWN)
        chunker = MarkdownChunker()

        chunks = chunker.chunk(doc)

        assert len(chunks) == 2
        assert chunks[1].metadata.custom["parent_headers"] == ["Chapter 1"]
        assert chunks[1].metadata.custom["header_path"] == "Chapter 1"

    def test_h3_inherits_full_hierarchy(self) -> None:
        """Test that H3 records hierarchy of all parents."""
        content = """# Chapter

## Section

### Subsection

Content."""
        doc = create_test_document(content, DocumentType.MARKDOWN)
        chunker = MarkdownChunker()

        chunks = chunker.chunk(doc)

        assert len(chunks) == 3
        assert chunks[2].metadata.custom["parent_headers"] == ["Chapter", "Section"]
        assert chunks[2].metadata.custom["header_path"] == "Chapter > Section"

    def test_hierarchy_resets_at_higher_level(self) -> None:
        """Test that hierarchy resets when returning to higher level header."""
        content = """# Chapter 1

## Section 1.1

Content 1.1

# Chapter 2

## Section 2.1

Content 2.1"""
        doc = create_test_document(content, DocumentType.MARKDOWN)
        chunker = MarkdownChunker()

        chunks = chunker.chunk(doc)

        assert chunks[1].metadata.custom["parent_headers"] == ["Chapter 1"]
        assert chunks[3].metadata.custom["parent_headers"] == ["Chapter 2"]

    def test_h3_under_h1_skips_h2_in_hierarchy(self) -> None:
        """Test hierarchy when H3 comes directly under H1 (skipping H2)."""
        content = """# Chapter

### Subsection

Content."""
        doc = create_test_document(content, DocumentType.MARKDOWN)
        chunker = MarkdownChunker()

        chunks = chunker.chunk(doc)

        assert len(chunks) == 2
        assert chunks[1].metadata.custom["parent_headers"] == ["Chapter"]

    def test_h4_skipping_levels(self) -> None:
        """Test hierarchy when skipping multiple levels."""
        content = """# Level 1

#### Level 4

Content."""
        doc = create_test_document(content, DocumentType.MARKDOWN)
        chunker = MarkdownChunker()

        chunks = chunker.chunk(doc)

        assert chunks[1].metadata.custom["parent_headers"] == ["Level 1"]
        assert chunks[1].metadata.custom["header_level"] == 4


# =============================================================================
# Code Block Atomicity Tests
# =============================================================================


class TestCodeBlockAtomicity:
    """Tests for code block handling - headers inside code blocks are ignored."""

    def test_header_inside_code_block_ignored(self) -> None:
        """Test that headers inside code blocks are ignored."""
        content = """# Real Header

Real content

```python
# This looks like header but is not
print("Not a header")
```

More content."""
        doc = create_test_document(content, DocumentType.MARKDOWN)
        chunker = MarkdownChunker()

        chunks = chunker.chunk(doc)

        assert len(chunks) == 1
        assert chunks[0].metadata.custom["header_title"] == "Real Header"

    def test_fenced_code_block_with_backticks(self) -> None:
        """Test code blocks fenced with backticks."""
        content = """# Section 1

```
def func():
    # This is not a header
    pass
```

Content."""
        doc = create_test_document(content, DocumentType.MARKDOWN)
        chunker = MarkdownChunker()

        chunks = chunker.chunk(doc)

        assert len(chunks) == 1

    def test_fenced_code_block_with_tildes(self) -> None:
        """Test code blocks fenced with tildes."""
        content = """# Section 1

~~~python
def func():
    # This is not a header
    pass
~~~

Content."""
        doc = create_test_document(content, DocumentType.MARKDOWN)
        chunker = MarkdownChunker()

        chunks = chunker.chunk(doc)

        assert len(chunks) == 1

    def test_multiple_code_blocks_with_headers(self) -> None:
        """Test multiple code blocks don't interfere with header detection."""
        content = """# First Section

```
# Not a header
```

## Second Section

```
## Also not a header
```

Content."""
        doc = create_test_document(content, DocumentType.MARKDOWN)
        chunker = MarkdownChunker()

        chunks = chunker.chunk(doc)

        assert len(chunks) == 2
        assert chunks[0].metadata.custom["header_title"] == "First Section"
        assert chunks[1].metadata.custom["header_title"] == "Second Section"

    def test_real_header_after_code_block(self) -> None:
        """Test that real headers after code blocks are recognized."""
        content = """# First

```
# Fake header
```

# Second

Content."""
        doc = create_test_document(content, DocumentType.MARKDOWN)
        chunker = MarkdownChunker()

        chunks = chunker.chunk(doc)

        assert len(chunks) == 2
        assert chunks[0].metadata.custom["header_title"] == "First"
        assert chunks[1].metadata.custom["header_title"] == "Second"


# =============================================================================
# Content Preservation Tests
# =============================================================================


class TestContentPreservation:
    """Tests for preservation of content formatting and completeness."""

    def test_chunk_content_includes_header(self) -> None:
        """Test that chunk content includes the header line."""
        content = """# Main Title

Content here."""
        doc = create_test_document(content, DocumentType.MARKDOWN)
        chunker = MarkdownChunker()

        chunks = chunker.chunk(doc)

        assert chunks[0].content.startswith("# Main Title")

    def test_chunk_content_includes_full_section(self) -> None:
        """Test that chunk content includes all content under header."""
        content = """# Section

Paragraph 1

Paragraph 2

Paragraph 3"""
        doc = create_test_document(content, DocumentType.MARKDOWN)
        chunker = MarkdownChunker()

        chunks = chunker.chunk(doc)

        assert "Paragraph 1" in chunks[0].content
        assert "Paragraph 2" in chunks[0].content
        assert "Paragraph 3" in chunks[0].content

    def test_tables_preserved(self) -> None:
        """Test that markdown tables are preserved intact."""
        content = """# Documentation

| Header 1 | Header 2 |
|----------|----------|
| Cell 1   | Cell 2   |
| Cell 3   | Cell 4   |"""
        doc = create_test_document(content, DocumentType.MARKDOWN)
        chunker = MarkdownChunker()

        chunks = chunker.chunk(doc)

        assert "| Header 1" in chunks[0].content
        assert "| Cell 1" in chunks[0].content

    def test_lists_preserved(self) -> None:
        """Test that markdown lists are preserved."""
        content = """# Setup

Steps:
- Item 1
- Item 2
- Item 3

Done."""
        doc = create_test_document(content, DocumentType.MARKDOWN)
        chunker = MarkdownChunker()

        chunks = chunker.chunk(doc)

        assert "- Item 1" in chunks[0].content
        assert "- Item 2" in chunks[0].content

    def test_code_blocks_preserved(self) -> None:
        """Test that code blocks are preserved intact."""
        content = """# Installation

```bash
pip install package
pip install another-package
```

Done."""
        doc = create_test_document(content, DocumentType.MARKDOWN)
        chunker = MarkdownChunker()

        chunks = chunker.chunk(doc)

        assert "pip install package" in chunks[0].content
        assert "pip install another-package" in chunks[0].content


# =============================================================================
# Metadata Inheritance Tests
# =============================================================================


class TestMetadataInheritance:
    """Tests for metadata inheritance and tracking."""

    def test_chunk_inherits_document_source_path(self, markdown_document: Document) -> None:
        """Test that chunks inherit document's source_path."""
        chunker = MarkdownChunker()
        chunks = chunker.chunk(markdown_document)

        for chunk in chunks:
            assert chunk.metadata.source_path == markdown_document.metadata.source_path

    def test_chunk_inherits_document_source_type(self, markdown_document: Document) -> None:
        """Test that chunks inherit document's source_type."""
        chunker = MarkdownChunker()
        chunks = chunker.chunk(markdown_document)

        for chunk in chunks:
            assert chunk.metadata.source_type == markdown_document.metadata.source_type

    def test_chunk_inherits_custom_metadata(self) -> None:
        """Test that chunks inherit document's custom metadata."""
        content = """# Title

Content."""
        metadata = Metadata(
            source_path="/test/doc.md",
            source_type=SourceType.FILESYSTEM,
            file_extension=".md",
        )
        metadata.custom["version"] = "1.0.0"
        metadata.custom["author"] = "Test Author"
        doc = Document(
            content=content,
            doc_type=DocumentType.MARKDOWN,
            metadata=metadata,
        )
        chunker = MarkdownChunker()

        chunks = chunker.chunk(doc)

        assert chunks[0].metadata.custom["version"] == "1.0.0"
        assert chunks[0].metadata.custom["author"] == "Test Author"

    def test_chunk_adds_chunking_metadata(self) -> None:
        """Test that chunking process adds chunking-related metadata."""
        content = """# Title

Content."""
        doc = create_test_document(content, DocumentType.MARKDOWN)
        chunker = MarkdownChunker()

        chunks = chunker.chunk(doc)

        assert chunks[0].metadata.custom["chunking_strategy"] == "markdown"
        assert "chunk_index" in chunks[0].metadata.custom
        assert "start_index" in chunks[0].metadata.custom
        assert "end_index" in chunks[0].metadata.custom
        assert "start_line" in chunks[0].metadata.custom
        assert "end_line" in chunks[0].metadata.custom

    def test_chunk_index_increments(self) -> None:
        """Test that chunk_index increments for each chunk."""
        content = """# Section 1

Content 1

# Section 2

Content 2"""
        doc = create_test_document(content, DocumentType.MARKDOWN)
        chunker = MarkdownChunker()

        chunks = chunker.chunk(doc)

        assert chunks[0].metadata.custom["chunk_index"] == 0
        assert chunks[1].metadata.custom["chunk_index"] == 1


# =============================================================================
# Index and Coverage Tests
# =============================================================================


class TestIndexAndCoverage:
    """Tests for index tracking and document coverage."""

    def test_chunk_start_and_end_indices(self) -> None:
        """Test that chunks have correct start and end indices."""
        content = """# Header 1

Content 1

# Header 2

Content 2"""
        doc = create_test_document(content, DocumentType.MARKDOWN)
        chunker = MarkdownChunker()

        chunks = chunker.chunk(doc)

        assert chunks[0].start_index == 0
        assert chunks[0].end_index > chunks[0].start_index
        assert chunks[1].start_index >= chunks[0].end_index

    def test_chunk_indices_within_document(self, markdown_document: Document) -> None:
        """Test that all chunk indices are within document bounds."""
        chunker = MarkdownChunker()
        chunks = chunker.chunk(markdown_document)

        for chunk in chunks:
            assert 0 <= chunk.start_index < len(markdown_document.content)
            assert chunk.start_index < chunk.end_index <= len(markdown_document.content)

    def test_document_fully_covered(self) -> None:
        """Test that chunks cover the entire document."""
        content = """# Section 1

Content 1

# Section 2

Content 2

# Section 3

Content 3"""
        doc = create_test_document(content, DocumentType.MARKDOWN)
        chunker = MarkdownChunker()

        chunks = chunker.chunk(doc)

        assert chunks[0].start_index == 0
        assert chunks[-1].end_index == len(content)


# =============================================================================
# Configuration Tests
# =============================================================================


class TestMarkdownChunkerConfiguration:
    """Tests for chunker configuration."""

    def test_default_config(self) -> None:
        """Test that chunker works with default config."""
        chunker = MarkdownChunker()
        assert chunker.config is not None

    def test_custom_config(self) -> None:
        """Test that chunker accepts custom configuration."""
        config = ChunkingConfig(
            chunk_size=500,
            chunk_overlap=50,
        )
        chunker = MarkdownChunker(config)
        assert chunker.config.chunk_size == 500
        assert chunker.config.chunk_overlap == 50

    def test_chunker_with_config_still_splits_on_headers(self) -> None:
        """Test that header-based splitting works regardless of config."""
        config = ChunkingConfig(
            chunk_size=100,
            chunk_overlap=10,
        )
        content = """# First

This content is longer than the chunk size configured above but should still be kept together with its header because headers determine the split points, not the chunk size configuration.

# Second

Another section here."""
        doc = create_test_document(content, DocumentType.MARKDOWN)
        chunker = MarkdownChunker(config)

        chunks = chunker.chunk(doc)

        assert len(chunks) == 2


# =============================================================================
# Runbook Document Tests
# =============================================================================


class TestRunbookSupport:
    """Tests for runbook document support."""

    def test_chunker_works_with_runbook_type(self, runbook_document: Document) -> None:
        """Test that chunker processes runbook documents."""
        chunker = MarkdownChunker()

        chunks = chunker.chunk(runbook_document)

        assert len(chunks) > 0
        for chunk in chunks:
            assert_chunk_valid(chunk)

    def test_runbook_headers_split_correctly(self, runbook_document: Document) -> None:
        """Test that runbook headers are recognized and split correctly."""
        chunker = MarkdownChunker()

        chunks = chunker.chunk(runbook_document)

        titles = [
            chunk.metadata.custom.get("header_title")
            for chunk in chunks
            if "header_title" in chunk.metadata.custom
        ]
        assert len(titles) > 0


# =============================================================================
# Edge Cases and Special Characters
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and special characters."""

    def test_header_with_special_characters(self) -> None:
        """Test headers containing special characters."""
        content = """# Title with @Special #Characters & Symbols!

Content."""
        doc = create_test_document(content, DocumentType.MARKDOWN)
        chunker = MarkdownChunker()

        chunks = chunker.chunk(doc)

        assert (
            chunks[0].metadata.custom["header_title"]
            == "Title with @Special #Characters & Symbols!"
        )

    def test_header_with_code_backticks(self) -> None:
        """Test headers containing code backticks."""
        content = """# Using `function()` in Python

Content."""
        doc = create_test_document(content, DocumentType.MARKDOWN)
        chunker = MarkdownChunker()

        chunks = chunker.chunk(doc)

        assert "`function()`" in chunks[0].metadata.custom["header_title"]

    def test_header_with_links(self) -> None:
        """Test headers containing markdown links."""
        content = """# Check out [Documentation](http://example.com)

Content."""
        doc = create_test_document(content, DocumentType.MARKDOWN)
        chunker = MarkdownChunker()

        chunks = chunker.chunk(doc)

        assert "[Documentation]" in chunks[0].metadata.custom["header_title"]

    def test_header_with_emoji(self) -> None:
        """Test headers containing emoji."""
        content = """# 🚀 Getting Started

Content."""
        doc = create_test_document(content, DocumentType.MARKDOWN)
        chunker = MarkdownChunker()

        chunks = chunker.chunk(doc)

        assert "🚀" in chunks[0].metadata.custom["header_title"]

    def test_header_with_unicode(self) -> None:
        """Test headers containing Unicode characters."""
        content = """# Über Guide: Ñoño Help 你好

Content."""
        doc = create_test_document(content, DocumentType.MARKDOWN)
        chunker = MarkdownChunker()

        chunks = chunker.chunk(doc)

        assert "Über" in chunks[0].metadata.custom["header_title"]

    def test_very_long_header(self) -> None:
        """Test header with very long text."""
        long_title = "A" * 200
        content = f"# {long_title}\n\nContent."
        doc = create_test_document(content, DocumentType.MARKDOWN)
        chunker = MarkdownChunker()

        chunks = chunker.chunk(doc)

        assert chunks[0].metadata.custom["header_title"] == long_title

    def test_header_with_leading_trailing_spaces(self) -> None:
        """Test that header title is trimmed of leading/trailing spaces."""
        content = """#   Title with spaces

Content."""
        doc = create_test_document(content, DocumentType.MARKDOWN)
        chunker = MarkdownChunker()

        chunks = chunker.chunk(doc)

        assert chunks[0].metadata.custom["header_title"] == "Title with spaces"


# =============================================================================
# Fixture-based Tests
# =============================================================================


class TestWithFixtures:
    """Tests using provided fixtures."""

    def test_markdown_document_fixture(self, markdown_document: Document) -> None:
        """Test chunking with markdown_document fixture."""
        chunker = MarkdownChunker()

        chunks = chunker.chunk(markdown_document)

        assert len(chunks) > 0
        assert all(len(chunk.content) > 0 for chunk in chunks)

    def test_runbook_document_fixture(self, runbook_document: Document) -> None:
        """Test chunking with runbook_document fixture."""
        chunker = MarkdownChunker()

        chunks = chunker.chunk(runbook_document)

        assert len(chunks) > 0
        assert all(len(chunk.content) > 0 for chunk in chunks)

    def test_all_chunks_valid(self, markdown_document: Document) -> None:
        """Test that all chunks from fixture pass validation."""
        chunker = MarkdownChunker()
        chunks = chunker.chunk(markdown_document)

        for chunk in chunks:
            assert_chunk_valid(chunk)

    def test_no_overlapping_chunks(self, markdown_document: Document) -> None:
        """Test that chunks don't improperly overlap."""
        chunker = MarkdownChunker()
        chunks = chunker.chunk(markdown_document)

        if len(chunks) > 1:
            assert_no_overlapping_chunks(chunks, allow_overlap=False)
