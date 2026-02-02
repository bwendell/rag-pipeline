"""Tests for AbstractChunker base class.

This module tests the functionality of the AbstractChunker base class,
including configuration validation, chunk ID generation, and utility methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from rag_pipeline.chunkers.base import (
    AbstractChunker,
    byte_offset_to_char_offset,
    calculate_line_numbers,
    char_offset_to_byte_offset,
)
from rag_pipeline.core.base_chunker import ChunkingConfig
from rag_pipeline.core.exceptions import ChunkingError

if TYPE_CHECKING:
    from rag_pipeline.core.types import Document


class MockChunker(AbstractChunker):
    """Concrete implementation of AbstractChunker for testing."""

    def chunk(self, document: Document) -> list:
        """Simple chunking for testing - splits by newlines."""
        chunks = []
        lines = document.content.split("\n")
        current_index = 0
        chunk_index = 0

        for line in lines:
            if line.strip():
                chunk = self._create_chunk(
                    content=line,
                    document=document,
                    start_index=current_index,
                    end_index=current_index + len(line),
                    chunk_index=chunk_index,
                )
                if chunk:
                    chunks.append(chunk)
                    chunk_index += 1
            current_index += len(line) + 1  # +1 for newline

        return chunks

    def get_supported_types(self) -> list[str]:
        """Return supported document types."""
        return ["text", "code"]

    def get_config(self) -> dict:
        """Return configuration as dictionary."""
        return self.config.to_dict()


# =============================================================================
# Configuration Validation Tests
# =============================================================================


class TestConfigValidation:
    """Tests for configuration validation."""

    def test_valid_config(self) -> None:
        """Test that valid configuration is accepted."""
        config = ChunkingConfig(
            chunk_size=500,
            chunk_overlap=50,
            min_chunk_size=50,
            max_chunk_size=1000,
        )
        chunker = MockChunker(config)
        assert chunker.config.chunk_size == 500
        assert chunker.config.chunk_overlap == 50

    def test_default_config(self) -> None:
        """Test that default configuration is used when not specified."""
        chunker = MockChunker()
        assert chunker.config.chunk_size == 1000
        assert chunker.config.chunk_overlap == 200
        assert chunker.config.min_chunk_size == 100
        assert chunker.config.max_chunk_size == 2000

    def test_config_validation_chunk_size_positive(self) -> None:
        """Test that chunk_size must be positive."""
        with pytest.raises(ChunkingError, match="chunk_size"):
            config = ChunkingConfig(chunk_size=0)
            MockChunker(config)

    def test_config_validation_min_chunk_size_not_negative(self) -> None:
        """Test that min_chunk_size cannot be negative."""
        with pytest.raises(ChunkingError, match="min_chunk_size"):
            config = ChunkingConfig(min_chunk_size=-1)
            MockChunker(config)

    def test_config_validation_overlap_non_negative(self) -> None:
        """Test that chunk_overlap cannot be negative."""
        with pytest.raises(ChunkingError, match="chunk_overlap"):
            config = ChunkingConfig(chunk_overlap=-10)
            MockChunker(config)

    def test_config_validation_overlap_less_than_chunk_size(self) -> None:
        """Test that chunk_overlap must be less than chunk_size."""
        with pytest.raises(ChunkingError, match="overlap"):
            config = ChunkingConfig(chunk_size=100, chunk_overlap=150)
            MockChunker(config)


# =============================================================================
# Chunk ID Generation Tests
# =============================================================================


class TestChunkIdGeneration:
    """Tests for _generate_chunk_id method."""

    def test_chunk_id_is_deterministic(self) -> None:
        """Test that chunk ID generation is deterministic for same inputs."""
        chunker = MockChunker()

        id1 = chunker._generate_chunk_id("doc-123", 0, "content hash")
        id2 = chunker._generate_chunk_id("doc-123", 0, "content hash")

        assert id1 == id2

    def test_chunk_id_differs_with_different_content(self) -> None:
        """Test that chunk IDs differ for different content."""
        chunker = MockChunker()

        id1 = chunker._generate_chunk_id("doc-123", 0, "content hash 1")
        id2 = chunker._generate_chunk_id("doc-123", 0, "content hash 2")

        assert id1 != id2

    def test_chunk_id_differs_with_different_index(self) -> None:
        """Test that chunk IDs differ for different chunk indices."""
        chunker = MockChunker()

        id1 = chunker._generate_chunk_id("doc-123", 0, "content")
        id2 = chunker._generate_chunk_id("doc-123", 1, "content")

        assert id1 != id2

    def test_chunk_id_differs_with_different_document(self) -> None:
        """Test that chunk IDs differ for different documents."""
        chunker = MockChunker()

        id1 = chunker._generate_chunk_id("doc-123", 0, "content")
        id2 = chunker._generate_chunk_id("doc-456", 0, "content")

        assert id1 != id2

    def test_chunk_id_format(self) -> None:
        """Test that chunk ID has expected format (16-char hex string)."""
        chunker = MockChunker()

        chunk_id = chunker._generate_chunk_id("doc-123", 0, "content")

        assert len(chunk_id) == 16
        assert all(c in "0123456789abcdef" for c in chunk_id)


# =============================================================================
# Chunk Creation Tests
# =============================================================================


class TestChunkCreation:
    """Tests for _create_chunk method."""

    def test_create_chunk_with_valid_content(self, text_document: Document) -> None:
        """Test creating a chunk with valid content."""
        chunker = MockChunker()

        chunk = chunker._create_chunk(
            content="Valid content",
            document=text_document,
            start_index=0,
            end_index=13,
            chunk_index=0,
        )

        assert chunk is not None
        assert chunk.content == "Valid content"
        assert chunk.document_id == text_document.id
        assert chunk.start_index == 0
        assert chunk.end_index == 13
        assert chunk.chunk_index == 0

    def test_create_chunk_returns_none_for_empty_content(self, text_document: Document) -> None:
        """Test that _create_chunk returns None for empty content."""
        chunker = MockChunker()

        chunk = chunker._create_chunk(
            content="",
            document=text_document,
            start_index=0,
            end_index=0,
            chunk_index=0,
        )

        assert chunk is None

    def test_create_chunk_returns_none_for_whitespace_content(
        self, text_document: Document
    ) -> None:
        """Test that _create_chunk returns None for whitespace-only content."""
        chunker = MockChunker()

        chunk = chunker._create_chunk(
            content="   \n\t  ",
            document=text_document,
            start_index=0,
            end_index=7,
            chunk_index=0,
        )

        assert chunk is None

    def test_create_chunk_inherits_metadata(self, text_document: Document) -> None:
        """Test that created chunks inherit document metadata."""
        chunker = MockChunker()

        chunk = chunker._create_chunk(
            content="Test content",
            document=text_document,
            start_index=0,
            end_index=12,
            chunk_index=0,
        )

        assert chunk is not None
        assert chunk.metadata.source_path == text_document.metadata.source_path
        assert chunk.metadata.source_type == text_document.metadata.source_type

    def test_create_chunk_preserves_custom_metadata(self, text_document: Document) -> None:
        """Test that custom metadata is preserved in created chunks."""
        text_document.metadata.custom["section"] = "introduction"
        chunker = MockChunker()

        chunk = chunker._create_chunk(
            content="Test content",
            document=text_document,
            start_index=0,
            end_index=12,
            chunk_index=0,
        )

        assert chunk is not None
        assert chunk.metadata.custom.get("section") == "introduction"


# =============================================================================
# Unicode and Encoding Tests
# =============================================================================


class TestUnicodeHandling:
    """Tests for Unicode content handling."""

    def test_byte_offset_to_char_offset_ascii(self) -> None:
        """Test byte to char offset conversion with ASCII text."""
        text = "Hello, World!"

        # ASCII: 1 byte per character
        assert byte_offset_to_char_offset(text, 0) == 0
        assert byte_offset_to_char_offset(text, 5) == 5
        assert byte_offset_to_char_offset(text, 13) == 13

    def test_byte_offset_to_char_offset_unicode(self) -> None:
        """Test byte to char offset conversion with multi-byte Unicode."""
        # 🚀 is 4 bytes in UTF-8
        text = "Hello 🚀 World"

        # Before emoji: byte offset = char offset
        assert byte_offset_to_char_offset(text, 0) == 0
        assert byte_offset_to_char_offset(text, 5) == 5
        assert byte_offset_to_char_offset(text, 6) == 6  # Space after Hello

        # After emoji (4 bytes): byte offset differs from char offset
        # "Hello 🚀" = 6 chars (H,e,l,l,o,space) + 4 bytes for emoji = 10 bytes
        assert byte_offset_to_char_offset(text, 10) == 7  # After emoji
        assert byte_offset_to_char_offset(text, 11) == 8

    def test_byte_offset_to_char_offset_chinese(self) -> None:
        """Test byte to char offset conversion with Chinese characters."""
        # Chinese characters are typically 3 bytes in UTF-8
        text = "Hello你好"

        # "Hello" = 5 bytes = 5 chars
        assert byte_offset_to_char_offset(text, 5) == 5

        # "你" = 3 more bytes (total 8), "好" = 3 more bytes (total 11)
        # But they are 2 characters
        assert byte_offset_to_char_offset(text, 8) == 6  # After first Chinese char
        assert byte_offset_to_char_offset(text, 11) == 7  # After second Chinese char

    def test_byte_offset_to_char_offset_mixed_unicode(self) -> None:
        """Test byte to char offset with mixed ASCII, emoji, and multi-byte chars."""
        # Mix: ASCII (1 byte) + Emoji (4 bytes) + Chinese (3 bytes)
        text = "A🚀中"

        assert byte_offset_to_char_offset(text, 0) == 0  # A
        assert byte_offset_to_char_offset(text, 1) == 1  # After A
        assert byte_offset_to_char_offset(text, 5) == 2  # After 🚀
        assert byte_offset_to_char_offset(text, 8) == 3  # After 中

    def test_byte_offset_beyond_content(self) -> None:
        """Test that byte offset beyond content raises ValueError."""
        text = "Hello"

        # Beyond content length should raise ValueError
        with pytest.raises(ValueError, match="byte_offset"):
            byte_offset_to_char_offset(text, 100)


# =============================================================================
# Line Number Calculation Tests
# =============================================================================


class TestLineNumberCalculation:
    """Tests for calculate_line_numbers function."""

    def test_calculate_line_numbers_single_line(self) -> None:
        """Test line number calculation for single line content."""
        content = "Single line"

        start_line, end_line = calculate_line_numbers(content, 0, 11)

        assert start_line == 1
        assert end_line == 1

    def test_calculate_line_numbers_with_lf(self) -> None:
        """Test line number calculation with LF (Unix) line endings."""
        content = "Line 1\nLine 2\nLine 3"

        # First line
        start_line, end_line = calculate_line_numbers(content, 0, 6)
        assert start_line == 1
        assert end_line == 1

        # Second line
        start_line, end_line = calculate_line_numbers(content, 7, 13)
        assert start_line == 2
        assert end_line == 2

        # Third line
        start_line, end_line = calculate_line_numbers(content, 14, 20)
        assert start_line == 3
        assert end_line == 3

    def test_calculate_line_numbers_with_crlf(self) -> None:
        """Test line number calculation with CRLF (Windows) line endings."""
        content = "Line 1\r\nLine 2\r\nLine 3"

        # First line
        start_line, end_line = calculate_line_numbers(content, 0, 6)
        assert start_line == 1
        assert end_line == 1

        # Second line (accounting for \r\n = 2 chars)
        start_line, end_line = calculate_line_numbers(content, 8, 14)
        assert start_line == 2
        assert end_line == 2

        # Third line
        start_line, end_line = calculate_line_numbers(content, 16, 22)
        assert start_line == 3
        assert end_line == 3

    def test_calculate_line_numbers_multi_line_chunk(self) -> None:
        """Test line numbers for chunk spanning multiple lines."""
        content = "Line 1\nLine 2\nLine 3\nLine 4"

        # Chunk covering lines 2-3
        start_line, end_line = calculate_line_numbers(content, 7, 20)
        assert start_line == 2
        assert end_line == 3

    def test_calculate_line_numbers_empty_content(self) -> None:
        """Test line number calculation with empty content."""
        content = ""

        start_line, end_line = calculate_line_numbers(content, 0, 0)
        assert start_line == 1
        assert end_line == 1

    def test_calculate_line_numbers_trailing_newline(self) -> None:
        """Test line number calculation with trailing newline."""
        content = "Line 1\nLine 2\n"

        # Position at the trailing newline (end of content)
        # "Line 1\nLine 2\n" = indices 0-6, 7-13, total 14 chars
        start_line, end_line = calculate_line_numbers(content, 13, 14)
        # The last newline creates a third line conceptually
        assert start_line == 2
        assert end_line == 3


# =============================================================================
# Integration Tests
# =============================================================================


class TestAbstractChunkerIntegration:
    """Integration tests for AbstractChunker functionality."""

    def test_chunker_implements_protocol(self) -> None:
        """Test that MockChunker properly implements the Chunker protocol."""
        from rag_pipeline.core.base_chunker import Chunker

        chunker = MockChunker()
        assert isinstance(chunker, Chunker)

    def test_chunk_with_real_document(self, text_document: Document) -> None:
        """Test chunking a real document."""
        chunker = MockChunker()

        chunks = chunker.chunk(text_document)

        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.document_id == text_document.id
            assert chunk.content.strip()

    def test_chunk_metadata_preservation(self, text_document: Document) -> None:
        """Test that document metadata is preserved in chunks."""
        chunker = MockChunker()

        chunks = chunker.chunk(text_document)

        for chunk in chunks:
            assert chunk.metadata.source_path == text_document.metadata.source_path
            assert chunk.metadata.source_type == text_document.metadata.source_type

    def test_chunk_indexing(self, text_document: Document) -> None:
        """Test that chunks are properly indexed."""
        chunker = MockChunker()

        chunks = chunker.chunk(text_document)

        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i


# =============================================================================
# Additional Coverage Tests
# =============================================================================


class TestCharOffsetToByteOffset:
    """Tests for char_offset_to_byte_offset function."""

    def test_char_to_byte_offset_ascii(self) -> None:
        """Test char to byte offset conversion with ASCII text."""
        text = "Hello, World!"
        # ASCII: 1 byte per character
        assert char_offset_to_byte_offset(text, 0) == 0
        assert char_offset_to_byte_offset(text, 5) == 5
        assert char_offset_to_byte_offset(text, 13) == 13

    def test_char_to_byte_offset_unicode(self) -> None:
        """Test char to byte offset conversion with multi-byte Unicode."""
        # 🚀 is 4 bytes in UTF-8
        text = "Hello 🚀 World"
        assert char_offset_to_byte_offset(text, 6) == 6  # Space after Hello
        assert char_offset_to_byte_offset(text, 7) == 10  # After emoji
        assert char_offset_to_byte_offset(text, 8) == 11  # Space after emoji

    def test_char_to_byte_offset_chinese(self) -> None:
        """Test char to byte offset conversion with Chinese characters."""
        text = "Hello你好"
        assert char_offset_to_byte_offset(text, 5) == 5  # End of "Hello"
        assert char_offset_to_byte_offset(text, 6) == 8  # After 你 (3 bytes)
        assert char_offset_to_byte_offset(text, 7) == 11  # After 好 (3 more bytes)

    def test_char_to_byte_offset_negative_raises(self) -> None:
        """Test that negative char offset raises ValueError."""
        with pytest.raises(ValueError, match="char_offset must be non-negative"):
            char_offset_to_byte_offset("Hello", -1)

    def test_char_to_byte_offset_exceeds_length_raises(self) -> None:
        """Test that char offset exceeding content length raises ValueError."""
        with pytest.raises(ValueError, match=r"char_offset.*exceeds content length"):
            char_offset_to_byte_offset("Hello", 100)


class TestByteOffsetEdgeCases:
    """Additional edge case tests for byte_offset_to_char_offset."""

    def test_negative_byte_offset_raises(self) -> None:
        """Test that negative byte offset raises ValueError."""
        with pytest.raises(ValueError, match="byte_offset must be non-negative"):
            byte_offset_to_char_offset("Hello", -1)


class TestCalculateLineNumbersEdgeCases:
    """Additional edge case tests for calculate_line_numbers."""

    def test_negative_start_index_raises(self) -> None:
        """Test that negative start index raises ValueError."""
        with pytest.raises(ValueError, match="Indices must be non-negative"):
            calculate_line_numbers("Hello\nWorld", -1, 5)

    def test_negative_end_index_raises(self) -> None:
        """Test that negative end index raises ValueError."""
        with pytest.raises(ValueError, match="Indices must be non-negative"):
            calculate_line_numbers("Hello\nWorld", 0, -1)

    def test_start_greater_than_end_raises(self) -> None:
        """Test that start_index > end_index raises ValueError."""
        with pytest.raises(ValueError, match=r"start_index.*must be <= end_index"):
            calculate_line_numbers("Hello\nWorld", 10, 5)

    def test_indices_exceed_content_length_raises(self) -> None:
        """Test that indices exceeding content length raise ValueError."""
        with pytest.raises(ValueError, match="Indices exceed content length"):
            calculate_line_numbers("Hello", 0, 100)


class TestChunkerConfigMethods:
    """Tests for chunker configuration methods."""

    def test_get_config_returns_dict(self) -> None:
        """Test that get_config returns a dictionary with config values."""
        config = ChunkingConfig(chunk_size=500, chunk_overlap=100)
        chunker = MockChunker(config)

        result = chunker.get_config()

        assert isinstance(result, dict)
        assert result["chunk_size"] == 500
        assert result["chunk_overlap"] == 100

    def test_set_chunk_size_updates_config(self) -> None:
        """Test that set_chunk_size updates the chunk size."""
        chunker = MockChunker()
        original_size = chunker.config.chunk_size

        chunker.set_chunk_size(2000)

        assert chunker.config.chunk_size == 2000
        assert chunker.config.chunk_size != original_size

    def test_set_chunk_size_invalid_raises(self) -> None:
        """Test that set_chunk_size with invalid value raises ChunkingError."""
        chunker = MockChunker(ChunkingConfig(chunk_size=1000, chunk_overlap=100))

        with pytest.raises(ChunkingError, match="chunk_size must be positive"):
            chunker.set_chunk_size(0)

    def test_set_chunk_overlap_updates_config(self) -> None:
        """Test that set_chunk_overlap updates the chunk overlap."""
        chunker = MockChunker(ChunkingConfig(chunk_size=1000, chunk_overlap=100))

        chunker.set_chunk_overlap(200)

        assert chunker.config.chunk_overlap == 200

    def test_set_chunk_overlap_invalid_raises(self) -> None:
        """Test that set_chunk_overlap with invalid value raises ChunkingError."""
        chunker = MockChunker(ChunkingConfig(chunk_size=1000, chunk_overlap=100))

        with pytest.raises(ChunkingError, match="chunk_overlap must be less than chunk_size"):
            chunker.set_chunk_overlap(1000)  # Same as chunk_size, invalid


class TestMaxChunkSizeValidation:
    """Tests for max_chunk_size validation."""

    def test_max_chunk_size_zero_raises(self) -> None:
        """Test that zero max_chunk_size raises ChunkingError."""
        config = ChunkingConfig()
        config.max_chunk_size = 0
        with pytest.raises(ChunkingError, match="max_chunk_size must be positive"):
            MockChunker(config)

    def test_max_chunk_size_less_than_min_raises(self) -> None:
        """Test that max_chunk_size < min_chunk_size raises ChunkingError."""
        config = ChunkingConfig()
        config.min_chunk_size = 500
        config.max_chunk_size = 100
        with pytest.raises(ChunkingError, match="max_chunk_size must be >= min_chunk_size"):
            MockChunker(config)
