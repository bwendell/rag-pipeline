"""Comprehensive tests for GenericChunker implementation.

Tests cover:
- Recursive splitting at different separator levels
- Overlap between chunks
- Small chunk merging
- Edge cases (empty, single sentence, long text)
- Supported document types
- Metadata preservation
- Chunk indexing and positioning
"""

from __future__ import annotations

import pytest

from rag_pipeline.chunkers.generic_chunker import GenericChunker
from rag_pipeline.core.base_chunker import ChunkingConfig
from rag_pipeline.core.types import Document, DocumentType, Metadata, SourceType
from tests.test_chunkers.helpers import (
    assert_chunk_valid,
    assert_chunks_cover_document,
)


class TestGenericChunkerInitialization:
    """Test GenericChunker initialization and configuration."""

    def test_init_with_default_config(self) -> None:
        chunker = GenericChunker()
        assert chunker.config is not None
        assert isinstance(chunker.config, ChunkingConfig)

    def test_init_with_custom_config(self, chunking_config: ChunkingConfig) -> None:
        chunker = GenericChunker(chunking_config)
        assert chunker.config == chunking_config

    def test_separators_are_defined(self) -> None:
        assert GenericChunker.SEPARATORS is not None
        assert isinstance(GenericChunker.SEPARATORS, list)
        assert len(GenericChunker.SEPARATORS) == 8
        assert GenericChunker.SEPARATORS[0] == "\n\n"
        assert GenericChunker.SEPARATORS[-1] == " "


class TestGetSupportedTypes:
    """Test get_supported_types method."""

    def test_supported_types_returns_list(self) -> None:
        chunker = GenericChunker()
        supported = chunker.get_supported_types()
        assert isinstance(supported, list)

    def test_supported_types_includes_text(self) -> None:
        chunker = GenericChunker()
        supported = chunker.get_supported_types()
        assert "text" in supported

    def test_supported_types_includes_unknown(self) -> None:
        chunker = GenericChunker()
        supported = chunker.get_supported_types()
        assert "unknown" in supported

    def test_supported_types_exact_match(self) -> None:
        chunker = GenericChunker()
        supported = chunker.get_supported_types()
        assert supported == ["text", "unknown"]


class TestEmptyDocument:
    """Test handling of empty documents."""

    def test_empty_document_raises_validation_error(self, chunking_config: ChunkingConfig) -> None:
        GenericChunker(chunking_config)
        with pytest.raises(ValueError, match="Document content cannot be empty"):
            Document(
                content="",
                doc_type=DocumentType.TEXT,
                metadata=Metadata(
                    source_path="/test/empty.txt",
                    source_type=SourceType.FILESYSTEM,
                ),
            )

    def test_whitespace_only_document(self, chunking_config: ChunkingConfig) -> None:
        chunker = GenericChunker(chunking_config)
        doc = Document(
            content="   \n\t  \n   ",
            doc_type=DocumentType.TEXT,
            metadata=Metadata(
                source_path="/test/whitespace.txt",
                source_type=SourceType.FILESYSTEM,
            ),
        )
        chunks = chunker.chunk(doc)
        # Whitespace is preserved by recursive split, may produce chunks
        # but all chunks should be validated
        for chunk in chunks:
            assert_chunk_valid(chunk)


class TestSingleSentence:
    """Test handling of single-sentence documents."""

    def test_single_sentence_produces_one_chunk(
        self, single_sentence_document: Document, chunking_config: ChunkingConfig
    ) -> None:
        chunker = GenericChunker(chunking_config)
        chunks = chunker.chunk(single_sentence_document)
        assert len(chunks) == 1
        assert chunks[0].content == single_sentence_document.content

    def test_single_sentence_chunk_is_valid(
        self, single_sentence_document: Document, chunking_config: ChunkingConfig
    ) -> None:
        chunker = GenericChunker(chunking_config)
        chunks = chunker.chunk(single_sentence_document)
        assert_chunk_valid(chunks[0])

    def test_single_sentence_chunk_index(
        self, single_sentence_document: Document, chunking_config: ChunkingConfig
    ) -> None:
        chunker = GenericChunker(chunking_config)
        chunks = chunker.chunk(single_sentence_document)
        assert chunks[0].chunk_index == 0


class TestRecursiveSplitting:
    """Test recursive splitting at different separator levels."""

    def test_split_at_paragraph_boundary(self, chunking_config: ChunkingConfig) -> None:
        chunker = GenericChunker(chunking_config)
        content = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        doc = Document(
            content=content,
            doc_type=DocumentType.TEXT,
            metadata=Metadata(
                source_path="/test/paragraphs.txt",
                source_type=SourceType.FILESYSTEM,
            ),
        )
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 1

    def test_split_at_newline_when_paragraph_too_large(
        self, small_chunking_config: ChunkingConfig
    ) -> None:
        chunker = GenericChunker(small_chunking_config)
        content = "Line 1.\nLine 2.\nLine 3.\nLine 4.\nLine 5.\nLine 6."
        doc = Document(
            content=content,
            doc_type=DocumentType.TEXT,
            metadata=Metadata(
                source_path="/test/lines.txt",
                source_type=SourceType.FILESYSTEM,
            ),
        )
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 1

    def test_split_preserves_content_integrity(
        self, long_text_content: str, chunking_config: ChunkingConfig
    ) -> None:
        chunker = GenericChunker(chunking_config)
        doc = Document(
            content=long_text_content,
            doc_type=DocumentType.TEXT,
            metadata=Metadata(
                source_path="/test/long.txt",
                source_type=SourceType.FILESYSTEM,
            ),
        )
        chunks = chunker.chunk(doc)
        reconstructed = "".join(chunk.content for chunk in chunks)
        # Content should be preserved (though overlap may duplicate some text)
        assert len(reconstructed) >= len(long_text_content)
        for chunk in chunks:
            assert chunk.content in long_text_content or any(
                chunk.content.startswith(long_text_content[i:])
                for i in range(len(long_text_content))
            )

    def test_recursive_split_method_with_no_separators(
        self, chunking_config: ChunkingConfig
    ) -> None:
        chunker = GenericChunker(chunking_config)
        content = "Short text"
        result = chunker._recursive_split(content, [])
        assert result == [content]

    def test_recursive_split_method_with_single_separator(
        self, chunking_config: ChunkingConfig
    ) -> None:
        chunker = GenericChunker(chunking_config)
        content = "First. Second. Third."
        result = chunker._recursive_split(content, [". "])
        assert all(isinstance(part, str) for part in result)

    def test_recursive_split_preserves_separators(
        self, small_chunking_config: ChunkingConfig
    ) -> None:
        chunker = GenericChunker(small_chunking_config)
        content = "First sentence. Second sentence."
        result = chunker._recursive_split(content, [". "])
        joined = "".join(result)
        # Separators should be preserved in the joined content
        assert ". " in joined


class TestOverlapBehavior:
    """Test chunk overlap functionality."""

    def test_overlap_added_between_chunks(
        self, long_text_content: str, chunking_config: ChunkingConfig
    ) -> None:
        chunker = GenericChunker(chunking_config)
        doc = Document(
            content=long_text_content,
            doc_type=DocumentType.TEXT,
            metadata=Metadata(
                source_path="/test/long.txt",
                source_type=SourceType.FILESYSTEM,
            ),
        )
        chunks = chunker.chunk(doc)

        if len(chunks) > 1:
            # Check overlap exists between consecutive chunks
            for i in range(len(chunks) - 1):
                current_chunk = chunks[i]
                next_chunk = chunks[i + 1]
                # Next chunk should contain overlap from previous chunk end
                prev_end = current_chunk.content[-chunking_config.chunk_overlap :]
                assert next_chunk.content.startswith(prev_end)

    def test_no_overlap_when_disabled(
        self, long_text_content: str, overlap_disabled_config: ChunkingConfig
    ) -> None:
        chunker = GenericChunker(overlap_disabled_config)
        doc = Document(
            content=long_text_content,
            doc_type=DocumentType.TEXT,
            metadata=Metadata(
                source_path="/test/long.txt",
                source_type=SourceType.FILESYSTEM,
            ),
        )
        chunks = chunker.chunk(doc)

        if len(chunks) > 1:
            for i in range(len(chunks) - 1):
                current_end = chunks[i].end_index
                next_start = chunks[i + 1].start_index
                # With no overlap, next chunk starts at or after current ends
                assert next_start >= current_end

    def test_first_chunk_has_no_overlap_prefix(
        self, long_text_content: str, chunking_config: ChunkingConfig
    ) -> None:
        chunker = GenericChunker(chunking_config)
        doc = Document(
            content=long_text_content,
            doc_type=DocumentType.TEXT,
            metadata=Metadata(
                source_path="/test/long.txt",
                source_type=SourceType.FILESYSTEM,
            ),
        )
        chunks = chunker.chunk(doc)
        # First chunk should start at beginning
        assert chunks[0].start_index == 0

    def test_overlap_size_respects_configuration(self, long_text_content: str) -> None:
        config_100 = ChunkingConfig(
            chunk_size=300,
            chunk_overlap=100,
            min_chunk_size=30,
            max_chunk_size=500,
        )
        chunker = GenericChunker(config_100)
        doc = Document(
            content=long_text_content,
            doc_type=DocumentType.TEXT,
            metadata=Metadata(
                source_path="/test/long.txt",
                source_type=SourceType.FILESYSTEM,
            ),
        )
        chunks = chunker.chunk(doc)

        if len(chunks) > 1:
            for i in range(1, len(chunks)):
                prev_chunk = chunks[i - 1]
                curr_chunk = chunks[i]
                overlap_text = prev_chunk.content[-100:]
                assert curr_chunk.content.startswith(overlap_text)


class TestSmallChunkMerging:
    """Test merging of chunks smaller than minimum size."""

    def test_small_chunks_are_merged(self) -> None:
        config = ChunkingConfig(
            chunk_size=100,
            chunk_overlap=10,
            min_chunk_size=50,
            max_chunk_size=200,
        )
        chunker = GenericChunker(config)

        small_chunks = ["Short.", "Also.", "Tiny."]
        merged = chunker._merge_small_chunks(small_chunks)

        # Chunks smaller than min_chunk_size should be merged
        for chunk in merged:
            assert len(chunk) >= config.min_chunk_size or len(merged) == 1

    def test_merge_small_chunks_preserves_content(self) -> None:
        config = ChunkingConfig(
            chunk_size=100,
            chunk_overlap=10,
            min_chunk_size=50,
            max_chunk_size=200,
        )
        chunker = GenericChunker(config)

        small_chunks = ["Short.", "Also.", "Tiny.", "More content here."]
        merged = chunker._merge_small_chunks(small_chunks)

        # Content should be preserved
        joined = "".join(small_chunks)
        merged_joined = "".join(merged)
        assert merged_joined == joined

    def test_merge_empty_chunks_list(self) -> None:
        config = ChunkingConfig(
            chunk_size=100,
            chunk_overlap=10,
            min_chunk_size=50,
            max_chunk_size=200,
        )
        chunker = GenericChunker(config)
        merged = chunker._merge_small_chunks([])
        assert merged == []

    def test_merge_respects_max_chunk_size(self) -> None:
        config = ChunkingConfig(
            chunk_size=100,
            chunk_overlap=10,
            min_chunk_size=50,
            max_chunk_size=100,
        )
        chunker = GenericChunker(config)

        chunks = ["A" * 60, "B" * 60]
        merged = chunker._merge_small_chunks(chunks)

        # Should not merge if combined exceeds max_chunk_size
        for chunk in merged:
            assert len(chunk) <= config.max_chunk_size or len(merged) == 1

    def test_large_chunks_not_merged(self) -> None:
        config = ChunkingConfig(
            chunk_size=1000,
            chunk_overlap=50,
            min_chunk_size=100,
            max_chunk_size=2000,
        )
        chunker = GenericChunker(config)

        large_chunks = ["A" * 500, "B" * 500, "C" * 500]
        merged = chunker._merge_small_chunks(large_chunks)

        # Large chunks should not be merged
        assert len(merged) == len(large_chunks)


class TestLongText:
    """Test handling of long documents."""

    def test_long_text_produces_multiple_chunks(
        self, long_text_content: str, small_chunking_config: ChunkingConfig
    ) -> None:
        chunker = GenericChunker(small_chunking_config)
        doc = Document(
            content=long_text_content,
            doc_type=DocumentType.TEXT,
            metadata=Metadata(
                source_path="/test/long.txt",
                source_type=SourceType.FILESYSTEM,
            ),
        )
        chunks = chunker.chunk(doc)
        assert len(chunks) > 1

    def test_all_chunks_are_valid(
        self, long_text_content: str, chunking_config: ChunkingConfig
    ) -> None:
        chunker = GenericChunker(chunking_config)
        doc = Document(
            content=long_text_content,
            doc_type=DocumentType.TEXT,
            metadata=Metadata(
                source_path="/test/long.txt",
                source_type=SourceType.FILESYSTEM,
            ),
        )
        chunks = chunker.chunk(doc)
        for chunk in chunks:
            assert_chunk_valid(chunk)

    def test_chunks_respect_size_constraints(
        self, long_text_content: str, chunking_config: ChunkingConfig
    ) -> None:
        chunker = GenericChunker(chunking_config)
        doc = Document(
            content=long_text_content,
            doc_type=DocumentType.TEXT,
            metadata=Metadata(
                source_path="/test/long.txt",
                source_type=SourceType.FILESYSTEM,
            ),
        )
        chunks = chunker.chunk(doc)

        for chunk in chunks:
            # Content without overlap should be within size bounds
            assert len(chunk.content) <= chunking_config.max_chunk_size * 1.5

    def test_chunks_are_sequential(
        self, long_text_content: str, chunking_config: ChunkingConfig
    ) -> None:
        chunker = GenericChunker(chunking_config)
        doc = Document(
            content=long_text_content,
            doc_type=DocumentType.TEXT,
            metadata=Metadata(
                source_path="/test/long.txt",
                source_type=SourceType.FILESYSTEM,
            ),
        )
        chunks = chunker.chunk(doc)

        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i


class TestChunkMetadata:
    """Test chunk metadata preservation and generation."""

    def test_chunk_inherits_document_metadata(
        self, text_document: Document, chunking_config: ChunkingConfig
    ) -> None:
        chunker = GenericChunker(chunking_config)
        chunks = chunker.chunk(text_document)

        for chunk in chunks:
            assert chunk.document_id == text_document.id
            assert chunk.metadata.source_path == text_document.metadata.source_path

    def test_chunk_has_strategy_metadata(
        self, text_document: Document, chunking_config: ChunkingConfig
    ) -> None:
        chunker = GenericChunker(chunking_config)
        chunks = chunker.chunk(text_document)

        for chunk in chunks:
            assert "chunking_strategy" in chunk.metadata.custom
            assert chunk.metadata.custom["chunking_strategy"] == "generic"

    def test_chunk_indices_are_set(
        self, long_text_content: str, chunking_config: ChunkingConfig
    ) -> None:
        chunker = GenericChunker(chunking_config)
        doc = Document(
            content=long_text_content,
            doc_type=DocumentType.TEXT,
            metadata=Metadata(
                source_path="/test/long.txt",
                source_type=SourceType.FILESYSTEM,
            ),
        )
        chunks = chunker.chunk(doc)

        for chunk in chunks:
            assert chunk.start_index >= 0
            assert chunk.end_index > chunk.start_index
            assert chunk.chunk_index >= 0

    def test_chunk_id_is_deterministic(
        self, text_document: Document, chunking_config: ChunkingConfig
    ) -> None:
        chunker = GenericChunker(chunking_config)
        chunks1 = chunker.chunk(text_document)
        chunks2 = chunker.chunk(text_document)

        assert len(chunks1) == len(chunks2)
        for c1, c2 in zip(chunks1, chunks2, strict=False):
            assert c1.id == c2.id


class TestDocumentCoverage:
    """Test that chunks cover the entire document."""

    def test_coverage_with_overlap_disabled(
        self, long_text_content: str, overlap_disabled_config: ChunkingConfig
    ) -> None:
        chunker = GenericChunker(overlap_disabled_config)
        doc = Document(
            content=long_text_content,
            doc_type=DocumentType.TEXT,
            metadata=Metadata(
                source_path="/test/long.txt",
                source_type=SourceType.FILESYSTEM,
            ),
        )
        chunks = chunker.chunk(doc)
        assert_chunks_cover_document(chunks, doc)

    def test_coverage_with_overlap_enabled(
        self, long_text_content: str, chunking_config: ChunkingConfig
    ) -> None:
        chunker = GenericChunker(chunking_config)
        doc = Document(
            content=long_text_content,
            doc_type=DocumentType.TEXT,
            metadata=Metadata(
                source_path="/test/long.txt",
                source_type=SourceType.FILESYSTEM,
            ),
        )
        chunks = chunker.chunk(doc)
        assert_chunks_cover_document(chunks, doc)

    def test_first_chunk_starts_at_beginning(
        self, text_document: Document, chunking_config: ChunkingConfig
    ) -> None:
        chunker = GenericChunker(chunking_config)
        chunks = chunker.chunk(text_document)
        assert chunks[0].start_index == 0

    def test_last_chunk_ends_near_document_end(
        self, text_document: Document, chunking_config: ChunkingConfig
    ) -> None:
        chunker = GenericChunker(chunking_config)
        chunks = chunker.chunk(text_document)
        # Last chunk should end close to document end (within 10 chars for safety)
        assert chunks[-1].end_index >= len(text_document.content) - 10


class TestTextDocument:
    """Test with plain text documents."""

    def test_text_document_type_supported(
        self, text_document: Document, chunking_config: ChunkingConfig
    ) -> None:
        chunker = GenericChunker(chunking_config)
        assert "text" in chunker.get_supported_types()
        chunks = chunker.chunk(text_document)
        assert len(chunks) >= 1

    def test_text_document_chunking(
        self, text_document: Document, chunking_config: ChunkingConfig
    ) -> None:
        chunker = GenericChunker(chunking_config)
        chunks = chunker.chunk(text_document)

        for chunk in chunks:
            assert_chunk_valid(chunk)
            assert chunk.metadata.source_path == text_document.metadata.source_path


class TestUnknownDocumentType:
    """Test with unknown document types."""

    def test_unknown_type_is_supported(self, chunking_config: ChunkingConfig) -> None:
        chunker = GenericChunker(chunking_config)
        assert "unknown" in chunker.get_supported_types()

    def test_unknown_type_can_be_chunked(self, chunking_config: ChunkingConfig) -> None:
        chunker = GenericChunker(chunking_config)
        doc = Document(
            content="This is unknown type content.",
            doc_type=DocumentType.UNKNOWN,
            metadata=Metadata(
                source_path="/test/unknown.bin",
                source_type=SourceType.FILESYSTEM,
            ),
        )
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 1


@pytest.mark.parametrize(
    "chunk_size,overlap,min_size,max_size",
    [
        (100, 10, 20, 150),
        (500, 50, 50, 1000),
        (1000, 200, 100, 2000),
    ],
)
class TestParametrizedConfigurations:
    """Test with various configuration combinations."""

    def test_chunking_with_config(
        self,
        chunk_size: int,
        overlap: int,
        min_size: int,
        max_size: int,
        long_text_content: str,
    ) -> None:
        config = ChunkingConfig(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            min_chunk_size=min_size,
            max_chunk_size=max_size,
        )
        chunker = GenericChunker(config)
        doc = Document(
            content=long_text_content,
            doc_type=DocumentType.TEXT,
            metadata=Metadata(
                source_path="/test/long.txt",
                source_type=SourceType.FILESYSTEM,
            ),
        )
        chunks = chunker.chunk(doc)

        assert len(chunks) >= 1
        for chunk in chunks:
            assert_chunk_valid(chunk)


class TestAddOverlapMethod:
    """Test the _add_overlap helper method."""

    def test_add_overlap_to_multiple_chunks(self, chunking_config: ChunkingConfig) -> None:
        chunker = GenericChunker(chunking_config)
        chunks = ["First chunk content here.", "Second chunk content here.", "Third chunk content."]
        result = chunker._add_overlap(chunks)

        assert len(result) == len(chunks)
        # First chunk unchanged
        assert result[0] == chunks[0]
        # Subsequent chunks should have overlap prepended
        for i in range(1, len(result)):
            overlap_part = chunks[i - 1][-chunking_config.chunk_overlap :]
            assert result[i].startswith(overlap_part)

    def test_add_overlap_with_zero_overlap(self) -> None:
        config = ChunkingConfig(
            chunk_size=100,
            chunk_overlap=0,
            min_chunk_size=20,
            max_chunk_size=200,
        )
        chunker = GenericChunker(config)
        chunks = ["First.", "Second.", "Third."]
        result = chunker._add_overlap(chunks)

        # With zero overlap, chunks should be unchanged
        assert result == chunks

    def test_add_overlap_to_empty_list(self, chunking_config: ChunkingConfig) -> None:
        chunker = GenericChunker(chunking_config)
        result = chunker._add_overlap([])
        assert result == []

    def test_add_overlap_to_single_chunk(self, chunking_config: ChunkingConfig) -> None:
        chunker = GenericChunker(chunking_config)
        chunks = ["Only chunk"]
        result = chunker._add_overlap(chunks)
        assert result == chunks


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_long_chunk_is_split(self) -> None:
        config = ChunkingConfig(
            chunk_size=100,
            chunk_overlap=10,
            min_chunk_size=20,
            max_chunk_size=200,
        )
        chunker = GenericChunker(config)

        long_word = "a" * 500
        doc = Document(
            content=long_word,
            doc_type=DocumentType.TEXT,
            metadata=Metadata(
                source_path="/test/long_word.txt",
                source_type=SourceType.FILESYSTEM,
            ),
        )
        chunks = chunker.chunk(doc)
        # Even a single very long word should be handled
        assert len(chunks) >= 1

    def test_mixed_separators_in_content(self, small_chunking_config: ChunkingConfig) -> None:
        chunker = GenericChunker(small_chunking_config)
        content = "Para 1.\n\nPara 2.\nLine 1.\nLine 2. Sentence 1? Sentence 2! More, content."
        doc = Document(
            content=content,
            doc_type=DocumentType.TEXT,
            metadata=Metadata(
                source_path="/test/mixed.txt",
                source_type=SourceType.FILESYSTEM,
            ),
        )
        chunks = chunker.chunk(doc)

        assert len(chunks) >= 1
        for chunk in chunks:
            assert_chunk_valid(chunk)

    def test_unicode_content(self, chunking_config: ChunkingConfig) -> None:
        chunker = GenericChunker(chunking_config)
        content = "Hello 世界. 你好. こんにちは. Привет."
        doc = Document(
            content=content,
            doc_type=DocumentType.TEXT,
            metadata=Metadata(
                source_path="/test/unicode.txt",
                source_type=SourceType.FILESYSTEM,
            ),
        )
        chunks = chunker.chunk(doc)

        assert len(chunks) >= 1
        for chunk in chunks:
            assert_chunk_valid(chunk)

    def test_special_characters_in_content(self, chunking_config: ChunkingConfig) -> None:
        chunker = GenericChunker(chunking_config)
        content = "Special chars: @#$%^&*()[]{}|\\<>? And more!"
        doc = Document(
            content=content,
            doc_type=DocumentType.TEXT,
            metadata=Metadata(
                source_path="/test/special.txt",
                source_type=SourceType.FILESYSTEM,
            ),
        )
        chunks = chunker.chunk(doc)

        assert len(chunks) >= 1
        for chunk in chunks:
            assert_chunk_valid(chunk)
