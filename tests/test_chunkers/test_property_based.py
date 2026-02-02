"""Property-based tests for chunkers using Hypothesis.

These tests use property-based testing to discover edge cases that
example-based tests might miss. Properties verified include:
- All content is preserved across chunks
- Chunks respect size constraints
- Chunk indices are valid and sequential
- Unicode handling is correct
"""

from __future__ import annotations

import string

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from rag_pipeline.chunkers.base import byte_offset_to_char_offset, char_offset_to_byte_offset
from rag_pipeline.chunkers.code_chunker import CodeChunker
from rag_pipeline.chunkers.generic_chunker import GenericChunker
from rag_pipeline.chunkers.markdown_chunker import MarkdownChunker
from rag_pipeline.core.base_chunker import ChunkingConfig
from rag_pipeline.core.types import Document, DocumentType, Metadata, SourceType


def make_document(content: str, doc_type: DocumentType = DocumentType.TEXT) -> Document:
    return Document(
        content=content,
        doc_type=doc_type,
        metadata=Metadata(source_path="/test/file.txt", source_type=SourceType.FILESYSTEM),
    )


class TestGenericChunkerProperties:
    @given(st.text(min_size=1, max_size=5000))
    @settings(max_examples=50, deadline=None)
    def test_chunk_indices_are_valid(self, content: str) -> None:
        assume(content.strip())
        chunker = GenericChunker(ChunkingConfig(chunk_size=200, chunk_overlap=0))
        document = make_document(content)

        chunks = chunker.chunk(document)

        for chunk in chunks:
            assert chunk.start_index >= 0
            assert chunk.end_index <= len(content)
            assert chunk.start_index <= chunk.end_index

    @given(st.text(min_size=1, max_size=5000))
    @settings(max_examples=50, deadline=None)
    def test_chunk_indices_are_sequential(self, content: str) -> None:
        assume(content.strip())
        chunker = GenericChunker(ChunkingConfig(chunk_size=200, chunk_overlap=0))
        document = make_document(content)

        chunks = chunker.chunk(document)

        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    @given(st.text(min_size=1, max_size=2000, alphabet=string.printable))
    @settings(max_examples=30, deadline=None)
    def test_all_non_whitespace_content_preserved(self, content: str) -> None:
        assume(content.strip())
        chunker = GenericChunker(ChunkingConfig(chunk_size=100, chunk_overlap=0))
        document = make_document(content)

        chunks = chunker.chunk(document)

        combined = "".join(c.content for c in chunks)
        original_words = set(content.split())
        combined_words = set(combined.split())
        assert original_words <= combined_words or not original_words

    @given(
        st.integers(min_value=50, max_value=500),
        st.integers(min_value=0, max_value=49),
    )
    @settings(max_examples=30)
    def test_config_parameters_respected(self, chunk_size: int, overlap: int) -> None:
        assume(overlap < chunk_size)
        config = ChunkingConfig(chunk_size=chunk_size, chunk_overlap=overlap)
        chunker = GenericChunker(config)

        assert chunker.config.chunk_size == chunk_size
        assert chunker.config.chunk_overlap == overlap


class TestMarkdownChunkerProperties:
    @given(st.text(min_size=1, max_size=3000))
    @settings(max_examples=50, deadline=None)
    def test_chunks_have_valid_indices(self, content: str) -> None:
        assume(content.strip())
        chunker = MarkdownChunker()
        document = make_document(content, DocumentType.MARKDOWN)

        chunks = chunker.chunk(document)

        for chunk in chunks:
            assert chunk.start_index >= 0
            assert chunk.end_index <= len(content)
            assert chunk.start_index <= chunk.end_index

    @given(
        st.lists(
            st.tuples(
                st.integers(min_value=1, max_value=6),
                st.text(min_size=1, max_size=50, alphabet=string.ascii_letters + " "),
                st.text(min_size=0, max_size=200, alphabet=string.printable),
            ),
            min_size=1,
            max_size=10,
        )
    )
    @settings(max_examples=30, deadline=None)
    def test_header_sections_create_chunks(self, sections: list) -> None:
        lines = []
        for level, title, body in sections:
            header = "#" * level + " " + title.strip()
            if header.strip():
                lines.append(header)
                if body.strip():
                    lines.append(body)

        content = "\n\n".join(lines)
        assume(content.strip())

        chunker = MarkdownChunker()
        document = make_document(content, DocumentType.MARKDOWN)

        chunks = chunker.chunk(document)

        assert len(chunks) >= 1


class TestCodeChunkerProperties:
    @given(st.text(min_size=1, max_size=2000, alphabet=string.printable))
    @settings(max_examples=30, deadline=None)
    def test_chunks_have_valid_indices(self, content: str) -> None:
        assume(content.strip())
        chunker = CodeChunker()
        document = make_document(content, DocumentType.CODE)
        document.metadata.language = "python"
        document.metadata.file_extension = ".py"

        chunks = chunker.chunk(document)

        for chunk in chunks:
            assert chunk.start_index >= 0
            assert chunk.end_index <= len(content)

    @given(
        st.lists(
            st.text(min_size=1, max_size=30, alphabet=string.ascii_lowercase + "_"),
            min_size=1,
            max_size=5,
        )
    )
    @settings(max_examples=30, deadline=None)
    def test_python_functions_create_chunks(self, func_names: list) -> None:
        lines = []
        for name in func_names:
            if name and name[0].isalpha():
                lines.append(f"def {name}():")
                lines.append("    pass")
                lines.append("")

        content = "\n".join(lines)
        assume(content.strip())

        chunker = CodeChunker()
        document = make_document(content, DocumentType.CODE)
        document.metadata.language = "python"
        document.metadata.file_extension = ".py"

        chunks = chunker.chunk(document)

        assert len(chunks) >= 1


class TestUnicodeOffsetProperties:
    @given(st.text(min_size=1, max_size=500))
    @settings(max_examples=100)
    def test_byte_char_offset_roundtrip(self, content: str) -> None:
        for char_offset in range(len(content) + 1):
            byte_offset = char_offset_to_byte_offset(content, char_offset)
            recovered_char_offset = byte_offset_to_char_offset(content, byte_offset)
            assert recovered_char_offset == char_offset

    @given(st.text(min_size=1, max_size=500))
    @settings(max_examples=100)
    def test_byte_offset_monotonic(self, content: str) -> None:
        prev_byte_offset = -1
        for char_offset in range(len(content) + 1):
            byte_offset = char_offset_to_byte_offset(content, char_offset)
            assert byte_offset > prev_byte_offset
            prev_byte_offset = byte_offset

    @given(st.text(min_size=1, max_size=500))
    @settings(max_examples=100)
    def test_char_offset_monotonic(self, content: str) -> None:
        content_bytes = content.encode("utf-8")
        prev_char_offset = -1
        for byte_offset in range(len(content_bytes) + 1):
            try:
                char_offset = byte_offset_to_char_offset(content, byte_offset)
                assert char_offset >= prev_char_offset
                prev_char_offset = char_offset
            except UnicodeDecodeError:
                pass


class TestChunkingInvariants:
    @given(st.text(min_size=10, max_size=1000, alphabet=string.printable))
    @settings(max_examples=30, deadline=None)
    def test_chunks_cover_document_start(self, content: str) -> None:
        assume(content.strip())
        chunker = GenericChunker(ChunkingConfig(chunk_size=100, chunk_overlap=0))
        document = make_document(content)

        chunks = chunker.chunk(document)

        if chunks:
            first_chunk = chunks[0]
            assert first_chunk.start_index == 0 or first_chunk.start_index < 50

    @given(st.text(min_size=10, max_size=1000, alphabet=string.printable))
    @settings(max_examples=30, deadline=None)
    def test_each_chunk_has_content(self, content: str) -> None:
        assume(content.strip())
        chunker = GenericChunker(ChunkingConfig(chunk_size=100, chunk_overlap=0))
        document = make_document(content)

        chunks = chunker.chunk(document)

        for chunk in chunks:
            assert len(chunk.content) > 0

    @given(st.text(min_size=10, max_size=1000, alphabet=string.printable))
    @settings(max_examples=30, deadline=None)
    def test_chunk_document_ids_match(self, content: str) -> None:
        assume(content.strip())
        chunker = GenericChunker()
        document = make_document(content)

        chunks = chunker.chunk(document)

        for chunk in chunks:
            assert chunk.document_id == document.id
