"""Tests for core data types."""

import pytest
from datetime import datetime

from rag_pipeline.core import (
    Chunk,
    Document,
    DocumentType,
    Metadata,
    QueryContext,
    SearchResult,
    SourceType,
)


class TestMetadata:
    def test_default_values(self):
        meta = Metadata()

        assert meta.source_path == ""
        assert meta.source_type == SourceType.UNKNOWN
        assert meta.language is None
        assert meta.file_extension is None
        assert isinstance(meta.created_at, datetime)
        assert isinstance(meta.custom, dict)

    def test_with_values(self):
        meta = Metadata(
            source_path="/path/to/file.py",
            source_type=SourceType.FILESYSTEM,
            language="python",
            file_extension=".py",
        )

        assert meta.source_path == "/path/to/file.py"
        assert meta.source_type == SourceType.FILESYSTEM
        assert meta.language == "python"
        assert meta.file_extension == ".py"

    def test_custom_fields(self):
        meta = Metadata()
        meta.custom["author"] = "John Doe"
        meta.custom["version"] = "1.0"

        assert meta.custom["author"] == "John Doe"
        assert meta.custom["version"] == "1.0"

    def test_to_dict(self):
        meta = Metadata(
            source_path="/path/to/file.py",
            source_type=SourceType.FILESYSTEM,
            language="python",
        )
        meta.custom["tag"] = "important"

        result = meta.to_dict()

        assert result["source_path"] == "/path/to/file.py"
        assert result["source_type"] == "filesystem"
        assert result["language"] == "python"
        assert result["tag"] == "important"
        assert "created_at" in result
        assert "updated_at" in result

    def test_from_dict(self):
        data = {
            "source_path": "/path/to/file.py",
            "source_type": "filesystem",
            "language": "python",
            "created_at": "2024-01-01T12:00:00",
            "updated_at": "2024-01-01T12:00:00",
            "custom_field": "value",
        }

        meta = Metadata.from_dict(data)

        assert meta.source_path == "/path/to/file.py"
        assert meta.source_type == SourceType.FILESYSTEM
        assert meta.language == "python"
        assert meta.custom["custom_field"] == "value"


class TestDocument:
    def test_creation(self):
        doc = Document(
            content="def hello(): pass",
            doc_type=DocumentType.CODE,
        )

        assert doc.content == "def hello(): pass"
        assert doc.doc_type == DocumentType.CODE
        assert doc.id is not None
        assert isinstance(doc.metadata, Metadata)

    def test_with_metadata(self):
        meta = Metadata(source_path="hello.py", language="python")
        doc = Document(
            content="def hello(): pass",
            doc_type=DocumentType.CODE,
            metadata=meta,
        )

        assert doc.metadata.source_path == "hello.py"
        assert doc.metadata.language == "python"

    def test_empty_content_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            Document(content="")

    def test_auto_generated_id(self):
        doc1 = Document(content="content1")
        doc2 = Document(content="content2")

        assert doc1.id != doc2.id

    def test_content_hash(self):
        doc1 = Document(content="hello world")
        doc2 = Document(content="hello world")
        doc3 = Document(content="different content")

        assert doc1.content_hash == doc2.content_hash
        assert doc1.content_hash != doc3.content_hash
        assert len(doc1.content_hash) == 16


class TestChunk:
    def test_creation(self):
        chunk = Chunk(
            content="def hello(): pass",
            document_id="doc-123",
        )

        assert chunk.content == "def hello(): pass"
        assert chunk.document_id == "doc-123"
        assert chunk.id is not None
        assert chunk.embedding is None
        assert chunk.start_index == 0
        assert chunk.end_index == 0
        assert chunk.chunk_index == 0

    def test_with_embedding(self):
        embedding = [0.1, 0.2, 0.3]
        chunk = Chunk(
            content="test",
            document_id="doc-123",
            embedding=embedding,
        )

        assert chunk.embedding == embedding
        assert chunk.has_embedding is True

    def test_without_embedding(self):
        chunk = Chunk(content="test", document_id="doc-123")

        assert chunk.has_embedding is False

    def test_empty_content_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            Chunk(content="", document_id="doc-123")

    def test_position_tracking(self):
        chunk = Chunk(
            content="hello",
            document_id="doc-123",
            start_index=10,
            end_index=15,
            chunk_index=2,
        )

        assert chunk.start_index == 10
        assert chunk.end_index == 15
        assert chunk.chunk_index == 2


class TestSearchResult:
    def test_creation(self):
        chunk = Chunk(content="test", document_id="doc-123")
        result = SearchResult(chunk=chunk, score=0.95, rank=1)

        assert result.chunk == chunk
        assert result.score == 0.95
        assert result.rank == 1

    def test_default_rank(self):
        chunk = Chunk(content="test", document_id="doc-123")
        result = SearchResult(chunk=chunk, score=0.9)

        assert result.rank == 1


class TestQueryContext:
    def test_creation(self):
        context = QueryContext(query="How does auth work?")

        assert context.query == "How does auth work?"
        assert context.chunks == []
        assert context.max_tokens == 4000

    def test_with_chunks(self):
        chunk1 = Chunk(content="auth code", document_id="doc1")
        chunk2 = Chunk(content="more auth", document_id="doc2")

        context = QueryContext(
            query="How does auth work?",
            chunks=[chunk1, chunk2],
        )

        assert context.total_chunks == 2

    def test_format_context_empty(self):
        context = QueryContext(query="test")

        assert context.format_context() == "No relevant context found."

    def test_format_context_with_chunks(self):
        chunk = Chunk(
            content="def authenticate(): pass",
            document_id="doc1",
            metadata=Metadata(source_path="auth.py"),
        )
        context = QueryContext(query="test", chunks=[chunk])

        formatted = context.format_context()

        assert "[1] Source: auth.py" in formatted
        assert "def authenticate(): pass" in formatted


class TestEnums:
    def test_document_type_values(self):
        assert DocumentType.CODE.value == "code"
        assert DocumentType.MARKDOWN.value == "markdown"
        assert DocumentType.TEXT.value == "text"
        assert DocumentType.RUNBOOK.value == "runbook"
        assert DocumentType.UNKNOWN.value == "unknown"

    def test_source_type_values(self):
        assert SourceType.FILESYSTEM.value == "filesystem"
        assert SourceType.GIT.value == "git"
        assert SourceType.CONFLUENCE.value == "confluence"
        assert SourceType.S3.value == "s3"
        assert SourceType.OCI_OBJECT_STORAGE.value == "oci_object_storage"
