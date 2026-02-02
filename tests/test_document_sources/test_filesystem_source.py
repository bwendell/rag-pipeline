"""Tests for FileSystemSource."""

import pytest
from pathlib import Path
from datetime import datetime

from rag_pipeline.document_sources import FileSystemSource, DocumentSourceConfig
from rag_pipeline.core.types import DocumentType, SourceType
from rag_pipeline.core.exceptions import DocumentSourceError


class TestFileSystemSource:
    @pytest.fixture
    def source(self, sample_directory: Path) -> FileSystemSource:
        return FileSystemSource(sample_directory)

    @pytest.mark.asyncio
    async def test_load_all_documents(self, source: FileSystemSource):
        docs = [doc async for doc in source.load()]
        assert len(docs) >= 4
        python_docs = [d for d in docs if d.doc_type == DocumentType.CODE]
        markdown_docs = [d for d in docs if d.doc_type == DocumentType.MARKDOWN]
        assert len(python_docs) >= 3
        assert len(markdown_docs) >= 2

    @pytest.mark.asyncio
    async def test_excludes_pycache(self, source: FileSystemSource):
        docs = [doc async for doc in source.load()]
        paths = [d.metadata.source_path for d in docs]
        assert not any("__pycache__" in p for p in paths)

    @pytest.mark.asyncio
    async def test_excludes_git(self, source: FileSystemSource):
        docs = [doc async for doc in source.load()]
        paths = [d.metadata.source_path for d in docs]
        assert not any(".git" in p for p in paths)

    @pytest.mark.asyncio
    async def test_include_patterns(self, sample_directory: Path):
        source = FileSystemSource(
            sample_directory,
            config=DocumentSourceConfig(include_patterns=["*.py"]),
        )
        docs = [doc async for doc in source.load()]
        for doc in docs:
            assert doc.doc_type == DocumentType.CODE
            assert doc.metadata.language == "python"

    @pytest.mark.asyncio
    async def test_exclude_patterns(self, sample_directory: Path):
        source = FileSystemSource(
            sample_directory,
            config=DocumentSourceConfig(exclude_patterns=["tests/**"]),
        )
        docs = [doc async for doc in source.load()]
        paths = [d.metadata.source_path for d in docs]
        assert not any("tests" in p for p in paths)

    @pytest.mark.asyncio
    async def test_load_by_path(self, source: FileSystemSource):
        doc = await source.load_by_path("src/main.py")
        assert doc is not None
        assert "def main" in doc.content
        assert doc.doc_type == DocumentType.CODE
        assert doc.metadata.language == "python"

    @pytest.mark.asyncio
    async def test_load_by_path_not_found(self, source: FileSystemSource):
        doc = await source.load_by_path("nonexistent.py")
        assert doc is None

    @pytest.mark.asyncio
    async def test_list_paths(self, source: FileSystemSource):
        paths = await source.list_paths()
        assert len(paths) >= 4
        assert any("main.py" in p for p in paths)
        assert any("README.md" in p for p in paths)

    @pytest.mark.asyncio
    async def test_health_check(self, source: FileSystemSource):
        assert await source.health_check() is True

    @pytest.mark.asyncio
    async def test_health_check_missing_dir(self, tmp_path: Path):
        with pytest.raises(DocumentSourceError):
            FileSystemSource(tmp_path / "nonexistent")

    def test_get_source_info(self, source: FileSystemSource, sample_directory: Path):
        info = source.get_source_info()
        assert info["type"] == "filesystem"
        assert info["root"] == str(sample_directory)

    @pytest.mark.asyncio
    async def test_metadata_populated(self, source: FileSystemSource):
        doc = await source.load_by_path("src/main.py")
        assert doc is not None
        assert doc.metadata.source_type == SourceType.FILESYSTEM
        assert doc.metadata.source_path == "src/main.py"
        assert doc.metadata.language == "python"
        assert doc.metadata.file_extension == ".py"

    @pytest.mark.asyncio
    async def test_get_deleted_paths(self, source: FileSystemSource):
        known = ["src/main.py", "src/deleted.py", "docs/README.md"]
        deleted = await source.get_deleted_paths(known)
        assert "src/deleted.py" in deleted
        assert "src/main.py" not in deleted

    @pytest.mark.asyncio
    async def test_max_file_size(self, sample_directory: Path):
        large_file = sample_directory / "large.txt"
        large_file.write_text("x" * 100_000)
        source = FileSystemSource(
            sample_directory,
            config=DocumentSourceConfig(max_file_size_bytes=1000),
        )
        docs = [doc async for doc in source.load()]
        paths = [d.metadata.source_path for d in docs]
        assert "large.txt" not in paths
