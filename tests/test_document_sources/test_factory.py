"""Tests for DocumentSourceFactory."""

import pytest
from pathlib import Path

from rag_pipeline.document_sources import (
    DocumentSourceFactory,
    FileSystemSource,
    GitSource,
    ConfluenceSource,
    S3Source,
)
from rag_pipeline.core.types import SourceType
from rag_pipeline.core.exceptions import ConfigurationError


class TestDocumentSourceFactory:
    @pytest.fixture
    def factory(self) -> DocumentSourceFactory:
        return DocumentSourceFactory()

    def test_create_filesystem(self, factory: DocumentSourceFactory, tmp_path: Path):
        (tmp_path / "file.txt").write_text("content")
        source = factory.create_filesystem(tmp_path)
        assert isinstance(source, FileSystemSource)

    def test_create_filesystem_with_patterns(self, factory: DocumentSourceFactory, tmp_path: Path):
        source = factory.create_filesystem(
            tmp_path,
            include_patterns=["*.py"],
            exclude_patterns=["tests/**"],
        )
        assert source.config.include_patterns == ["*.py"]
        assert source.config.exclude_patterns == ["tests/**"]

    def test_create_by_type(self, factory: DocumentSourceFactory, tmp_path: Path):
        source = factory.create(SourceType.FILESYSTEM, root_path=tmp_path)
        assert isinstance(source, FileSystemSource)

    def test_create_git_stub(self, factory: DocumentSourceFactory):
        source = factory.create_git("https://github.com/org/repo.git")
        assert isinstance(source, GitSource)

    def test_create_confluence_stub(self, factory: DocumentSourceFactory):
        source = factory.create_confluence("https://wiki.example.com")
        assert isinstance(source, ConfluenceSource)

    def test_create_s3_stub(self, factory: DocumentSourceFactory):
        source = factory.create_s3("my-bucket", prefix="docs/")
        assert isinstance(source, S3Source)

    def test_is_implemented(self):
        assert DocumentSourceFactory.is_implemented(SourceType.FILESYSTEM) is True
        assert DocumentSourceFactory.is_implemented(SourceType.GIT) is False
        assert DocumentSourceFactory.is_implemented(SourceType.CONFLUENCE) is False
        assert DocumentSourceFactory.is_implemented(SourceType.S3) is False

    def test_get_available_sources(self):
        sources = DocumentSourceFactory.get_available_sources()
        assert SourceType.FILESYSTEM in sources
        assert SourceType.GIT in sources

    def test_unknown_source_type(self, factory: DocumentSourceFactory):
        with pytest.raises(ConfigurationError):
            factory.create("unknown_type")
