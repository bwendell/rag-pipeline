"""Tests for stub document sources."""

import pytest

from rag_pipeline.document_sources import GitSource, ConfluenceSource, S3Source
from rag_pipeline.document_sources.stubs.git_source import GitSourceConfig
from rag_pipeline.document_sources.stubs.confluence_source import ConfluenceSourceConfig
from rag_pipeline.document_sources.stubs.s3_source import S3SourceConfig
from rag_pipeline.core.exceptions import NotImplementedSourceError


class TestGitSourceStub:
    def test_creates_successfully(self):
        config = GitSourceConfig(repository_url="https://github.com/org/repo.git")
        source = GitSource(config)
        assert source is not None

    @pytest.mark.asyncio
    async def test_load_raises_not_implemented(self):
        source = GitSource()
        with pytest.raises(NotImplementedSourceError):
            async for _ in source.load():
                pass

    @pytest.mark.asyncio
    async def test_health_check_returns_false(self):
        source = GitSource()
        assert await source.health_check() is False

    def test_get_source_info_shows_stub(self):
        source = GitSource()
        info = source.get_source_info()
        assert info["status"] == "stub"


class TestConfluenceSourceStub:
    def test_creates_successfully(self):
        config = ConfluenceSourceConfig(base_url="https://wiki.example.com")
        source = ConfluenceSource(config)
        assert source is not None

    @pytest.mark.asyncio
    async def test_load_raises_not_implemented(self):
        source = ConfluenceSource()
        with pytest.raises(NotImplementedSourceError):
            async for _ in source.load():
                pass


class TestS3SourceStub:
    def test_creates_successfully(self):
        config = S3SourceConfig(bucket_name="my-bucket")
        source = S3Source(config)
        assert source is not None

    @pytest.mark.asyncio
    async def test_load_raises_not_implemented(self):
        source = S3Source()
        with pytest.raises(NotImplementedSourceError):
            async for _ in source.load():
                pass
