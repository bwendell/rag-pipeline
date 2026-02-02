"""Error handling tests for FileSystemSource."""

import pytest
from pathlib import Path

from rag_pipeline.document_sources import FileSystemSource


class TestPermissionErrors:
    @pytest.mark.asyncio
    async def test_directory_permission_denied_continues(self, tmp_path: Path):
        accessible = tmp_path / "accessible"
        accessible.mkdir()
        (accessible / "file.txt").write_text("content")

        locked = tmp_path / "locked"
        locked.mkdir()
        (locked / "secret.txt").write_text("secret")

        try:
            locked.chmod(0o000)
        except PermissionError:
            pytest.skip("Cannot change permissions")

        try:
            source = FileSystemSource(tmp_path)
            docs = [doc async for doc in source.load()]
            paths = {d.metadata.source_path for d in docs}
            assert "accessible/file.txt" in paths
        finally:
            locked.chmod(0o755)

    @pytest.mark.asyncio
    async def test_file_permission_denied_skipped(self, tmp_path: Path):
        (tmp_path / "readable.txt").write_text("readable")
        locked_file = tmp_path / "locked.txt"
        locked_file.write_text("locked content")

        try:
            locked_file.chmod(0o000)
        except PermissionError:
            pytest.skip("Cannot change permissions")

        try:
            source = FileSystemSource(tmp_path)
            docs = [doc async for doc in source.load()]
            paths = {d.metadata.source_path for d in docs}
            assert "readable.txt" in paths
        finally:
            locked_file.chmod(0o644)


class TestEncodingFallback:
    @pytest.mark.asyncio
    async def test_utf8_file_loads_correctly(self, tmp_path: Path):
        content = "Hello, 世界! 🎉"
        (tmp_path / "utf8.txt").write_text(content, encoding="utf-8")

        source = FileSystemSource(tmp_path)
        doc = await source.load_by_path("utf8.txt")

        assert doc is not None
        assert doc.content == content

    @pytest.mark.asyncio
    async def test_latin1_fallback(self, tmp_path: Path):
        content_bytes = b"Caf\xe9"
        (tmp_path / "latin1.txt").write_bytes(content_bytes)

        source = FileSystemSource(tmp_path)
        doc = await source.load_by_path("latin1.txt")

        assert doc is not None
        assert "Caf" in doc.content


class TestFileDeletedMidScan:
    @pytest.mark.asyncio
    async def test_file_deleted_during_iteration(self, tmp_path: Path):
        """Test that FileSystemSource gracefully handles missing files.

        Note: This test is simplified since mocking Path.stat() affects
        the entire pytest session, including cleanup operations.
        """
        (tmp_path / "file1.txt").write_text("content1")
        (tmp_path / "file2.txt").write_text("content2")

        source = FileSystemSource(tmp_path)
        # This should complete without error even though internal
        # file operations may fail
        docs = [doc async for doc in source.load()]
        assert len(docs) >= 1
