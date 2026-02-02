"""Security tests for FileSystemSource."""

import pytest
from pathlib import Path
import asyncio

from rag_pipeline.document_sources import FileSystemSource, DocumentSourceConfig


class TestPathTraversalPrevention:
    @pytest.mark.asyncio
    async def test_load_by_path_rejects_parent_traversal(self, tmp_path: Path):
        root = tmp_path / "root"
        root.mkdir()
        (tmp_path / "outside.txt").write_text("SENSITIVE DATA")
        (root / "inside.txt").write_text("safe content")

        source = FileSystemSource(root)
        doc = await source.load_by_path("../outside.txt")
        assert doc is None, "Path traversal should return None"

        doc = await source.load_by_path("../../outside.txt")
        assert doc is None

        doc = await source.load_by_path("inside.txt")
        assert doc is not None

    @pytest.mark.asyncio
    async def test_load_by_path_rejects_absolute_path(self, tmp_path: Path):
        root = tmp_path / "root"
        root.mkdir()
        (root / "file.txt").write_text("content")

        source = FileSystemSource(root)
        doc = await source.load_by_path("/etc/passwd")
        assert doc is None


class TestEnvFileExclusion:
    @pytest.mark.asyncio
    async def test_env_file_excluded_by_default(self, tmp_path: Path):
        (tmp_path / ".env").write_text("SECRET_KEY=abc123")
        (tmp_path / "app.py").write_text("print('hello')")

        source = FileSystemSource(tmp_path)
        docs = [doc async for doc in source.load()]
        paths = {d.metadata.source_path for d in docs}

        assert ".env" not in paths
        assert "app.py" in paths

    @pytest.mark.asyncio
    async def test_env_variants_excluded(self, tmp_path: Path):
        (tmp_path / ".env.local").write_text("LOCAL_SECRET=xyz")
        (tmp_path / ".env.production").write_text("PROD_SECRET=xyz")
        (tmp_path / "nested").mkdir()
        (tmp_path / "nested/.env").write_text("NESTED_SECRET=xyz")
        (tmp_path / "app.py").write_text("print('hello')")

        source = FileSystemSource(tmp_path)
        docs = [doc async for doc in source.load()]
        paths = {d.metadata.source_path for d in docs}

        assert ".env.local" not in paths
        assert ".env.production" not in paths
        assert "nested/.env" not in paths


class TestSymlinkSecurity:
    @pytest.mark.asyncio
    async def test_symlink_cycle_does_not_hang(self, tmp_path: Path):
        root = tmp_path / "root"
        root.mkdir()
        subdir = root / "subdir"
        subdir.mkdir()

        try:
            (subdir / "link").symlink_to(subdir, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("Symlinks not supported on this platform")

        source = FileSystemSource(root, config=DocumentSourceConfig(follow_symlinks=True))

        async def collect():
            return [doc async for doc in source.load()]

        try:
            await asyncio.wait_for(collect(), timeout=5.0)
        except asyncio.TimeoutError:
            pytest.fail("Symlink cycle caused infinite loop")

    @pytest.mark.asyncio
    async def test_symlink_outside_root_skipped(self, tmp_path: Path):
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("SECRET")

        root = tmp_path / "root"
        root.mkdir()
        (root / "safe.txt").write_text("safe")

        try:
            (root / "escape").symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("Symlinks not supported")

        source = FileSystemSource(root, config=DocumentSourceConfig(follow_symlinks=True))
        docs = [doc async for doc in source.load()]
        paths = {d.metadata.source_path for d in docs}

        assert "safe.txt" in paths
        assert "escape/secret.txt" not in paths


class TestBinaryContentDetection:
    @pytest.mark.asyncio
    async def test_binary_without_extension_skipped(self, tmp_path: Path):
        (tmp_path / "binary_blob").write_bytes(b"\x00\x01\x02\x03" * 100)
        (tmp_path / "text.txt").write_text("hello world")

        source = FileSystemSource(tmp_path)
        docs = [doc async for doc in source.load()]
        paths = {d.metadata.source_path for d in docs}

        assert "binary_blob" not in paths
        assert "text.txt" in paths

    @pytest.mark.asyncio
    async def test_binary_with_text_extension_skipped(self, tmp_path: Path):
        (tmp_path / "fake.txt").write_bytes(b"\x00\x01\x02" * 1000)
        (tmp_path / "real.txt").write_text("real text content")

        source = FileSystemSource(tmp_path)
        docs = [doc async for doc in source.load()]
        paths = {d.metadata.source_path for d in docs}

        assert "fake.txt" not in paths
        assert "real.txt" in paths
