"""Filesystem document source implementation.

Loads documents from a local directory with support for:
- Recursive directory scanning
- Pattern-based filtering
- Incremental updates based on modification time
- Automatic document type detection
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
from typing import Any

import aiofiles
import structlog

from rag_pipeline.core.exceptions import DocumentSourceError
from rag_pipeline.core.types import Document, SourceType
from rag_pipeline.document_sources.base import (
    AbstractDocumentSource,
    DocumentSourceConfig,
)

logger = structlog.get_logger(__name__)


class FileSystemSource(AbstractDocumentSource):
    """Document source for local filesystem.

    Scans a directory recursively for documents, respecting
    include/exclude patterns and detecting document types.

    SECURITY FEATURES:
        - Path traversal protection in load_by_path
        - Symlink cycle detection to prevent infinite loops
        - Binary content sniffing to prevent garbage ingestion
        - Sensitive file exclusion (.env, credentials, keys)

    Example:
        >>> source = FileSystemSource(
        ...     root_path="/path/to/codebase",
        ...     config=DocumentSourceConfig(
        ...         include_patterns=["*.py", "*.md"],
        ...         exclude_patterns=["tests/**"],
        ...     )
        ... )
        >>> async for doc in source.load():
        ...     print(f"Loaded: {doc.metadata.source_path}")
    """

    def __init__(
        self,
        root_path: str | Path,
        config: DocumentSourceConfig | None = None,
    ) -> None:
        """Initialize filesystem source.

        Args:
            root_path: Root directory to scan.
            config: Source configuration.

        Raises:
            DocumentSourceError: If root_path doesn't exist.
        """
        super().__init__(config, SourceType.FILESYSTEM)
        self.root_path = Path(root_path).resolve()
        self._visited_inodes: set[tuple[int, int]] = set()  # (device, inode) for cycle detection

        if not self.root_path.exists():
            raise DocumentSourceError(f"Root path does not exist: {self.root_path}")
        if not self.root_path.is_dir():
            raise DocumentSourceError(f"Root path is not a directory: {self.root_path}")

    async def load(self) -> AsyncIterator[Document]:
        """Load all documents from the filesystem.

        Yields documents one at a time, skipping binary files
        and files matching exclude patterns.

        Yields:
            Document objects with content and metadata.

        Raises:
            DocumentSourceError: If critical errors occur during scanning.
        """
        logger.info("Starting filesystem scan", root=str(self.root_path))
        stats = {"scanned": 0, "loaded": 0, "skipped": 0, "errors": 0}

        try:
            async for doc in self._scan_directory(self.root_path, stats):
                yield doc
        finally:
            logger.info(
                "Filesystem scan complete",
                **stats,
                root=str(self.root_path),
            )

    async def _scan_directory(
        self,
        directory: Path,
        stats: dict[str, int],
    ) -> AsyncIterator[Document]:
        """Recursively scan a directory.

        SECURITY: Tracks visited directories by (device, inode) to prevent
        infinite loops from symlink cycles.

        Args:
            directory: Directory to scan.
            stats: Statistics dictionary to update.

        Yields:
            Document objects.
        """
        # SECURITY: Cycle detection using (device, inode)
        try:
            dir_stat = directory.stat()
            dir_key = (dir_stat.st_dev, dir_stat.st_ino)
            if dir_key in self._visited_inodes:
                logger.debug("Skipping already visited directory (cycle)", path=str(directory))
                return
            self._visited_inodes.add(dir_key)
        except OSError as e:
            logger.warning("Cannot stat directory", path=str(directory), error=str(e))
            stats["errors"] += 1
            return

        try:
            entries = list(directory.iterdir())
        except PermissionError:
            logger.warning("Permission denied", path=str(directory))
            stats["errors"] += 1
            return
        except OSError as e:
            logger.warning("Error reading directory", path=str(directory), error=str(e))
            stats["errors"] += 1
            return

        for entry in entries:
            # Handle symlinks
            if entry.is_symlink():
                if not self.config.follow_symlinks:
                    stats["skipped"] += 1
                    continue
                # SECURITY: For symlinks we're following, check they stay within root
                try:
                    resolved = entry.resolve()
                    resolved.relative_to(self.root_path.resolve())
                except (ValueError, OSError):
                    logger.debug("Skipping symlink outside root", path=str(entry))
                    stats["skipped"] += 1
                    continue

            # Get relative path for pattern matching
            try:
                rel_path = entry.relative_to(self.root_path)
            except ValueError:
                rel_path = entry

            rel_path_str = str(rel_path)

            if entry.is_dir():
                # CORRECTNESS: For directories, apply EXCLUDE-only pruning
                # Include patterns are for files, not directories
                # This prevents include patterns like "*.py" from blocking recursion
                if self._should_exclude_directory(rel_path_str):
                    stats["skipped"] += 1
                    continue
                # Recurse into subdirectory
                async for doc in self._scan_directory(entry, stats):
                    yield doc
            elif entry.is_file():
                stats["scanned"] += 1

                # Check patterns
                if not self._should_include(rel_path_str):
                    stats["skipped"] += 1
                    continue

                # Check if text file
                if not self._is_text_file(entry):
                    stats["skipped"] += 1
                    continue

                # Check file size
                try:
                    size = entry.stat().st_size
                    if size > self.config.max_file_size_bytes:
                        logger.debug("Skipping large file", path=rel_path_str, size=size)
                        stats["skipped"] += 1
                        continue
                    if size == 0:
                        logger.debug("Skipping empty file", path=rel_path_str)
                        stats["skipped"] += 1
                        continue
                except OSError:
                    stats["errors"] += 1
                    continue

                # Load document
                doc = await self._load_file(entry)
                if doc:
                    stats["loaded"] += 1
                    yield doc
                else:
                    stats["errors"] += 1

    async def _load_file(self, path: Path) -> Document | None:
        """Load a single file into a Document.

        SECURITY: Uses binary content sniffing to prevent garbage ingestion.
        CORRECTNESS: Catches only specific exceptions, not generic Exception.

        Args:
            path: Path to the file.

        Returns:
            Document if successful, None if failed.
        """
        try:
            stat = path.stat()
            # Use mtime for both timestamps (ctime is not creation time on Linux)
            updated_at = datetime.fromtimestamp(stat.st_mtime)
            created_at = updated_at  # Best we can do portably

            # SECURITY: Read raw bytes first for binary detection
            async with aiofiles.open(path, "rb") as f:
                raw_content = await f.read()

            # SECURITY: Check for binary content before decoding
            if self._is_binary_content(raw_content):
                logger.debug("Skipping binary content", path=str(path))
                return None

            # Try to decode with configured encoding
            try:
                content = raw_content.decode(self.config.encoding)
            except UnicodeDecodeError:
                # Fallback to latin-1 only for text files that failed UTF-8
                try:
                    content = raw_content.decode("latin-1")
                    logger.debug("Used latin-1 fallback", path=str(path))
                except UnicodeDecodeError:
                    logger.warning("Failed to decode file", path=str(path))
                    return None

            # Use path relative to root for source_path
            try:
                rel_path = str(path.relative_to(self.root_path))
            except ValueError:
                rel_path = str(path)

            return self._create_document(
                content=content,
                path=rel_path,
                created_at=created_at,
                updated_at=updated_at,
            )

        except PermissionError:
            logger.warning("Permission denied reading file", path=str(path))
            return None
        except FileNotFoundError:
            logger.warning("File not found (deleted during scan?)", path=str(path))
            return None
        except OSError as e:
            logger.warning("Error loading file", path=str(path), error=str(e))
            return None

    async def load_changed(
        self,
        since: datetime,
    ) -> AsyncIterator[Document]:
        """Load documents modified since a given time.

        Args:
            since: Only load documents modified after this time.

        Yields:
            Document objects for modified files.
        """
        logger.info("Loading changed documents", since=since.isoformat())

        async for doc in self.load():
            if doc.metadata.updated_at > since:
                yield doc

    async def load_by_path(self, path: str) -> Document | None:
        """Load a single document by relative path.

        SECURITY: This method validates that the path stays within root_path
        to prevent path traversal attacks (e.g., "../../../etc/passwd").

        Args:
            path: Path relative to root directory.

        Returns:
            Document if found and valid, None otherwise.
        """
        # SECURITY: Resolve and validate path containment
        try:
            full_path = (self.root_path / path).resolve()
        except (ValueError, OSError):
            logger.warning("Invalid path", path=path)
            return None

        # SECURITY: Ensure resolved path is within root_path
        # This prevents path traversal attacks like "../../../etc/passwd"
        try:
            full_path.relative_to(self.root_path.resolve())
        except ValueError:
            logger.warning(
                "Path traversal attempt blocked",
                path=path,
                resolved=str(full_path),
                root=str(self.root_path),
            )
            return None

        if not full_path.exists() or not full_path.is_file():
            return None

        return await self._load_file(full_path)

    async def list_paths(self) -> list[str]:
        """List all document paths in the source.

        Returns:
            List of relative paths.
        """
        paths: list[str] = []

        async for doc in self.load():
            paths.append(doc.metadata.source_path)

        return paths

    async def get_changed_paths(self, since: datetime) -> list[str]:
        """Get paths of documents changed since a given time.

        Args:
            since: Cutoff datetime.

        Returns:
            List of paths for changed documents.
        """
        paths: list[str] = []

        for path in self._iter_paths_sync():
            try:
                full_path = self.root_path / path
                mtime = datetime.fromtimestamp(full_path.stat().st_mtime)
                if mtime > since:
                    paths.append(path)
            except OSError:
                continue

        return paths

    def _iter_paths_sync(self) -> list[str]:
        """Synchronously iterate all matching paths.

        Returns:
            List of relative paths.
        """
        paths: list[str] = []

        for root, dirs, files in os.walk(self.root_path, followlinks=self.config.follow_symlinks):
            root_path = Path(root)

            # Filter directories in-place
            dirs[:] = [
                d
                for d in dirs
                if self._should_include(str((root_path / d).relative_to(self.root_path)) + "/")
            ]

            for file in files:
                file_path = root_path / file
                try:
                    rel_path = str(file_path.relative_to(self.root_path))
                except ValueError:
                    continue

                if self._should_include(rel_path) and self._is_text_file(file_path):
                    paths.append(rel_path)

        return paths

    async def get_deleted_paths(self, known_paths: list[str]) -> list[str]:
        """Find paths that no longer exist.

        Args:
            known_paths: Previously indexed paths.

        Returns:
            Paths that have been deleted.
        """
        current_paths = set(self._iter_paths_sync())
        return [p for p in known_paths if p not in current_paths]

    def get_source_info(self) -> dict[str, Any]:
        """Get information about this filesystem source.

        Returns:
            Dictionary with source details.
        """
        return {
            "type": "filesystem",
            "root": str(self.root_path),
            "patterns": {
                "include": self.config.include_patterns,
                "exclude": self.config.exclude_patterns + self.DEFAULT_EXCLUDE_PATTERNS,
            },
            "max_file_size_bytes": self.config.max_file_size_bytes,
            "follow_symlinks": self.config.follow_symlinks,
            "encoding": self.config.encoding,
        }

    async def health_check(self) -> bool:
        """Check if the root directory is accessible.

        Returns:
            True if accessible.
        """
        try:
            return self.root_path.exists() and self.root_path.is_dir()
        except OSError:
            return False
