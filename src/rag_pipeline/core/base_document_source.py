"""DocumentSource Protocol definition.

This module defines the abstract interface for document source implementations.
All document sources (filesystem, Git, Confluence, etc.) must conform to
this protocol.

Example usage:
    >>> from rag_pipeline.core import DocumentSource
    >>>
    >>> class MyDocumentSource:
    ...     async def load(self) -> AsyncIterator[Document]:
    ...         # Load all documents from source
    ...         ...
    >>>
    >>> # Type checking: isinstance works at runtime
    >>> assert isinstance(MyDocumentSource(), DocumentSource)
"""

from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from rag_pipeline.core.types import Document


@runtime_checkable
class DocumentSource(Protocol):
    """Protocol for document source implementations.

    Document sources are responsible for:
    1. Loading documents from various sources (filesystem, APIs, etc.)
    2. Detecting document types and extracting metadata
    3. Supporting incremental updates (loading only changed documents)
    4. Handling authentication and connection management

    All loading methods are async generators to support streaming
    large document sets without loading everything into memory.

    Implementations:
        - FileSystemSource: Local filesystem
        - GitSource (stub): Git repositories
        - ConfluenceSource (stub): Confluence wiki
        - S3Source (stub): AWS S3 / OCI Object Storage
    """

    async def load(self) -> AsyncIterator[Document]:
        """Load all documents from the source.

        Yields documents one at a time to support large document sets
        without loading everything into memory.

        Yields:
            Document objects with content and metadata populated.

        Raises:
            DocumentSourceError: If loading fails.

        Example:
            >>> async for doc in source.load():
            ...     print(f"Loaded: {doc.metadata.source_path}")
            ...     chunks = chunker.chunk(doc)
        """
        ...

    async def load_changed(
        self,
        since: datetime
    ) -> AsyncIterator[Document]:
        """Load only documents changed since a given time.

        Useful for incremental indexing - only process documents
        that have been modified since the last indexing run.

        Args:
            since: Datetime to compare against. Only documents
                modified after this time are returned.

        Yields:
            Document objects that were modified since the given time.

        Raises:
            DocumentSourceError: If loading fails.

        Example:
            >>> last_run = datetime(2024, 1, 1)
            >>> async for doc in source.load_changed(since=last_run):
            ...     print(f"Changed: {doc.metadata.source_path}")
        """
        ...

    async def load_by_path(self, path: str) -> Document | None:
        """Load a single document by its path.

        Args:
            path: Path or identifier for the document.

        Returns:
            The document if found, None otherwise.

        Raises:
            DocumentSourceError: If loading fails (not for missing docs).

        Example:
            >>> doc = await source.load_by_path("/path/to/file.py")
            >>> if doc:
            ...     print(f"Found: {doc.content[:100]}")
        """
        ...

    async def list_paths(self) -> list[str]:
        """List all available document paths.

        Returns a list of paths/identifiers without loading content.
        Useful for showing what's available or planning batch operations.

        Returns:
            List of document paths/identifiers.

        Raises:
            DocumentSourceError: If listing fails.

        Example:
            >>> paths = await source.list_paths()
            >>> print(f"Found {len(paths)} documents")
        """
        ...

    async def get_changed_paths(
        self,
        since: datetime
    ) -> list[str]:
        """Get paths of documents changed since a given time.

        Lighter weight than load_changed() - just returns paths
        without loading content.

        Args:
            since: Datetime to compare against.

        Returns:
            List of paths for changed documents.

        Raises:
            DocumentSourceError: If checking fails.
        """
        ...

    async def get_deleted_paths(
        self,
        known_paths: list[str]
    ) -> list[str]:
        """Find paths that no longer exist in the source.

        Compares a list of known paths against current source state
        to find documents that have been deleted.

        Args:
            known_paths: List of paths previously indexed.

        Returns:
            List of paths that no longer exist.

        Example:
            >>> indexed = ["file1.py", "file2.py", "deleted.py"]
            >>> deleted = await source.get_deleted_paths(indexed)
            >>> print(f"Deleted: {deleted}")  # ["deleted.py"]
        """
        ...

    def get_source_info(self) -> dict[str, Any]:
        """Get information about the document source.

        Returns:
            Dictionary containing source information:
                - type: Source type (e.g., "filesystem", "git")
                - root: Root path or URL
                - patterns: File patterns being scanned
                - ignore_patterns: Patterns being ignored

        Example:
            >>> info = source.get_source_info()
            >>> print(f"Source: {info['type']} at {info['root']}")
        """
        ...

    async def health_check(self) -> bool:
        """Check if the document source is accessible.

        Returns:
            True if the source is accessible, False otherwise.

        Example:
            >>> if await source.health_check():
            ...     print("Source is accessible")
        """
        ...


@runtime_checkable
class WatchableDocumentSource(DocumentSource, Protocol):
    """Extended protocol for document sources that support watching.

    Adds real-time change detection for sources that support it
    (e.g., filesystem watchers, webhooks).
    """

    async def watch(
        self,
        callback: "DocumentChangeCallback"
    ) -> "WatchHandle":
        """Start watching for document changes.

        Args:
            callback: Async function called when documents change.

        Returns:
            Handle to stop watching.

        Example:
            >>> async def on_change(event):
            ...     print(f"{event.type}: {event.path}")
            >>> handle = await source.watch(on_change)
            >>> # Later...
            >>> await handle.stop()
        """
        ...


class DocumentChangeEvent:
    """Event representing a document change."""

    path: str
    change_type: str  # "created", "modified", "deleted"
    timestamp: datetime


DocumentChangeCallback = Callable[[DocumentChangeEvent], Awaitable[None]]
"""Callback type for document change notifications."""


class WatchHandle(Protocol):
    """Handle for stopping a document watch."""

    async def stop(self) -> None:
        """Stop watching for changes."""
        ...

    @property
    def is_active(self) -> bool:
        """Check if the watch is still active."""
        ...
