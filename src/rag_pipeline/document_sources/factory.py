"""Factory for creating document sources.

Provides centralized creation of document sources based on configuration,
with support for registration of custom sources.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Type

from rag_pipeline.document_sources.base import (
    AbstractDocumentSource,
    DocumentSourceConfig,
)
from rag_pipeline.document_sources.filesystem_source import FileSystemSource
from rag_pipeline.document_sources.stubs.git_source import GitSource, GitSourceConfig
from rag_pipeline.document_sources.stubs.confluence_source import (
    ConfluenceSource,
    ConfluenceSourceConfig,
)
from rag_pipeline.document_sources.stubs.s3_source import S3Source, S3SourceConfig
from rag_pipeline.core.types import SourceType
from rag_pipeline.core.exceptions import ConfigurationError


class DocumentSourceFactory:
    """Factory for creating document sources.

    Maintains a registry of source types and creates appropriate
    instances based on configuration.

    Example:
        >>> factory = DocumentSourceFactory()
        >>>
        >>> # Create filesystem source
        >>> source = factory.create_filesystem("/path/to/code")
        >>>
        >>> # Create from source type
        >>> source = factory.create(
        ...     source_type=SourceType.FILESYSTEM,
        ...     root_path="/path/to/code"
        ... )
    """

    # Registry of source types to classes
    _registry: dict[SourceType, Type[AbstractDocumentSource]] = {
        SourceType.FILESYSTEM: FileSystemSource,
        SourceType.GIT: GitSource,
        SourceType.CONFLUENCE: ConfluenceSource,
        SourceType.S3: S3Source,
        SourceType.OCI_OBJECT_STORAGE: S3Source,  # Uses same class with config flag
    }

    def __init__(self, default_config: DocumentSourceConfig | None = None) -> None:
        """Initialize factory with optional default configuration.

        Args:
            default_config: Default configuration for sources.
        """
        self.default_config = default_config or DocumentSourceConfig()

    def create(
        self,
        source_type: SourceType | str,
        **kwargs: Any,
    ) -> AbstractDocumentSource:
        """Create a document source by type.

        Args:
            source_type: Type of source to create.
            **kwargs: Arguments passed to source constructor.

        Returns:
            Document source instance.

        Raises:
            ConfigurationError: If source type is not registered.
        """
        if isinstance(source_type, str):
            try:
                source_type = SourceType(source_type)
            except ValueError:
                raise ConfigurationError(f"Unknown source type: {source_type}")

        if source_type not in self._registry:
            raise ConfigurationError(f"No source registered for type: {source_type}")

        source_cls = self._registry[source_type]

        # Merge with default config if not provided
        if "config" not in kwargs and source_type == SourceType.FILESYSTEM:
            kwargs["config"] = self.default_config

        return source_cls(**kwargs)

    def create_filesystem(
        self,
        root_path: str | Path,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        **kwargs: Any,
    ) -> FileSystemSource:
        """Create a filesystem document source.

        Args:
            root_path: Root directory to scan.
            include_patterns: Glob patterns to include.
            exclude_patterns: Glob patterns to exclude.
            **kwargs: Additional config options.

        Returns:
            FileSystemSource instance.
        """
        config = DocumentSourceConfig(
            include_patterns=include_patterns or ["*"],
            exclude_patterns=exclude_patterns or [],
            max_file_size_bytes=kwargs.pop("max_file_size_bytes", 10 * 1024 * 1024),
            follow_symlinks=kwargs.pop("follow_symlinks", False),
            encoding=kwargs.pop("encoding", "utf-8"),
        )
        return FileSystemSource(root_path, config)

    def create_git(
        self,
        repository_url: str,
        branch: str | None = None,
        **kwargs: Any,
    ) -> GitSource:
        """Create a Git document source (stub).

        Args:
            repository_url: Git repository URL.
            branch: Branch to load from.
            **kwargs: Additional config options.

        Returns:
            GitSource instance (stub - not implemented).
        """
        config = GitSourceConfig(
            repository_url=repository_url,
            branch=branch,
            **kwargs,
        )
        return GitSource(config)

    def create_confluence(
        self,
        base_url: str,
        space_keys: list[str] | None = None,
        **kwargs: Any,
    ) -> ConfluenceSource:
        """Create a Confluence document source (stub).

        Args:
            base_url: Confluence instance URL.
            space_keys: Space keys to scan.
            **kwargs: Additional config options.

        Returns:
            ConfluenceSource instance (stub - not implemented).
        """
        config = ConfluenceSourceConfig(
            base_url=base_url,
            space_keys=space_keys,
            **kwargs,
        )
        return ConfluenceSource(config)

    def create_s3(
        self,
        bucket_name: str,
        prefix: str = "",
        use_oci_sdk: bool = False,
        **kwargs: Any,
    ) -> S3Source:
        """Create an S3/OCI Object Storage document source (stub).

        Args:
            bucket_name: Bucket name.
            prefix: Object key prefix.
            use_oci_sdk: Use OCI SDK instead of S3 API.
            **kwargs: Additional config options.

        Returns:
            S3Source instance (stub - not implemented).
        """
        config = S3SourceConfig(
            bucket_name=bucket_name,
            prefix=prefix,
            use_oci_sdk=use_oci_sdk,
            **kwargs,
        )
        return S3Source(config)

    @classmethod
    def register(
        cls,
        source_type: SourceType,
        source_cls: Type[AbstractDocumentSource],
    ) -> None:
        """Register a custom document source.

        Args:
            source_type: Source type to register.
            source_cls: Source class to use.
        """
        cls._registry[source_type] = source_cls

    @classmethod
    def get_available_sources(cls) -> list[SourceType]:
        """Get list of available source types.

        Returns:
            List of registered source types.
        """
        return list(cls._registry.keys())

    @classmethod
    def is_implemented(cls, source_type: SourceType) -> bool:
        """Check if a source type is fully implemented (not a stub).

        Args:
            source_type: Source type to check.

        Returns:
            True if implemented, False if stub.
        """
        implemented_sources = {SourceType.FILESYSTEM}
        return source_type in implemented_sources


# Global factory instance for convenience
default_factory = DocumentSourceFactory()
