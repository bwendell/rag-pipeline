"""Document source implementations for loading content.

This module provides abstractions and implementations for loading documents
from various sources including filesystem, Git repositories, Confluence,
and cloud object storage.

Example:
    >>> from rag_pipeline.document_sources import FileSystemSource, DocumentSourceConfig
    >>>
    >>> # Create a filesystem source
    >>> source = FileSystemSource("/path/to/code")
    >>> documents = list(source.load())
    >>>
    >>> # Use the factory for convenience
    >>> from rag_pipeline.document_sources import default_factory
    >>> source = default_factory.create_filesystem("/path/to/code")
"""

from rag_pipeline.document_sources.base import (
    AbstractDocumentSource,
    DocumentSourceConfig,
    EXTENSION_TYPE_MAP,
    EXTENSION_LANGUAGE_MAP,
)
from rag_pipeline.document_sources.filesystem_source import FileSystemSource
from rag_pipeline.document_sources.factory import (
    DocumentSourceFactory,
    default_factory,
)
from rag_pipeline.document_sources.stubs.git_source import GitSource, GitSourceConfig
from rag_pipeline.document_sources.stubs.confluence_source import (
    ConfluenceSource,
    ConfluenceSourceConfig,
)
from rag_pipeline.document_sources.stubs.s3_source import S3Source, S3SourceConfig

__all__ = [
    # Base classes
    "AbstractDocumentSource",
    "DocumentSourceConfig",
    # Mappings
    "EXTENSION_TYPE_MAP",
    "EXTENSION_LANGUAGE_MAP",
    # Implemented sources
    "FileSystemSource",
    # Factory
    "DocumentSourceFactory",
    "default_factory",
    # Stubs (not yet implemented)
    "GitSource",
    "GitSourceConfig",
    "ConfluenceSource",
    "ConfluenceSourceConfig",
    "S3Source",
    "S3SourceConfig",
]
