"""Stub implementations for remote document sources.

These sources are not yet fully implemented but provide the interface
and configuration classes for future development.

All stub sources will raise NotImplementedSourceError when their
load() methods are called.
"""

from rag_pipeline.document_sources.stubs.git_source import (
    GitSource,
    GitSourceConfig,
    VBSConfig,
)
from rag_pipeline.document_sources.stubs.confluence_source import (
    ConfluenceSource,
    ConfluenceSourceConfig,
)
from rag_pipeline.document_sources.stubs.s3_source import (
    S3Source,
    S3SourceConfig,
)

__all__ = [
    # Git
    "GitSource",
    "GitSourceConfig",
    "VBSConfig",
    # Confluence
    "ConfluenceSource",
    "ConfluenceSourceConfig",
    # S3/OCI Object Storage
    "S3Source",
    "S3SourceConfig",
]
