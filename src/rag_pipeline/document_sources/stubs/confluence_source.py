"""Confluence document source stub.

This module provides a stub implementation for loading documents
from Atlassian Confluence. Implementation is planned for future phases.

OCI Context:
    Oracle teams commonly use Confluence for documentation and runbooks.
    Access typically requires API token authentication.

Future Implementation Notes:
    - REST API v2 for modern Confluence
    - Page content in storage format (XHTML) or view format
    - Space-based filtering
    - Label-based filtering
    - Attachment handling
    - CQL (Confluence Query Language) support
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from rag_pipeline.document_sources.base import AbstractDocumentSource, DocumentSourceConfig
from rag_pipeline.core.types import Document, SourceType
from rag_pipeline.core.exceptions import NotImplementedSourceError


@dataclass
class ConfluenceSourceConfig:
    """Configuration for Confluence document source.

    NOTE: Uses COMPOSITION, not inheritance from DocumentSourceConfig.
    This avoids dataclass/non-dataclass inheritance issues.

    Attributes:
        base_config: Common document source configuration.
        base_url: Confluence instance URL.
        space_keys: List of space keys to scan (empty = all accessible).
        page_ids: Specific page IDs to load (optional).
        labels: Filter pages by labels (optional).
        include_attachments: Whether to load attachments.
        include_archived: Whether to include archived pages.
        cql_query: Custom CQL query (overrides other filters).
        auth_type: Authentication method ("token", "oauth", "basic").
        username: Username for basic auth.
        api_token: API token or password.
    """

    base_url: str = ""
    space_keys: list[str] | None = None
    page_ids: list[str] | None = None
    labels: list[str] | None = None
    include_attachments: bool = False
    include_archived: bool = False
    cql_query: str | None = None
    auth_type: str = "token"
    username: str = ""
    api_token: str = ""
    # COMPOSITION: Include base config as a field
    base_config: DocumentSourceConfig = field(default_factory=DocumentSourceConfig)


class ConfluenceSource(AbstractDocumentSource):
    """Confluence document source (stub).

    Provides an interface for loading documents from Confluence wikis.
    Not yet implemented - raises NotImplementedSourceError.

    Future Features:
        - Page and blog post loading
        - Space-based and label-based filtering
        - CQL query support
        - Attachment extraction
        - HTML to Markdown conversion
        - Incremental updates via page history

    Example (future usage):
        >>> source = ConfluenceSource(
        ...     config=ConfluenceSourceConfig(
        ...         base_url="https://company.atlassian.net/wiki",
        ...         space_keys=["ENG", "OPS"],
        ...         labels=["runbook"],
        ...     )
        ... )
        >>> async for doc in source.load():
        ...     process(doc)
    """

    def __init__(
        self,
        config: ConfluenceSourceConfig | None = None,
    ) -> None:
        """Initialize Confluence source.

        Args:
            config: Confluence source configuration.
        """
        self.confluence_config = config or ConfluenceSourceConfig()
        # COMPOSITION: Pass base_config to parent
        super().__init__(self.confluence_config.base_config, SourceType.CONFLUENCE)

    async def load(self) -> AsyncIterator[Document]:
        """Load all documents from Confluence.

        Raises:
            NotImplementedSourceError: This source is not yet implemented.
        """
        raise NotImplementedSourceError(
            "ConfluenceSource is not yet implemented. "
            "Export pages as Markdown and use FileSystemSource, "
            "or implement ConfluenceSource for your use case. "
            "Recommended library: atlassian-python-api"
        )
        yield  # type: ignore

    async def load_changed(self, since: datetime) -> AsyncIterator[Document]:
        """Load pages modified since a given time."""
        raise NotImplementedSourceError("ConfluenceSource.load_changed not yet implemented")
        yield  # type: ignore

    async def load_by_path(self, path: str) -> Document | None:
        """Load a single page by ID or title."""
        raise NotImplementedSourceError("ConfluenceSource.load_by_path not yet implemented")

    async def list_paths(self) -> list[str]:
        """List all page IDs/titles in configured spaces."""
        raise NotImplementedSourceError("ConfluenceSource.list_paths not yet implemented")

    async def get_changed_paths(self, since: datetime) -> list[str]:
        """Get IDs of pages changed since a given time."""
        raise NotImplementedSourceError("ConfluenceSource.get_changed_paths not yet implemented")

    async def get_deleted_paths(self, known_paths: list[str]) -> list[str]:
        """Find pages that have been deleted."""
        raise NotImplementedSourceError("ConfluenceSource.get_deleted_paths not yet implemented")

    def get_source_info(self) -> dict[str, Any]:
        """Get source information."""
        return {
            "type": "confluence",
            "status": "stub",
            "base_url": self.confluence_config.base_url,
            "space_keys": self.confluence_config.space_keys,
            "message": "ConfluenceSource is a stub. Implementation planned for future phase.",
        }

    async def health_check(self) -> bool:
        """Check Confluence accessibility."""
        return False


# Implementation notes for future development:
#
# 1. Use atlassian-python-api: pip install atlassian-python-api
# 2. API endpoint patterns:
#    - Pages: GET /wiki/rest/api/content?type=page&spaceKey=X
#    - CQL: GET /wiki/rest/api/search?cql=...
# 3. Content formats:
#    - storage: Raw XHTML storage format
#    - view: Rendered HTML
#    - Use markdownify or html2text for conversion
# 4. For runbooks, filter by label: label = "runbook"
# 5. Rate limiting: Respect Confluence rate limits
# 6. Pagination: Handle via _links.next in response
