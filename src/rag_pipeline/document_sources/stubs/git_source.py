"""Git repository document source stub.

This module provides a stub implementation for loading documents
from Git repositories. Implementation is planned for future phases.

OCI Context:
    Oracle teams commonly use Visual Builder Studio (VBS) for Git hosting.
    VBS provides REST API access with OAuth authentication.

Future Implementation Notes:
    - Support for cloning repositories
    - Branch/commit/tag selection
    - Sparse checkout for large repos
    - Credential management (SSH keys, tokens)
    - VBS-specific authentication flow
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from rag_pipeline.core.exceptions import NotImplementedSourceError
from rag_pipeline.core.types import Document, SourceType
from rag_pipeline.document_sources.base import AbstractDocumentSource, DocumentSourceConfig


@dataclass
class GitSourceConfig:
    """Configuration for Git document source.

    NOTE: Uses COMPOSITION, not inheritance from DocumentSourceConfig.
    This avoids dataclass/non-dataclass inheritance issues.

    Attributes:
        base_config: Common document source configuration.
        repository_url: Git repository URL (HTTPS or SSH).
        branch: Branch to load from (default: main/master).
        commit: Specific commit hash (optional, overrides branch).
        tag: Specific tag (optional, overrides branch).
        depth: Clone depth for shallow clones (0 = full clone).
        sparse_paths: Paths for sparse checkout (None = full repo).
        credential_type: Authentication method.
        credential_path: Path to credential file (SSH key, token file).
    """

    repository_url: str = ""
    branch: str | None = None  # Auto-detect default branch
    commit: str | None = None
    tag: str | None = None
    depth: int = 1  # Shallow clone by default
    sparse_paths: list[str] | None = None
    credential_type: str = "none"  # "none", "ssh_key", "token", "oauth"
    credential_path: str | None = None
    # COMPOSITION: Include base config as a field
    base_config: DocumentSourceConfig = field(default_factory=DocumentSourceConfig)


@dataclass
class VBSConfig:
    """Configuration specific to Visual Builder Studio.

    Attributes:
        instance_url: VBS instance URL.
        project_id: VBS project identifier.
        oauth_client_id: OAuth client ID.
        oauth_client_secret: OAuth client secret (stored securely).
    """

    instance_url: str = ""
    project_id: str = ""
    oauth_client_id: str = ""
    oauth_client_secret: str = ""


class GitSource(AbstractDocumentSource):
    """Git repository document source (stub).

    Provides an interface for loading documents from Git repositories.
    Not yet implemented - raises NotImplementedSourceError.

    Future Features:
        - Repository cloning with configurable depth
        - Branch, commit, and tag selection
        - Sparse checkout for large repositories
        - SSH and token-based authentication
        - VBS (Visual Builder Studio) integration
        - Incremental updates via git diff

    Example (future usage):
        >>> source = GitSource(
        ...     config=GitSourceConfig(
        ...         repository_url="https://github.com/org/repo.git",
        ...         branch="main",
        ...         depth=1,
        ...     )
        ... )
        >>> async for doc in source.load():
        ...     process(doc)
    """

    def __init__(
        self,
        config: GitSourceConfig | None = None,
        vbs_config: VBSConfig | None = None,
    ) -> None:
        """Initialize Git source.

        Args:
            config: Git source configuration.
            vbs_config: VBS-specific configuration (optional).
        """
        self.git_config = config or GitSourceConfig()
        # COMPOSITION: Pass base_config to parent
        super().__init__(self.git_config.base_config, SourceType.GIT)
        self.vbs_config = vbs_config

    async def load(self) -> AsyncIterator[Document]:
        """Load all documents from the Git repository.

        Raises:
            NotImplementedSourceError: This source is not yet implemented.
        """
        raise NotImplementedSourceError(
            "GitSource is not yet implemented. "
            "Use FileSystemSource with a locally cloned repository, "
            "or implement GitSource for your use case. "
            "See docs/OCI_CONTEXT.md for VBS integration notes."
        )
        yield  # type: ignore

    async def load_changed(self, since: datetime) -> AsyncIterator[Document]:
        """Load changed documents using git diff."""
        raise NotImplementedSourceError("GitSource.load_changed not yet implemented")
        yield  # type: ignore

    async def load_by_path(self, path: str) -> Document | None:
        """Load a single file from the repository."""
        raise NotImplementedSourceError("GitSource.load_by_path not yet implemented")

    async def list_paths(self) -> list[str]:
        """List all files in the repository."""
        raise NotImplementedSourceError("GitSource.list_paths not yet implemented")

    async def get_changed_paths(self, since: datetime) -> list[str]:
        """Get paths changed since a given time using git log."""
        raise NotImplementedSourceError("GitSource.get_changed_paths not yet implemented")

    async def get_deleted_paths(self, known_paths: list[str]) -> list[str]:
        """Find deleted paths using git diff."""
        raise NotImplementedSourceError("GitSource.get_deleted_paths not yet implemented")

    def get_source_info(self) -> dict[str, Any]:
        """Get source information."""
        return {
            "type": "git",
            "status": "stub",
            "repository_url": self.git_config.repository_url,
            "branch": self.git_config.branch,
            "message": "GitSource is a stub. Implementation planned for future phase.",
        }

    async def health_check(self) -> bool:
        """Check repository accessibility."""
        # Stub always returns False
        return False


# Implementation notes for future development:
#
# 1. Use gitpython or dulwich for Git operations
# 2. Clone to a temporary directory or configurable workspace
# 3. For VBS, use REST API: https://docs.oracle.com/en/cloud/paas/visual-builder/
# 4. Authentication options:
#    - SSH: Use paramiko or subprocess with ssh-agent
#    - Token: Pass via HTTPS URL or header
#    - OAuth (VBS): Follow OAuth 2.0 flow
# 5. For incremental updates:
#    - Store last indexed commit hash
#    - Use git diff --name-status to find changes
# 6. For large repos:
#    - Use shallow clones (--depth)
#    - Use sparse checkout for specific paths
#    - Consider git archive for read-only access
