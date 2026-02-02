"""Base document source with shared utilities.

This module provides the AbstractDocumentSource base class that all
document source implementations should extend. It handles:
- Pattern matching for file filtering
- Document type detection
- Metadata extraction
- Configuration validation
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pathspec

from rag_pipeline.core.exceptions import DocumentSourceError
from rag_pipeline.core.types import (
    Document,
    DocumentType,
    Metadata,
    SourceType,
)

# File extension to DocumentType mapping
EXTENSION_TYPE_MAP: dict[str, DocumentType] = {
    # Code files
    ".py": DocumentType.CODE,
    ".pyw": DocumentType.CODE,
    ".js": DocumentType.CODE,
    ".jsx": DocumentType.CODE,
    ".ts": DocumentType.CODE,
    ".tsx": DocumentType.CODE,
    ".java": DocumentType.CODE,
    ".go": DocumentType.CODE,
    ".rs": DocumentType.CODE,
    ".c": DocumentType.CODE,
    ".cpp": DocumentType.CODE,
    ".cc": DocumentType.CODE,
    ".h": DocumentType.CODE,
    ".hpp": DocumentType.CODE,
    ".cs": DocumentType.CODE,
    ".rb": DocumentType.CODE,
    ".php": DocumentType.CODE,
    ".swift": DocumentType.CODE,
    ".kt": DocumentType.CODE,
    ".scala": DocumentType.CODE,
    ".sh": DocumentType.CODE,
    ".bash": DocumentType.CODE,
    ".zsh": DocumentType.CODE,
    ".ps1": DocumentType.CODE,
    ".sql": DocumentType.CODE,
    # Markdown/Documentation
    ".md": DocumentType.MARKDOWN,
    ".markdown": DocumentType.MARKDOWN,
    ".mdx": DocumentType.MARKDOWN,
    ".rst": DocumentType.MARKDOWN,
    # Text
    ".txt": DocumentType.TEXT,
    ".log": DocumentType.TEXT,
    ".csv": DocumentType.TEXT,
    ".json": DocumentType.TEXT,
    ".yaml": DocumentType.TEXT,
    ".yml": DocumentType.TEXT,
    ".xml": DocumentType.TEXT,
    ".toml": DocumentType.TEXT,
    ".ini": DocumentType.TEXT,
    ".cfg": DocumentType.TEXT,
    ".conf": DocumentType.TEXT,
}

# Extension to language mapping for code files
EXTENSION_LANGUAGE_MAP: dict[str, str] = {
    ".py": "python",
    ".pyw": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "zsh",
    ".ps1": "powershell",
    ".sql": "sql",
}


@dataclass
class DocumentSourceConfig:
    """Configuration for document sources.

    Attributes:
        include_patterns: Glob patterns for files to include.
        exclude_patterns: Glob patterns for files to exclude.
        max_file_size_bytes: Maximum file size to process.
        follow_symlinks: Whether to follow symbolic links.
        encoding: Default text encoding.
    """

    include_patterns: list[str] = field(default_factory=lambda: ["*"])
    exclude_patterns: list[str] = field(default_factory=list)
    max_file_size_bytes: int = 10 * 1024 * 1024  # 10 MB
    follow_symlinks: bool = False
    encoding: str = "utf-8"

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "include_patterns": self.include_patterns,
            "exclude_patterns": self.exclude_patterns,
            "max_file_size_bytes": self.max_file_size_bytes,
            "follow_symlinks": self.follow_symlinks,
            "encoding": self.encoding,
        }


class AbstractDocumentSource(ABC):
    """Abstract base class for document source implementations.

    Provides common functionality shared across all sources:
    - Pattern matching for file filtering
    - Document type detection
    - Metadata extraction
    - Configuration management

    Subclasses must implement all methods from DocumentSource protocol.
    """

    # Default patterns to always exclude
    # SECURITY: These patterns protect against ingesting sensitive files
    DEFAULT_EXCLUDE_PATTERNS: list[str] = [
        # Version control
        ".git/**",
        ".git",
        ".svn/**",
        ".hg/**",
        # Dependencies and caches
        "node_modules/**",
        "__pycache__/**",
        "*.pyc",
        ".venv/**",
        "venv/**",
        "env/**",
        # SECURITY: Environment and secrets files - MUST exclude
        ".env",  # Root .env file
        ".env.*",  # .env.local, .env.production, etc.
        "**/.env",  # Nested .env files
        "**/.env.*",  # Nested variants
        "*.pem",  # Private keys
        "*.key",  # Private keys
        "*credentials*",  # Credential files
        "*secret*",  # Secret files
        # IDE and editor
        ".idea/**",
        ".vscode/**",
        # Lock files
        "*.lock",
        "package-lock.json",
        "yarn.lock",
        # OS files
        ".DS_Store",
        "Thumbs.db",
        # Build artifacts
        "*.min.js",
        "*.min.css",
        "*.map",
        "dist/**",
        "build/**",
        "target/**",
        "*.egg-info/**",
    ]

    def __init__(
        self,
        config: DocumentSourceConfig | None = None,
        source_type: SourceType = SourceType.UNKNOWN,
    ) -> None:
        """Initialize the document source.

        Args:
            config: Source configuration.
            source_type: Type of this source.
        """
        self.config = config or DocumentSourceConfig()
        self.source_type = source_type
        self._validate_config()
        self._precompile_patterns()

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.config.max_file_size_bytes <= 0:
            raise DocumentSourceError("max_file_size_bytes must be positive")

    def _precompile_patterns(self) -> None:
        """Precompile PathSpec patterns for performance.

        Called once in __init__ to avoid recompiling on every _should_include call.
        """
        all_excludes = self.DEFAULT_EXCLUDE_PATTERNS + self.config.exclude_patterns
        self._exclude_spec = pathspec.PathSpec.from_lines("gitwildmatch", all_excludes)
        self._include_spec = pathspec.PathSpec.from_lines(
            "gitwildmatch", self.config.include_patterns
        )

    def _detect_document_type(self, path: str | Path) -> DocumentType:
        """Detect document type from file extension.

        Args:
            path: File path.

        Returns:
            Detected DocumentType.
        """
        ext = Path(path).suffix.lower()
        return EXTENSION_TYPE_MAP.get(ext, DocumentType.UNKNOWN)

    def _detect_language(self, path: str | Path) -> str | None:
        """Detect programming language from file extension.

        Args:
            path: File path.

        Returns:
            Language string or None.
        """
        ext = Path(path).suffix.lower()
        return EXTENSION_LANGUAGE_MAP.get(ext)

    def _create_metadata(
        self,
        path: str,
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
    ) -> Metadata:
        """Create metadata for a document.

        Args:
            path: Source path of the document.
            created_at: Creation timestamp.
            updated_at: Last modification timestamp.

        Returns:
            Metadata instance.
        """
        now = datetime.now()
        ext = Path(path).suffix.lower()

        return Metadata(
            source_path=path,
            source_type=self.source_type,
            language=self._detect_language(path),
            file_extension=ext if ext else None,
            created_at=created_at or now,
            updated_at=updated_at or now,
        )

    def _create_document(
        self,
        content: str,
        path: str,
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
    ) -> Document:
        """Create a Document from content and path.

        Args:
            content: Document text content.
            path: Source path.
            created_at: Creation timestamp.
            updated_at: Last modification timestamp.

        Returns:
            Document instance.
        """
        return Document(
            content=content,
            doc_type=self._detect_document_type(path),
            metadata=self._create_metadata(path, created_at, updated_at),
        )

    def _should_include(self, path: str) -> bool:
        """Check if a FILE path should be included based on patterns.

        NOTE: For directories, use _should_exclude_directory instead.

        Args:
            path: File path to check.

        Returns:
            True if path should be included.
        """
        # Check exclude patterns first (including defaults)
        if self._exclude_spec.match_file(path):
            return False

        # Check include patterns
        return self._include_spec.match_file(path)

    def _should_exclude_directory(self, dir_path: str) -> bool:
        """Check if a DIRECTORY should be excluded from traversal.

        CORRECTNESS: Only applies exclude patterns to directories.
        Include patterns (like "*.py") should NOT prevent directory recursion.

        Args:
            dir_path: Directory path to check (without trailing slash).

        Returns:
            True if directory should be excluded (skipped).
        """
        # Check both with and without trailing slash for gitignore compatibility
        return self._exclude_spec.match_file(dir_path) or self._exclude_spec.match_file(
            dir_path + "/"
        )

    def _is_text_file(self, path: Path) -> bool:
        """Check if a file is likely a text file.

        Args:
            path: Path to check.

        Returns:
            True if file appears to be text.
        """
        # Check by extension first
        ext = path.suffix.lower()
        if ext in EXTENSION_TYPE_MAP:
            return True

        # Binary file extensions
        binary_extensions = {
            ".exe",
            ".dll",
            ".so",
            ".dylib",
            ".bin",
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".bmp",
            ".ico",
            ".webp",
            ".mp3",
            ".mp4",
            ".wav",
            ".avi",
            ".mov",
            ".mkv",
            ".pdf",
            ".doc",
            ".docx",
            ".xls",
            ".xlsx",
            ".ppt",
            ".pptx",
            ".zip",
            ".tar",
            ".gz",
            ".7z",
            ".rar",
            ".woff",
            ".woff2",
            ".ttf",
            ".eot",
            ".otf",
            ".class",
            ".jar",
            ".war",
            ".ear",
            ".o",
            ".a",
            ".lib",
            ".pyc",
            ".pyo",
            ".pyd",
            ".db",
            ".sqlite",
            ".sqlite3",
        }
        if ext in binary_extensions:
            return False

        # Unknown extension - could be text
        return True

    def _is_binary_content(self, content: bytes, sample_size: int = 8192) -> bool:
        """Check if content appears to be binary by examining bytes.

        SECURITY: This prevents binary garbage from being ingested into embeddings.

        Args:
            content: Raw bytes to check.
            sample_size: Number of bytes to sample from start.

        Returns:
            True if content appears to be binary.
        """
        if not content:
            return False

        sample = content[:sample_size]

        # NUL bytes strongly indicate binary content
        if b"\x00" in sample:
            return True

        # High ratio of non-printable bytes indicates binary
        # Allow tabs (9), newlines (10), carriage returns (13)
        non_text_bytes = sum(1 for b in sample if b < 32 and b not in (9, 10, 13))
        ratio = non_text_bytes / len(sample) if sample else 0

        return ratio > 0.3  # More than 30% non-text bytes = binary

    @abstractmethod
    async def load(self) -> AsyncIterator[Document]:
        """Load all documents from the source."""
        ...

    @abstractmethod
    async def load_changed(self, since: datetime) -> AsyncIterator[Document]:
        """Load documents changed since a given time."""
        ...

    @abstractmethod
    async def load_by_path(self, path: str) -> Document | None:
        """Load a single document by path."""
        ...

    @abstractmethod
    async def list_paths(self) -> list[str]:
        """List all available document paths."""
        ...

    @abstractmethod
    async def get_changed_paths(self, since: datetime) -> list[str]:
        """Get paths of documents changed since a given time."""
        ...

    @abstractmethod
    async def get_deleted_paths(self, known_paths: list[str]) -> list[str]:
        """Find paths that no longer exist."""
        ...

    @abstractmethod
    def get_source_info(self) -> dict[str, Any]:
        """Get information about the document source."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the source is accessible."""
        ...
