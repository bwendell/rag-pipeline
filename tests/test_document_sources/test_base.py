"""Tests for base document source utilities."""

import pytest
from rag_pipeline.document_sources.base import (
    DocumentSourceConfig,
    EXTENSION_TYPE_MAP,
    EXTENSION_LANGUAGE_MAP,
)
from rag_pipeline.core.types import DocumentType


class TestDocumentTypeDetection:
    @pytest.mark.parametrize(
        "extension,expected_type",
        [
            (".py", DocumentType.CODE),
            (".js", DocumentType.CODE),
            (".java", DocumentType.CODE),
            (".go", DocumentType.CODE),
            (".md", DocumentType.MARKDOWN),
            (".markdown", DocumentType.MARKDOWN),
            (".txt", DocumentType.TEXT),
            (".json", DocumentType.TEXT),
            (".unknown", DocumentType.UNKNOWN),
        ],
    )
    def test_extension_to_document_type(self, extension: str, expected_type: DocumentType):
        result = EXTENSION_TYPE_MAP.get(extension, DocumentType.UNKNOWN)
        assert result == expected_type

    @pytest.mark.parametrize(
        "extension,expected_language",
        [
            (".py", "python"),
            (".js", "javascript"),
            (".ts", "typescript"),
            (".java", "java"),
            (".go", "go"),
            (".rs", "rust"),
            (".unknown", None),
        ],
    )
    def test_extension_to_language(self, extension: str, expected_language: str | None):
        result = EXTENSION_LANGUAGE_MAP.get(extension)
        assert result == expected_language


class TestDocumentSourceConfig:
    def test_default_config(self):
        config = DocumentSourceConfig()
        assert config.include_patterns == ["*"]
        assert config.exclude_patterns == []
        assert config.max_file_size_bytes == 10 * 1024 * 1024
        assert config.follow_symlinks is False
        assert config.encoding == "utf-8"

    def test_custom_config(self):
        config = DocumentSourceConfig(
            include_patterns=["*.py", "*.md"],
            exclude_patterns=["tests/**"],
            max_file_size_bytes=5 * 1024 * 1024,
            follow_symlinks=True,
            encoding="latin-1",
        )
        assert config.include_patterns == ["*.py", "*.md"]
        assert config.exclude_patterns == ["tests/**"]
        assert config.max_file_size_bytes == 5 * 1024 * 1024
        assert config.follow_symlinks is True
        assert config.encoding == "latin-1"

    def test_to_dict(self):
        config = DocumentSourceConfig()
        result = config.to_dict()
        assert "include_patterns" in result
        assert "exclude_patterns" in result
        assert "max_file_size_bytes" in result
