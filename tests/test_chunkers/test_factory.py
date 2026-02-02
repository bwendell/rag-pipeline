"""Tests for ChunkerFactory.

This module comprehensively tests the chunker factory functions including:
- create_chunker() with DocumentType and string type names
- get_chunker_for_document() with type auto-detection
- register_chunker() for custom chunker registration
- get_supported_types() and get_chunker_class() utility functions
- Extension-based type inference for UNKNOWN documents
- Config passing to chunkers
"""

from __future__ import annotations

import pytest

from rag_pipeline.chunkers import (
    CHUNKER_REGISTRY,
    AbstractChunker,
    CodeChunker,
    GenericChunker,
    MarkdownChunker,
    create_chunker,
    get_chunker_class,
    get_chunker_for_document,
    get_supported_types,
    register_chunker,
)
from rag_pipeline.core.base_chunker import ChunkingConfig
from rag_pipeline.core.exceptions import ChunkingError
from rag_pipeline.core.types import Document, DocumentType, Metadata


class CustomChunker(AbstractChunker):
    """Custom chunker for testing registration."""

    def chunk(self, document: Document) -> list:  # noqa: ARG002
        return []

    def get_supported_types(self) -> list[str]:
        return ["custom"]


class TestCreateChunkerByDocumentType:
    """Tests for create_chunker() with DocumentType enum values."""

    def test_create_code_chunker(self) -> None:
        chunker = create_chunker(DocumentType.CODE)
        assert isinstance(chunker, CodeChunker)
        assert isinstance(chunker, AbstractChunker)

    def test_create_markdown_chunker(self) -> None:
        chunker = create_chunker(DocumentType.MARKDOWN)
        assert isinstance(chunker, MarkdownChunker)
        assert isinstance(chunker, AbstractChunker)

    def test_create_runbook_chunker(self) -> None:
        chunker = create_chunker(DocumentType.RUNBOOK)
        assert isinstance(chunker, MarkdownChunker)

    def test_create_text_chunker(self) -> None:
        chunker = create_chunker(DocumentType.TEXT)
        assert isinstance(chunker, GenericChunker)
        assert isinstance(chunker, AbstractChunker)

    def test_create_unknown_type_chunker(self) -> None:
        chunker = create_chunker(DocumentType.UNKNOWN)
        assert isinstance(chunker, GenericChunker)

    def test_all_registered_types_have_chunkers(self) -> None:
        for doc_type in DocumentType:
            chunker = create_chunker(doc_type)
            assert chunker is not None
            assert isinstance(chunker, AbstractChunker)


class TestCreateChunkerByString:
    """Tests for create_chunker() with string type names."""

    def test_create_chunker_with_valid_string_code(self) -> None:
        chunker = create_chunker("code")
        assert isinstance(chunker, CodeChunker)

    def test_create_chunker_with_valid_string_markdown(self) -> None:
        chunker = create_chunker("markdown")
        assert isinstance(chunker, MarkdownChunker)

    def test_create_chunker_with_valid_string_text(self) -> None:
        chunker = create_chunker("text")
        assert isinstance(chunker, GenericChunker)

    def test_create_chunker_with_valid_string_runbook(self) -> None:
        chunker = create_chunker("runbook")
        assert isinstance(chunker, MarkdownChunker)

    def test_create_chunker_with_invalid_string_falls_back_to_generic(self) -> None:
        chunker = create_chunker("invalid_type_that_does_not_exist")
        assert isinstance(chunker, GenericChunker)

    def test_create_chunker_with_empty_string_falls_back_to_generic(self) -> None:
        chunker = create_chunker("")
        assert isinstance(chunker, GenericChunker)

    def test_create_chunker_with_mixed_case_string_falls_back(self) -> None:
        chunker = create_chunker("CODE")
        assert isinstance(chunker, GenericChunker)

    def test_create_chunker_string_only_lowercase_works(self) -> None:
        chunker_code = create_chunker("code")
        assert isinstance(chunker_code, CodeChunker)


class TestCreateChunkerWithConfig:
    """Tests for create_chunker() with custom ChunkingConfig."""

    def test_create_chunker_passes_config_to_chunker(self) -> None:
        config = ChunkingConfig(chunk_size=500, chunk_overlap=50)
        chunker = create_chunker(DocumentType.CODE, config)
        assert chunker.config == config
        assert chunker.config.chunk_size == 500
        assert chunker.config.chunk_overlap == 50

    def test_create_chunker_with_none_config_uses_defaults(self) -> None:
        chunker = create_chunker(DocumentType.MARKDOWN, None)
        assert chunker.config is not None
        assert chunker.config.chunk_size == 1000
        assert chunker.config.chunk_overlap == 200

    def test_create_chunker_custom_config_applied_to_all_types(self) -> None:
        config = ChunkingConfig(chunk_size=2000, chunk_overlap=100)

        code_chunker = create_chunker(DocumentType.CODE, config)
        md_chunker = create_chunker(DocumentType.MARKDOWN, config)
        text_chunker = create_chunker(DocumentType.TEXT, config)

        assert code_chunker.config.chunk_size == 2000
        assert md_chunker.config.chunk_size == 2000
        assert text_chunker.config.chunk_size == 2000

    def test_different_configs_create_independent_chunkers(self) -> None:
        config1 = ChunkingConfig(chunk_size=500)
        config2 = ChunkingConfig(chunk_size=1500)

        chunker1 = create_chunker(DocumentType.CODE, config1)
        chunker2 = create_chunker(DocumentType.CODE, config2)

        assert chunker1.config.chunk_size == 500
        assert chunker2.config.chunk_size == 1500


class TestGetChunkerForDocument:
    """Tests for get_chunker_for_document() with type detection."""

    def test_get_chunker_for_code_document(self, code_document: Document) -> None:
        chunker = get_chunker_for_document(code_document)
        assert isinstance(chunker, CodeChunker)

    def test_get_chunker_for_markdown_document(self, markdown_document: Document) -> None:
        chunker = get_chunker_for_document(markdown_document)
        assert isinstance(chunker, MarkdownChunker)

    def test_get_chunker_for_text_document(self, text_document: Document) -> None:
        chunker = get_chunker_for_document(text_document)
        assert isinstance(chunker, GenericChunker)

    def test_get_chunker_for_runbook_document(self, runbook_document: Document) -> None:
        chunker = get_chunker_for_document(runbook_document)
        assert isinstance(chunker, MarkdownChunker)

    def test_get_chunker_for_unknown_type_without_extension(self) -> None:
        doc = Document(
            content="Some content",
            doc_type=DocumentType.UNKNOWN,
            metadata=Metadata(source_path="/test/file"),
        )
        chunker = get_chunker_for_document(doc)
        assert isinstance(chunker, GenericChunker)

    def test_get_chunker_infers_type_from_extension_py(self) -> None:
        doc = Document(
            content="def foo(): pass",
            doc_type=DocumentType.UNKNOWN,
            metadata=Metadata(
                source_path="/test/file.py",
                file_extension=".py",
            ),
        )
        chunker = get_chunker_for_document(doc)
        assert isinstance(chunker, CodeChunker)

    def test_get_chunker_infers_type_from_extension_js(self) -> None:
        doc = Document(
            content="function foo() {}",
            doc_type=DocumentType.UNKNOWN,
            metadata=Metadata(
                source_path="/test/file.js",
                file_extension=".js",
            ),
        )
        chunker = get_chunker_for_document(doc)
        assert isinstance(chunker, CodeChunker)

    def test_get_chunker_infers_type_from_extension_md(self) -> None:
        doc = Document(
            content="# Header",
            doc_type=DocumentType.UNKNOWN,
            metadata=Metadata(
                source_path="/test/file.md",
                file_extension=".md",
            ),
        )
        chunker = get_chunker_for_document(doc)
        assert isinstance(chunker, MarkdownChunker)

    def test_get_chunker_infers_type_from_extension_txt(self) -> None:
        doc = Document(
            content="Plain text",
            doc_type=DocumentType.UNKNOWN,
            metadata=Metadata(
                source_path="/test/file.txt",
                file_extension=".txt",
            ),
        )
        chunker = get_chunker_for_document(doc)
        assert isinstance(chunker, GenericChunker)

    def test_get_chunker_handles_extension_without_dot(self) -> None:
        doc = Document(
            content="def foo(): pass",
            doc_type=DocumentType.UNKNOWN,
            metadata=Metadata(
                source_path="/test/file.py",
                file_extension="py",
            ),
        )
        chunker = get_chunker_for_document(doc)
        assert isinstance(chunker, CodeChunker)

    def test_get_chunker_handles_uppercase_extension(self) -> None:
        doc = Document(
            content="def foo(): pass",
            doc_type=DocumentType.UNKNOWN,
            metadata=Metadata(
                source_path="/test/file.PY",
                file_extension=".PY",
            ),
        )
        chunker = get_chunker_for_document(doc)
        assert isinstance(chunker, CodeChunker)

    def test_get_chunker_unknown_extension_falls_back_to_generic(self) -> None:
        doc = Document(
            content="Some content",
            doc_type=DocumentType.UNKNOWN,
            metadata=Metadata(
                source_path="/test/file.xyz",
                file_extension=".xyz",
            ),
        )
        chunker = get_chunker_for_document(doc)
        assert isinstance(chunker, GenericChunker)

    def test_get_chunker_for_document_with_config(self, code_document: Document) -> None:
        config = ChunkingConfig(chunk_size=750, chunk_overlap=75)
        chunker = get_chunker_for_document(code_document, config)
        assert chunker.config.chunk_size == 750
        assert chunker.config.chunk_overlap == 75

    def test_explicit_type_takes_precedence_over_extension(self) -> None:
        doc = Document(
            content="# This is markdown content",
            doc_type=DocumentType.MARKDOWN,
            metadata=Metadata(
                source_path="/test/file.py",
                file_extension=".py",
            ),
        )
        chunker = get_chunker_for_document(doc)
        assert isinstance(chunker, MarkdownChunker)


class TestRegisterChunker:
    """Tests for register_chunker() custom registration."""

    def test_register_custom_chunker(self) -> None:
        original_chunker = CHUNKER_REGISTRY.get(DocumentType.UNKNOWN)
        try:
            register_chunker(DocumentType.UNKNOWN, CustomChunker)
            chunker = create_chunker(DocumentType.UNKNOWN)
            assert isinstance(chunker, CustomChunker)
        finally:
            if original_chunker:
                CHUNKER_REGISTRY[DocumentType.UNKNOWN] = original_chunker

    def test_register_overwrites_existing_chunker(self) -> None:
        original_chunker = CHUNKER_REGISTRY[DocumentType.TEXT]
        try:
            register_chunker(DocumentType.TEXT, CustomChunker)
            chunker = create_chunker(DocumentType.TEXT)
            assert isinstance(chunker, CustomChunker)
        finally:
            CHUNKER_REGISTRY[DocumentType.TEXT] = original_chunker

    def test_register_invalid_chunker_raises_type_error(self) -> None:
        class NotAChunker:
            pass

        with pytest.raises(TypeError) as exc_info:
            register_chunker(DocumentType.CODE, NotAChunker)  # type: ignore

        assert "must be a subclass of AbstractChunker" in str(exc_info.value)

    def test_register_non_class_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            register_chunker(DocumentType.CODE, "not a class")  # type: ignore

    def test_registered_chunker_works_with_factory_functions(self) -> None:
        original_chunker = CHUNKER_REGISTRY.get(DocumentType.UNKNOWN)
        try:
            register_chunker(DocumentType.UNKNOWN, CustomChunker)

            doc = Document(
                content="test",
                doc_type=DocumentType.UNKNOWN,
            )
            chunker = get_chunker_for_document(doc)
            assert isinstance(chunker, CustomChunker)
        finally:
            if original_chunker:
                CHUNKER_REGISTRY[DocumentType.UNKNOWN] = original_chunker

    def test_registered_chunker_receives_config(self) -> None:
        original_chunker = CHUNKER_REGISTRY.get(DocumentType.UNKNOWN)
        try:
            register_chunker(DocumentType.UNKNOWN, CustomChunker)
            config = ChunkingConfig(chunk_size=999)
            chunker = create_chunker(DocumentType.UNKNOWN, config)
            assert chunker.config.chunk_size == 999
        finally:
            if original_chunker:
                CHUNKER_REGISTRY[DocumentType.UNKNOWN] = original_chunker


class TestGetSupportedTypes:
    """Tests for get_supported_types() registry inspection."""

    def test_get_supported_types_returns_list(self) -> None:
        types = get_supported_types()
        assert isinstance(types, list)

    def test_get_supported_types_contains_all_defaults(self) -> None:
        types = get_supported_types()

        assert DocumentType.CODE in types
        assert DocumentType.MARKDOWN in types
        assert DocumentType.RUNBOOK in types
        assert DocumentType.TEXT in types
        assert DocumentType.UNKNOWN in types

    def test_get_supported_types_minimum_count(self) -> None:
        types = get_supported_types()
        assert len(types) >= 5

    def test_get_supported_types_contains_only_document_types(self) -> None:
        types = get_supported_types()
        for doc_type in types:
            assert isinstance(doc_type, DocumentType)

    def test_get_supported_types_after_registration(self) -> None:
        original_chunker = CHUNKER_REGISTRY.get(DocumentType.UNKNOWN)
        try:
            initial_types = get_supported_types()
            initial_count = len(initial_types)

            register_chunker(DocumentType.UNKNOWN, CustomChunker)

            updated_types = get_supported_types()
            assert len(updated_types) == initial_count
            assert DocumentType.UNKNOWN in updated_types
        finally:
            if original_chunker:
                CHUNKER_REGISTRY[DocumentType.UNKNOWN] = original_chunker


class TestGetChunkerClass:
    """Tests for get_chunker_class() class retrieval."""

    def test_get_chunker_class_code(self) -> None:
        chunker_class = get_chunker_class(DocumentType.CODE)
        assert chunker_class is CodeChunker

    def test_get_chunker_class_markdown(self) -> None:
        chunker_class = get_chunker_class(DocumentType.MARKDOWN)
        assert chunker_class is MarkdownChunker

    def test_get_chunker_class_runbook(self) -> None:
        chunker_class = get_chunker_class(DocumentType.RUNBOOK)
        assert chunker_class is MarkdownChunker

    def test_get_chunker_class_text(self) -> None:
        chunker_class = get_chunker_class(DocumentType.TEXT)
        assert chunker_class is GenericChunker

    def test_get_chunker_class_unknown(self) -> None:
        chunker_class = get_chunker_class(DocumentType.UNKNOWN)
        assert chunker_class is GenericChunker

    def test_get_chunker_class_returns_class_not_instance(self) -> None:
        chunker_class = get_chunker_class(DocumentType.CODE)
        assert isinstance(chunker_class, type)
        assert issubclass(chunker_class, AbstractChunker)

    def test_get_chunker_class_can_be_instantiated(self) -> None:
        chunker_class = get_chunker_class(DocumentType.CODE)
        instance = chunker_class()
        assert isinstance(instance, AbstractChunker)

    def test_get_chunker_class_unregistered_type_returns_generic(self) -> None:
        fake_type = DocumentType.UNKNOWN
        chunker_class = get_chunker_class(fake_type)
        assert chunker_class is GenericChunker


class TestExtensionTypeMapping:
    """Tests for file extension to DocumentType mapping."""

    def test_extension_mapping_python_files(self) -> None:
        for ext in [".py"]:
            doc = Document(
                content="pass",
                doc_type=DocumentType.UNKNOWN,
                metadata=Metadata(file_extension=ext),
            )
            chunker = get_chunker_for_document(doc)
            assert isinstance(chunker, CodeChunker)

    def test_extension_mapping_javascript_files(self) -> None:
        for ext in [".js", ".jsx", ".ts", ".tsx"]:
            doc = Document(
                content="const x = 1;",
                doc_type=DocumentType.UNKNOWN,
                metadata=Metadata(file_extension=ext),
            )
            chunker = get_chunker_for_document(doc)
            assert isinstance(chunker, CodeChunker), f"Failed for {ext}"

    def test_extension_mapping_markdown_files(self) -> None:
        for ext in [".md", ".markdown", ".mdx"]:
            doc = Document(
                content="#",
                doc_type=DocumentType.UNKNOWN,
                metadata=Metadata(file_extension=ext),
            )
            chunker = get_chunker_for_document(doc)
            assert isinstance(chunker, MarkdownChunker), f"Failed for {ext}"

    def test_extension_mapping_text_files(self) -> None:
        for ext in [".txt", ".text", ".log", ".csv"]:
            doc = Document(
                content="content",
                doc_type=DocumentType.UNKNOWN,
                metadata=Metadata(file_extension=ext),
            )
            chunker = get_chunker_for_document(doc)
            assert isinstance(chunker, GenericChunker), f"Failed for {ext}"

    def test_extension_mapping_many_code_languages(self) -> None:
        code_extensions = [
            ".py",
            ".js",
            ".java",
            ".go",
            ".rs",
            ".c",
            ".cpp",
            ".h",
            ".cs",
            ".rb",
            ".php",
        ]
        for ext in code_extensions:
            doc = Document(
                content="x",
                doc_type=DocumentType.UNKNOWN,
                metadata=Metadata(file_extension=ext),
            )
            chunker = get_chunker_for_document(doc)
            assert isinstance(chunker, CodeChunker), f"Extension {ext} should map to CodeChunker"


class TestConfigurationValidation:
    """Tests for configuration validation in factory."""

    def test_invalid_config_chunk_size_zero(self) -> None:
        config = ChunkingConfig(chunk_size=0)
        with pytest.raises(ChunkingError):
            create_chunker(DocumentType.CODE, config)

    def test_invalid_config_negative_chunk_size(self) -> None:
        config = ChunkingConfig(chunk_size=-100)
        with pytest.raises(ChunkingError):
            create_chunker(DocumentType.CODE, config)

    def test_invalid_config_overlap_greater_than_size(self) -> None:
        config = ChunkingConfig(chunk_size=100, chunk_overlap=150)
        with pytest.raises(ChunkingError):
            create_chunker(DocumentType.CODE, config)

    def test_valid_config_edge_case_overlap_equals_size_minus_one(self) -> None:
        config = ChunkingConfig(chunk_size=100, chunk_overlap=99)
        chunker = create_chunker(DocumentType.CODE, config)
        assert chunker.config.chunk_overlap == 99


class TestIntegration:
    """Integration tests combining multiple factory features."""

    def test_create_and_chunk_workflow(self, code_document: Document) -> None:
        config = ChunkingConfig(chunk_size=500, chunk_overlap=50)
        chunker = get_chunker_for_document(code_document, config)

        assert isinstance(chunker, CodeChunker)
        assert chunker.config.chunk_size == 500

    def test_factory_functions_consistency(self) -> None:
        by_type = create_chunker(DocumentType.CODE)
        by_class = get_chunker_class(DocumentType.CODE)()

        assert type(by_type) is type(by_class)

    def test_multiple_calls_create_independent_instances(self) -> None:
        chunker1 = create_chunker(DocumentType.CODE)
        chunker2 = create_chunker(DocumentType.CODE)

        assert chunker1 is not chunker2
        assert type(chunker1) is type(chunker2)

    def test_registry_persists_across_calls(self) -> None:
        initial_types = get_supported_types()

        for _ in range(3):
            create_chunker(DocumentType.CODE)

        final_types = get_supported_types()
        assert initial_types == final_types

    def test_all_document_types_workflow(self) -> None:
        for doc_type in [
            DocumentType.CODE,
            DocumentType.MARKDOWN,
            DocumentType.TEXT,
            DocumentType.RUNBOOK,
        ]:
            doc = Document(
                content="test content",
                doc_type=doc_type,
                metadata=Metadata(source_path="/test/file"),
            )
            chunker = get_chunker_for_document(doc)
            assert isinstance(chunker, AbstractChunker)
            assert chunker.config is not None
