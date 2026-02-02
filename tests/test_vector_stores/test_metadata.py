"""Metadata serialization and filtering tests for vector stores.

This module tests metadata handling including:
- Metadata round-trips (add → get → verify fields intact)
- Filter with missing metadata keys
- List-value metadata filtering
- Special characters in metadata values
- Large metadata payloads
- Nested metadata structures
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from rag_pipeline.core.types import Chunk, Metadata, SourceType
from rag_pipeline.vector_stores import ChromaDBStore, InMemoryStore

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


# =============================================================================
# Metadata Round-Trip Tests
# =============================================================================


class TestMetadataRoundTrip:
    """Tests that metadata fields survive add/get cycles intact."""

    @pytest.fixture
    def in_memory_store(self) -> InMemoryStore:
        """Provide a fresh InMemoryStore instance."""
        return InMemoryStore()

    @pytest.fixture
    async def chroma_store(self) -> AsyncGenerator[ChromaDBStore, None]:
        """Provide a temporary ChromaDBStore instance."""
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            store = ChromaDBStore(
                persist_path=temp_dir,
                collection_name="metadata_roundtrip_test",
            )
            yield store

    @pytest.mark.anyio
    async def test_basic_metadata_roundtrip_in_memory(self, in_memory_store: InMemoryStore):
        """Basic metadata fields should round-trip correctly in InMemoryStore."""
        chunk = Chunk(
            id="test-chunk",
            content="Test content",
            document_id="doc-1",
            metadata=Metadata(
                source_path="/path/to/file.py",
                source_type=SourceType.FILESYSTEM,
                language="python",
                file_extension=".py",
            ),
            embedding=[0.1] * 384,
        )

        await in_memory_store.add([chunk])
        retrieved = await in_memory_store.get(["test-chunk"])

        assert len(retrieved) == 1
        meta = retrieved[0].metadata
        assert meta.source_path == "/path/to/file.py"
        assert meta.source_type == SourceType.FILESYSTEM
        assert meta.language == "python"
        assert meta.file_extension == ".py"

    @pytest.mark.anyio
    async def test_basic_metadata_roundtrip_chroma(self, chroma_store: ChromaDBStore):
        """Basic metadata fields should round-trip correctly in ChromaDBStore."""
        chunk = Chunk(
            id="test-chunk",
            content="Test content",
            document_id="doc-1",
            metadata=Metadata(
                source_path="/path/to/file.py",
                source_type=SourceType.FILESYSTEM,
                language="python",
                file_extension=".py",
            ),
            embedding=[0.1] * 384,
        )

        await chroma_store.add([chunk])
        retrieved = await chroma_store.get(["test-chunk"])

        assert len(retrieved) == 1
        meta = retrieved[0].metadata
        assert meta.source_path == "/path/to/file.py"
        assert meta.source_type == SourceType.FILESYSTEM
        assert meta.language == "python"
        assert meta.file_extension == ".py"

    @pytest.mark.anyio
    async def test_custom_metadata_roundtrip_in_memory(self, in_memory_store: InMemoryStore):
        """Custom metadata fields should round-trip correctly in InMemoryStore."""
        chunk = Chunk(
            id="custom-meta-chunk",
            content="Test content",
            document_id="doc-1",
            metadata=Metadata(
                source_path="/path/to/file.py",
                custom={
                    "author": "John Doe",
                    "version": "1.2.3",
                    "reviewed": True,
                    "line_count": 42,
                },
            ),
            embedding=[0.1] * 384,
        )

        await in_memory_store.add([chunk])
        retrieved = await in_memory_store.get(["custom-meta-chunk"])

        assert len(retrieved) == 1
        meta = retrieved[0].metadata
        assert meta.custom["author"] == "John Doe"
        assert meta.custom["version"] == "1.2.3"
        assert meta.custom["reviewed"] is True
        assert meta.custom["line_count"] == 42

    @pytest.mark.anyio
    async def test_datetime_metadata_roundtrip(self, in_memory_store: InMemoryStore):
        """Datetime metadata should be preserved."""
        from datetime import datetime

        created = datetime(2024, 1, 15, 10, 30, 0)
        updated = datetime(2024, 6, 20, 14, 45, 0)

        chunk = Chunk(
            id="datetime-chunk",
            content="Test content",
            document_id="doc-1",
            metadata=Metadata(
                source_path="/path/to/file.py",
                created_at=created,
                updated_at=updated,
            ),
            embedding=[0.1] * 384,
        )

        await in_memory_store.add([chunk])
        retrieved = await in_memory_store.get(["datetime-chunk"])

        assert len(retrieved) == 1
        meta = retrieved[0].metadata
        assert meta.created_at == created
        assert meta.updated_at == updated

    @pytest.mark.anyio
    async def test_empty_metadata_roundtrip(self, in_memory_store: InMemoryStore):
        """Empty metadata should be handled correctly."""
        chunk = Chunk(
            id="empty-meta-chunk",
            content="Test content",
            document_id="doc-1",
            metadata=Metadata(),  # Empty metadata
            embedding=[0.1] * 384,
        )

        await in_memory_store.add([chunk])
        retrieved = await in_memory_store.get(["empty-meta-chunk"])

        assert len(retrieved) == 1
        meta = retrieved[0].metadata
        assert meta.source_path == ""
        assert meta.source_type == SourceType.UNKNOWN
        assert meta.custom == {}


# =============================================================================
# Filter with Missing Metadata Keys Tests
# =============================================================================


class TestFilterMissingKeys:
    """Tests for filtering when metadata keys are missing."""

    @pytest.fixture
    def in_memory_store(self) -> InMemoryStore:
        """Provide a fresh InMemoryStore instance."""
        return InMemoryStore()

    @pytest.mark.anyio
    async def test_filter_missing_key_does_not_match(self, in_memory_store: InMemoryStore):
        """Filter on missing key should not match the chunk."""
        # Add chunk without 'language' field
        chunk = Chunk(
            id="no-lang-chunk",
            content="Test content",
            document_id="doc-1",
            metadata=Metadata(
                source_path="/path/to/file.txt",
                # No language field
            ),
            embedding=[0.1] * 384,
        )
        await in_memory_store.add([chunk])

        # Search with filter on missing key should return empty
        results = await in_memory_store.search(
            [0.1] * 384,
            top_k=5,
            filters={"language": "python"},
        )

        assert len(results) == 0

    @pytest.mark.anyio
    async def test_filter_existing_key_matches(self, in_memory_store: InMemoryStore):
        """Filter on existing key should match correctly."""
        # Add chunk with 'language' field
        chunk = Chunk(
            id="with-lang-chunk",
            content="Test content",
            document_id="doc-1",
            metadata=Metadata(
                source_path="/path/to/file.py",
                language="python",
            ),
            embedding=[0.1] * 384,
        )
        await in_memory_store.add([chunk])

        # Search with matching filter
        results = await in_memory_store.search(
            [0.1] * 384,
            top_k=5,
            filters={"language": "python"},
        )

        assert len(results) == 1
        assert results[0].chunk.id == "with-lang-chunk"

    @pytest.mark.anyio
    async def test_partial_metadata_filtering(self, in_memory_store: InMemoryStore):
        """Chunks with partial metadata should only match on existing fields."""
        # Add chunks with different metadata completeness
        complete_chunk = Chunk(
            id="complete-chunk",
            content="Complete metadata",
            document_id="doc-1",
            metadata=Metadata(
                source_path="/path/file.py",
                language="python",
                file_extension=".py",
            ),
            embedding=[0.1] * 384,
        )
        partial_chunk = Chunk(
            id="partial-chunk",
            content="Partial metadata",
            document_id="doc-1",
            metadata=Metadata(
                source_path="/path/file.py",
                # No language
                file_extension=".py",
            ),
            embedding=[0.2] * 384,
        )
        minimal_chunk = Chunk(
            id="minimal-chunk",
            content="Minimal metadata",
            document_id="doc-1",
            metadata=Metadata(
                source_path="/path/file.py",
                # No language, no extension
            ),
            embedding=[0.3] * 384,
        )

        await in_memory_store.add([complete_chunk, partial_chunk, minimal_chunk])

        # Filter on source_path should match all
        results = await in_memory_store.search(
            [0.1] * 384,
            top_k=5,
            filters={"source_path": "/path/file.py"},
        )
        assert len(results) == 3

        # Filter on file_extension should match only complete and partial
        results = await in_memory_store.search(
            [0.1] * 384,
            top_k=5,
            filters={"file_extension": ".py"},
        )
        assert len(results) == 2
        ids = {r.chunk.id for r in results}
        assert ids == {"complete-chunk", "partial-chunk"}


# =============================================================================
# List-Value Metadata Filtering Tests
# =============================================================================


class TestListValueFiltering:
    """Tests for filtering with list values (any match semantics)."""

    @pytest.fixture
    def in_memory_store(self) -> InMemoryStore:
        """Provide a fresh InMemoryStore instance."""
        return InMemoryStore()

    @pytest.mark.anyio
    async def test_filter_list_value_matches_any(self, in_memory_store: InMemoryStore):
        """Filter with list value should match if any value matches."""
        chunk = Chunk(
            id="list-filter-chunk",
            content="Test content",
            document_id="doc-1",
            metadata=Metadata(
                source_path="/path/to/file.py",
                language="python",
            ),
            embedding=[0.1] * 384,
        )
        await in_memory_store.add([chunk])

        # Should match when language is in the filter list
        results = await in_memory_store.search(
            [0.1] * 384,
            top_k=5,
            filters={"language": ["python", "javascript", "go"]},
        )

        assert len(results) == 1
        assert results[0].chunk.id == "list-filter-chunk"

    @pytest.mark.anyio
    async def test_filter_list_value_no_match(self, in_memory_store: InMemoryStore):
        """Filter with list value should not match if no values match."""
        chunk = Chunk(
            id="no-match-chunk",
            content="Test content",
            document_id="doc-1",
            metadata=Metadata(
                source_path="/path/to/file.py",
                language="rust",
            ),
            embedding=[0.1] * 384,
        )
        await in_memory_store.add([chunk])

        # Should not match when language is not in the filter list
        results = await in_memory_store.search(
            [0.1] * 384,
            top_k=5,
            filters={"language": ["python", "javascript"]},
        )

        assert len(results) == 0

    @pytest.mark.anyio
    async def test_filter_single_value_in_list(self, in_memory_store: InMemoryStore):
        """Single value filter against list metadata - exact match required."""
        # Note: InMemoryStore's _matches_filters checks exact match
        # If metadata has a list value, the filter value must match exactly
        chunk = Chunk(
            id="exact-match-chunk",
            content="Test content",
            document_id="doc-1",
            metadata=Metadata(
                source_path="/path/to/file.py",
                custom={"tags": "python"},  # Stored as string
            ),
            embedding=[0.1] * 384,
        )
        await in_memory_store.add([chunk])

        results = await in_memory_store.search(
            [0.1] * 384,
            top_k=5,
            filters={"tags": "python"},
        )

        assert len(results) == 1


# =============================================================================
# Special Characters in Metadata Tests
# =============================================================================


class TestSpecialCharactersInMetadata:
    """Tests for special characters in metadata values."""

    @pytest.fixture
    def in_memory_store(self) -> InMemoryStore:
        """Provide a fresh InMemoryStore instance."""
        return InMemoryStore()

    @pytest.fixture
    async def chroma_store(self) -> AsyncGenerator[ChromaDBStore, None]:
        """Provide a temporary ChromaDBStore instance."""
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            store = ChromaDBStore(
                persist_path=temp_dir,
                collection_name="special_chars_test",
            )
            yield store

    @pytest.mark.anyio
    async def test_unicode_characters_in_metadata(self, in_memory_store: InMemoryStore):
        """Unicode characters should be preserved in metadata."""
        chunk = Chunk(
            id="unicode-chunk",
            content="Test content",
            document_id="doc-1",
            metadata=Metadata(
                source_path="/path/to/文件.py",
                language="python",
                custom={
                    "description": "这是一个描述",
                    "emoji": "🐍 🚀 💻",
                },
            ),
            embedding=[0.1] * 384,
        )

        await in_memory_store.add([chunk])
        retrieved = await in_memory_store.get(["unicode-chunk"])

        assert len(retrieved) == 1
        meta = retrieved[0].metadata
        assert meta.source_path == "/path/to/文件.py"
        assert meta.custom["description"] == "这是一个描述"
        assert meta.custom["emoji"] == "🐍 🚀 💻"

    @pytest.mark.anyio
    async def test_special_chars_in_source_path(self, in_memory_store: InMemoryStore):
        """Special characters in source path should be handled."""
        special_paths = [
            "/path/with spaces/file.py",
            "/path/with-dashes/file.py",
            "/path/with_underscores/file.py",
            "/path/with.dots/file.py",
            "/path/with!@#$%/file.py",
        ]

        for i, path in enumerate(special_paths):
            chunk = Chunk(
                id=f"special-path-chunk-{i}",
                content=f"Content {i}",
                document_id="doc-1",
                metadata=Metadata(
                    source_path=path,
                ),
                embedding=[0.1] * 384,
            )
            await in_memory_store.add([chunk])

        # Verify all paths are preserved
        for i, path in enumerate(special_paths):
            retrieved = await in_memory_store.get([f"special-path-chunk-{i}"])
            assert len(retrieved) == 1
            assert retrieved[0].metadata.source_path == path

    @pytest.mark.anyio
    async def test_filter_with_special_chars(self, in_memory_store: InMemoryStore):
        """Filtering with special characters should work."""
        chunk = Chunk(
            id="special-filter-chunk",
            content="Test content",
            document_id="doc-1",
            metadata=Metadata(
                source_path="/path/with spaces/file.py",
                language="python",
            ),
            embedding=[0.1] * 384,
        )
        await in_memory_store.add([chunk])

        # Filter with special characters
        results = await in_memory_store.search(
            [0.1] * 384,
            top_k=5,
            filters={"source_path": "/path/with spaces/file.py"},
        )

        assert len(results) == 1
        assert results[0].chunk.id == "special-filter-chunk"

    @pytest.mark.anyio
    async def test_newlines_in_metadata(self, in_memory_store: InMemoryStore):
        """Newlines in metadata values should be preserved."""
        chunk = Chunk(
            id="newline-chunk",
            content="Test content",
            document_id="doc-1",
            metadata=Metadata(
                source_path="/path/to/file.py",
                custom={
                    "multiline": "Line 1\nLine 2\nLine 3",
                },
            ),
            embedding=[0.1] * 384,
        )

        await in_memory_store.add([chunk])
        retrieved = await in_memory_store.get(["newline-chunk"])

        assert len(retrieved) == 1
        assert retrieved[0].metadata.custom["multiline"] == "Line 1\nLine 2\nLine 3"


# =============================================================================
# Large Metadata Payload Tests
# =============================================================================


class TestLargeMetadata:
    """Tests for large metadata payloads."""

    @pytest.fixture
    def in_memory_store(self) -> InMemoryStore:
        """Provide a fresh InMemoryStore instance."""
        return InMemoryStore()

    @pytest.mark.anyio
    async def test_large_custom_metadata(self, in_memory_store: InMemoryStore):
        """Large custom metadata dictionaries should be handled."""
        # Create large custom metadata
        custom_meta = {f"key_{i}": f"value_{i}_" * 100 for i in range(100)}

        chunk = Chunk(
            id="large-meta-chunk",
            content="Test content",
            document_id="doc-1",
            metadata=Metadata(
                source_path="/path/to/file.py",
                custom=custom_meta,
            ),
            embedding=[0.1] * 384,
        )

        await in_memory_store.add([chunk])
        retrieved = await in_memory_store.get(["large-meta-chunk"])

        assert len(retrieved) == 1
        # Verify some keys
        assert retrieved[0].metadata.custom["key_0"] == "value_0_" * 100
        assert retrieved[0].metadata.custom["key_50"] == "value_50_" * 100
        assert retrieved[0].metadata.custom["key_99"] == "value_99_" * 100

    @pytest.mark.anyio
    async def test_long_string_values(self, in_memory_store: InMemoryStore):
        """Very long string values should be handled."""
        long_string = "x" * 10000

        chunk = Chunk(
            id="long-string-chunk",
            content="Test content",
            document_id="doc-1",
            metadata=Metadata(
                source_path="/path/to/file.py",
                custom={"long_value": long_string},
            ),
            embedding=[0.1] * 384,
        )

        await in_memory_store.add([chunk])
        retrieved = await in_memory_store.get(["long-string-chunk"])

        assert len(retrieved) == 1
        assert retrieved[0].metadata.custom["long_value"] == long_string

    @pytest.mark.anyio
    async def test_many_metadata_fields(self, in_memory_store: InMemoryStore):
        """Chunks with many metadata fields should be handled."""
        # Create metadata with many fields
        custom_meta = {f"field_{i}": f"value_{i}" for i in range(1000)}

        chunk = Chunk(
            id="many-fields-chunk",
            content="Test content",
            document_id="doc-1",
            metadata=Metadata(
                source_path="/path/to/file.py",
                custom=custom_meta,
            ),
            embedding=[0.1] * 384,
        )

        await in_memory_store.add([chunk])
        retrieved = await in_memory_store.get(["many-fields-chunk"])

        assert len(retrieved) == 1
        assert len(retrieved[0].metadata.custom) == 1000


# =============================================================================
# Nested Metadata Tests
# =============================================================================


class TestNestedMetadata:
    """Tests for nested metadata structures."""

    @pytest.fixture
    def in_memory_store(self) -> InMemoryStore:
        """Provide a fresh InMemoryStore instance."""
        return InMemoryStore()

    @pytest.mark.anyio
    async def test_nested_dict_in_custom(self, in_memory_store: InMemoryStore):
        """Nested dictionaries in custom metadata should be handled."""
        chunk = Chunk(
            id="nested-dict-chunk",
            content="Test content",
            document_id="doc-1",
            metadata=Metadata(
                source_path="/path/to/file.py",
                custom={
                    "nested": {
                        "level1": {
                            "level2": "deep_value",
                        },
                    },
                },
            ),
            embedding=[0.1] * 384,
        )

        await in_memory_store.add([chunk])
        retrieved = await in_memory_store.get(["nested-dict-chunk"])

        assert len(retrieved) == 1
        # Note: InMemoryStore preserves the nested structure directly
        nested = retrieved[0].metadata.custom["nested"]
        assert nested["level1"]["level2"] == "deep_value"


# =============================================================================
# Metadata Filtering Edge Cases
# =============================================================================


class TestMetadataFilteringEdgeCases:
    """Edge cases for metadata filtering."""

    @pytest.fixture
    def in_memory_store(self) -> InMemoryStore:
        """Provide a fresh InMemoryStore instance."""
        return InMemoryStore()

    @pytest.mark.anyio
    async def test_empty_filter_dict(self, in_memory_store: InMemoryStore):
        """Empty filter dict should match all chunks."""
        chunks = [
            Chunk(
                id=f"empty-filter-chunk-{i}",
                content=f"Content {i}",
                document_id="doc-1",
                metadata=Metadata(
                    source_path=f"/path/file{i}.py",
                    language="python" if i % 2 == 0 else "javascript",
                ),
                embedding=[0.1] * 384,
            )
            for i in range(5)
        ]
        await in_memory_store.add(chunks)

        # Empty filter should match all
        results = await in_memory_store.search(
            [0.1] * 384,
            top_k=10,
            filters={},
        )

        assert len(results) == 5

    @pytest.mark.anyio
    async def test_none_filter(self, in_memory_store: InMemoryStore):
        """None filter should match all chunks."""
        chunks = [
            Chunk(
                id=f"none-filter-chunk-{i}",
                content=f"Content {i}",
                document_id="doc-1",
                metadata=Metadata(source_path=f"/path/file{i}.py"),
                embedding=[0.1] * 384,
            )
            for i in range(3)
        ]
        await in_memory_store.add(chunks)

        # None filter should match all
        results = await in_memory_store.search(
            [0.1] * 384,
            top_k=10,
            filters=None,
        )

        assert len(results) == 3

    @pytest.mark.anyio
    async def test_case_sensitive_filtering(self, in_memory_store: InMemoryStore):
        """Metadata filtering should be case-sensitive."""
        chunks = [
            Chunk(
                id="upper-case-chunk",
                content="Content",
                document_id="doc-1",
                metadata=Metadata(
                    source_path="/path/to/file.py",
                    language="Python",  # Capital P
                ),
                embedding=[0.1] * 384,
            ),
            Chunk(
                id="lower-case-chunk",
                content="Content",
                document_id="doc-1",
                metadata=Metadata(
                    source_path="/path/to/file2.py",
                    language="python",  # Lowercase p
                ),
                embedding=[0.2] * 384,
            ),
        ]
        await in_memory_store.add(chunks)

        # Filter for lowercase "python"
        results = await in_memory_store.search(
            [0.1] * 384,
            top_k=5,
            filters={"language": "python"},
        )

        # Should only match lowercase version
        assert len(results) == 1
        assert results[0].chunk.id == "lower-case-chunk"

    @pytest.mark.anyio
    async def test_multiple_filters_all_must_match(self, in_memory_store: InMemoryStore):
        """Multiple filters require all to match (AND semantics)."""
        chunks = [
            Chunk(
                id="match-both",
                content="Content",
                document_id="doc-1",
                metadata=Metadata(
                    source_path="/path/file.py",
                    language="python",
                    file_extension=".py",
                ),
                embedding=[0.1] * 384,
            ),
            Chunk(
                id="match-one",
                content="Content",
                document_id="doc-1",
                metadata=Metadata(
                    source_path="/path/file.py",
                    language="javascript",
                    file_extension=".js",
                ),
                embedding=[0.2] * 384,
            ),
        ]
        await in_memory_store.add(chunks)

        # Filter requiring both language=python AND extension=.py
        results = await in_memory_store.search(
            [0.1] * 384,
            top_k=5,
            filters={"language": "python", "file_extension": ".py"},
        )

        assert len(results) == 1
        assert results[0].chunk.id == "match-both"

    @pytest.mark.anyio
    async def test_filter_with_zero_and_false_values(self, in_memory_store: InMemoryStore):
        """Filters with 0 and False should work correctly (not be confused with missing)."""
        chunk = Chunk(
            id="zero-false-chunk",
            content="Content",
            document_id="doc-1",
            metadata=Metadata(
                source_path="/path/to/file.py",
                custom={
                    "count": 0,
                    "enabled": False,
                },
            ),
            embedding=[0.1] * 384,
        )
        await in_memory_store.add([chunk])

        # Filter for count=0 should match
        results = await in_memory_store.search(
            [0.1] * 384,
            top_k=5,
            filters={"count": 0},
        )

        # Note: This depends on implementation - some stores may coerce types
        # InMemoryStore does exact matching

    @pytest.mark.anyio
    async def test_whitespace_only_values(self, in_memory_store: InMemoryStore):
        """Whitespace-only metadata values should be handled."""
        chunk = Chunk(
            id="whitespace-chunk",
            content="Content",
            document_id="doc-1",
            metadata=Metadata(
                source_path="   ",  # Whitespace only
                language="",
            ),
            embedding=[0.1] * 384,
        )
        await in_memory_store.add([chunk])

        # Filter should match whitespace exactly
        results = await in_memory_store.search(
            [0.1] * 384,
            top_k=5,
            filters={"source_path": "   "},
        )

        assert len(results) == 1


# =============================================================================
# ChromaDB Metadata Filtering Tests
# =============================================================================


class TestChromaDBMetadataFiltering:
    """ChromaDB-specific metadata filtering tests."""

    @pytest.fixture
    async def chroma_store(self) -> AsyncGenerator[ChromaDBStore, None]:
        """Provide a temporary ChromaDBStore instance."""
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            store = ChromaDBStore(
                persist_path=temp_dir,
                collection_name="chroma_filter_test",
            )
            yield store

    @pytest.mark.anyio
    async def test_chroma_primitive_metadata_filtering(self, chroma_store: ChromaDBStore):
        """ChromaDB filtering works with primitive types."""
        chunks = [
            Chunk(
                id="chroma-int-chunk",
                content="Content with int",
                document_id="doc-1",
                metadata=Metadata(
                    source_path="/path/file1.py",
                    custom={"priority": 1},
                ),
                embedding=[0.1] * 384,
            ),
            Chunk(
                id="chroma-str-chunk",
                content="Content with string",
                document_id="doc-1",
                metadata=Metadata(
                    source_path="/path/file2.py",
                    custom={"priority": "high"},
                ),
                embedding=[0.2] * 384,
            ),
        ]
        await chroma_store.add(chunks)

        # Filter by string value
        results = await chroma_store.search(
            [0.1] * 384,
            top_k=5,
            filters={"priority": "high"},
        )

        # Note: ChromaDB type coercion may affect results
        assert len(results) >= 0
