"""Unit tests for vector store stubs."""

from __future__ import annotations

import pytest

from rag_pipeline.vector_stores.stubs import OCIVectorStore, QdrantStore


class TestOCIVectorStoreStub:
    """Tests for OCI vector store stub."""

    def test_instantiation(self):
        """Test OCIVectorStore can be instantiated."""
        store = OCIVectorStore()

        assert store is not None
        assert isinstance(store, OCIVectorStore)

    def test_instantiation_with_defaults(self):
        """Test OCIVectorStore has sensible defaults."""
        store = OCIVectorStore()

        assert store.collection_name == "rag_chunks"
        assert store.compartment_id is None
        assert store.database_connection_string is None

    def test_instantiation_with_compartment_id(self):
        """Test OCIVectorStore accepts compartment_id parameter."""
        compartment_id = "ocid1.compartment.oc1.region.aaaaaaaxxxxx"
        store = OCIVectorStore(compartment_id=compartment_id)

        assert store.compartment_id == compartment_id

    def test_instantiation_with_database_connection_string(self):
        """Test OCIVectorStore accepts database_connection_string parameter."""
        conn_str = "oracle://user:pass@host:1521/ORCL"
        store = OCIVectorStore(database_connection_string=conn_str)

        assert store.database_connection_string == conn_str

    def test_instantiation_with_collection_name(self):
        """Test OCIVectorStore accepts custom collection_name."""
        store = OCIVectorStore(collection_name="my_collection")

        assert store.collection_name == "my_collection"

    def test_instantiation_with_all_parameters(self):
        """Test OCIVectorStore accepts all parameters."""
        compartment_id = "ocid1.compartment.oc1..xxx"
        conn_str = "oracle://user:pass@host:1521/ORCL"
        collection = "docs_chunks"

        store = OCIVectorStore(
            compartment_id=compartment_id,
            database_connection_string=conn_str,
            collection_name=collection,
        )

        assert store.compartment_id == compartment_id
        assert store.database_connection_string == conn_str
        assert store.collection_name == collection

    @pytest.mark.anyio
    async def test_add_raises_not_implemented(self):
        """Test add method raises NotImplementedError."""
        store = OCIVectorStore()

        with pytest.raises(NotImplementedError) as exc_info:
            await store.add([])

        assert "stub" in str(exc_info.value).lower()

    @pytest.mark.anyio
    async def test_search_raises_not_implemented(self):
        """Test search method raises NotImplementedError."""
        store = OCIVectorStore()

        with pytest.raises(NotImplementedError) as exc_info:
            await store.search([0.1, 0.2, 0.3])

        assert "stub" in str(exc_info.value).lower()

    @pytest.mark.anyio
    async def test_delete_raises_not_implemented(self):
        """Test delete method raises NotImplementedError."""
        store = OCIVectorStore()

        with pytest.raises(NotImplementedError) as exc_info:
            await store.delete(["id1", "id2"])

        assert "stub" in str(exc_info.value).lower()

    @pytest.mark.anyio
    async def test_delete_by_document_raises_not_implemented(self):
        """Test delete_by_document method raises NotImplementedError."""
        store = OCIVectorStore()

        with pytest.raises(NotImplementedError) as exc_info:
            await store.delete_by_document("doc-1")

        assert "stub" in str(exc_info.value).lower()

    @pytest.mark.anyio
    async def test_delete_by_source_raises_not_implemented(self):
        """Test delete_by_source method raises NotImplementedError."""
        store = OCIVectorStore()

        with pytest.raises(NotImplementedError) as exc_info:
            await store.delete_by_source("/path/to/source")

        assert "stub" in str(exc_info.value).lower()

    @pytest.mark.anyio
    async def test_get_raises_not_implemented(self):
        """Test get method raises NotImplementedError."""
        store = OCIVectorStore()

        with pytest.raises(NotImplementedError) as exc_info:
            await store.get(["id1", "id2"])

        assert "stub" in str(exc_info.value).lower()

    @pytest.mark.anyio
    async def test_get_by_source_raises_not_implemented(self):
        """Test get_by_source method raises NotImplementedError."""
        store = OCIVectorStore()

        with pytest.raises(NotImplementedError) as exc_info:
            await store.get_by_source("/path/to/source")

        assert "stub" in str(exc_info.value).lower()

    @pytest.mark.anyio
    async def test_get_all_source_paths_raises_not_implemented(self):
        """Test get_all_source_paths method raises NotImplementedError."""
        store = OCIVectorStore()

        with pytest.raises(NotImplementedError) as exc_info:
            await store.get_all_source_paths()

        assert "stub" in str(exc_info.value).lower()

    @pytest.mark.anyio
    async def test_get_stats_raises_not_implemented(self):
        """Test get_stats method raises NotImplementedError."""
        store = OCIVectorStore()

        with pytest.raises(NotImplementedError) as exc_info:
            await store.get_stats()

        assert "stub" in str(exc_info.value).lower()

    @pytest.mark.anyio
    async def test_count_raises_not_implemented(self):
        """Test count method raises NotImplementedError."""
        store = OCIVectorStore()

        with pytest.raises(NotImplementedError) as exc_info:
            await store.count()

        assert "stub" in str(exc_info.value).lower()

    @pytest.mark.anyio
    async def test_clear_raises_not_implemented(self):
        """Test clear method raises NotImplementedError."""
        store = OCIVectorStore()

        with pytest.raises(NotImplementedError) as exc_info:
            await store.clear()

        assert "stub" in str(exc_info.value).lower()

    @pytest.mark.anyio
    async def test_health_check_raises_not_implemented(self):
        """Test health_check method raises NotImplementedError."""
        store = OCIVectorStore()

        with pytest.raises(NotImplementedError) as exc_info:
            await store.health_check()

        assert "stub" in str(exc_info.value).lower()

    @pytest.mark.anyio
    async def test_error_messages_mention_oci(self):
        """Test error messages mention OCI."""
        store = OCIVectorStore()

        try:
            await store.add([])
            pytest.fail("Should raise NotImplementedError")
        except NotImplementedError as e:
            error_msg = str(e).lower()
            assert "oci" in error_msg or "stub" in error_msg

    @pytest.mark.anyio
    async def test_all_methods_have_helpful_messages(self):
        """Test that all methods have helpful error messages."""
        store = OCIVectorStore()

        methods = [
            store.add([]),
            store.search([0.1]),
            store.delete([]),
            store.delete_by_document("doc"),
            store.delete_by_source("path"),
            store.get([]),
            store.get_by_source("path"),
            store.get_all_source_paths(),
            store.get_stats(),
            store.count(),
            store.clear(),
            store.health_check(),
        ]

        for method_call in methods:
            try:
                await method_call
                pytest.fail("Should raise NotImplementedError")
            except NotImplementedError as e:
                # Verify message is not empty and contains helpful info
                msg = str(e)
                assert len(msg) > 0
                assert "stub" in msg.lower() or "not implemented" in msg.lower()


class TestQdrantVectorStoreStub:
    """Tests for Qdrant vector store stub."""

    def test_instantiation(self):
        """Test QdrantStore can be instantiated."""
        store = QdrantStore()

        assert store is not None
        assert isinstance(store, QdrantStore)

    def test_instantiation_with_defaults(self):
        """Test QdrantStore has sensible defaults."""
        store = QdrantStore()

        assert store.url == "http://localhost:6333"
        assert store.collection_name == "rag_chunks"
        assert store.api_key is None

    def test_instantiation_with_url(self):
        """Test QdrantStore accepts url parameter."""
        url = "http://example.com:6333"
        store = QdrantStore(url=url)

        assert store.url == url

    def test_instantiation_with_api_key(self):
        """Test QdrantStore accepts api_key parameter."""
        api_key = "test-api-key-123"
        store = QdrantStore(api_key=api_key)

        assert store.api_key == api_key

    def test_instantiation_with_collection_name(self):
        """Test QdrantStore accepts custom collection_name."""
        store = QdrantStore(collection_name="my_vectors")

        assert store.collection_name == "my_vectors"

    def test_instantiation_with_all_parameters(self):
        """Test QdrantStore accepts all parameters."""
        url = "https://qdrant-cloud.example.com:6333"
        api_key = "cloud-api-key"
        collection = "embeddings"

        store = QdrantStore(url=url, api_key=api_key, collection_name=collection)

        assert store.url == url
        assert store.api_key == api_key
        assert store.collection_name == collection

    @pytest.mark.anyio
    async def test_add_raises_not_implemented(self):
        """Test add method raises NotImplementedError."""
        store = QdrantStore()

        with pytest.raises(NotImplementedError) as exc_info:
            await store.add([])

        assert "stub" in str(exc_info.value).lower()

    @pytest.mark.anyio
    async def test_search_raises_not_implemented(self):
        """Test search method raises NotImplementedError."""
        store = QdrantStore()

        with pytest.raises(NotImplementedError) as exc_info:
            await store.search([0.1, 0.2, 0.3])

        assert "stub" in str(exc_info.value).lower()

    @pytest.mark.anyio
    async def test_delete_raises_not_implemented(self):
        """Test delete method raises NotImplementedError."""
        store = QdrantStore()

        with pytest.raises(NotImplementedError) as exc_info:
            await store.delete(["id1", "id2"])

        assert "stub" in str(exc_info.value).lower()

    @pytest.mark.anyio
    async def test_delete_by_document_raises_not_implemented(self):
        """Test delete_by_document method raises NotImplementedError."""
        store = QdrantStore()

        with pytest.raises(NotImplementedError) as exc_info:
            await store.delete_by_document("doc-1")

        assert "stub" in str(exc_info.value).lower()

    @pytest.mark.anyio
    async def test_delete_by_source_raises_not_implemented(self):
        """Test delete_by_source method raises NotImplementedError."""
        store = QdrantStore()

        with pytest.raises(NotImplementedError) as exc_info:
            await store.delete_by_source("/path/to/source")

        assert "stub" in str(exc_info.value).lower()

    @pytest.mark.anyio
    async def test_get_raises_not_implemented(self):
        """Test get method raises NotImplementedError."""
        store = QdrantStore()

        with pytest.raises(NotImplementedError) as exc_info:
            await store.get(["id1", "id2"])

        assert "stub" in str(exc_info.value).lower()

    @pytest.mark.anyio
    async def test_get_by_source_raises_not_implemented(self):
        """Test get_by_source method raises NotImplementedError."""
        store = QdrantStore()

        with pytest.raises(NotImplementedError) as exc_info:
            await store.get_by_source("/path/to/source")

        assert "stub" in str(exc_info.value).lower()

    @pytest.mark.anyio
    async def test_get_all_source_paths_raises_not_implemented(self):
        """Test get_all_source_paths method raises NotImplementedError."""
        store = QdrantStore()

        with pytest.raises(NotImplementedError) as exc_info:
            await store.get_all_source_paths()

        assert "stub" in str(exc_info.value).lower()

    @pytest.mark.anyio
    async def test_get_stats_raises_not_implemented(self):
        """Test get_stats method raises NotImplementedError."""
        store = QdrantStore()

        with pytest.raises(NotImplementedError) as exc_info:
            await store.get_stats()

        assert "stub" in str(exc_info.value).lower()

    @pytest.mark.anyio
    async def test_count_raises_not_implemented(self):
        """Test count method raises NotImplementedError."""
        store = QdrantStore()

        with pytest.raises(NotImplementedError) as exc_info:
            await store.count()

        assert "stub" in str(exc_info.value).lower()

    @pytest.mark.anyio
    async def test_clear_raises_not_implemented(self):
        """Test clear method raises NotImplementedError."""
        store = QdrantStore()

        with pytest.raises(NotImplementedError) as exc_info:
            await store.clear()

        assert "stub" in str(exc_info.value).lower()

    @pytest.mark.anyio
    async def test_health_check_raises_not_implemented(self):
        """Test health_check method raises NotImplementedError."""
        store = QdrantStore()

        with pytest.raises(NotImplementedError) as exc_info:
            await store.health_check()

        assert "stub" in str(exc_info.value).lower()

    @pytest.mark.anyio
    async def test_error_messages_mention_qdrant(self):
        """Test error messages mention Qdrant."""
        store = QdrantStore()

        try:
            await store.add([])
            pytest.fail("Should raise NotImplementedError")
        except NotImplementedError as e:
            error_msg = str(e).lower()
            assert "qdrant" in error_msg or "stub" in error_msg

    @pytest.mark.anyio
    async def test_all_methods_have_helpful_messages(self):
        """Test that all methods have helpful error messages."""
        store = QdrantStore()

        methods = [
            store.add([]),
            store.search([0.1]),
            store.delete([]),
            store.delete_by_document("doc"),
            store.delete_by_source("path"),
            store.get([]),
            store.get_by_source("path"),
            store.get_all_source_paths(),
            store.get_stats(),
            store.count(),
            store.clear(),
            store.health_check(),
        ]

        for method_call in methods:
            try:
                await method_call
                pytest.fail("Should raise NotImplementedError")
            except NotImplementedError as e:
                # Verify message is not empty and contains helpful info
                msg = str(e)
                assert len(msg) > 0
                assert "stub" in msg.lower() or "not implemented" in msg.lower()


class TestStubsComparison:
    """Tests comparing OCI and Qdrant stubs."""

    def test_both_stubs_accept_collection_name(self):
        """Test both stubs accept collection_name parameter."""
        oci_store = OCIVectorStore(collection_name="test_oci")
        qdrant_store = QdrantStore(collection_name="test_qdrant")

        assert oci_store.collection_name == "test_oci"
        assert qdrant_store.collection_name == "test_qdrant"

    def test_both_stubs_have_same_12_methods(self):
        """Test both stubs implement the same 12 methods."""
        oci_methods = {
            "add",
            "search",
            "delete",
            "delete_by_document",
            "delete_by_source",
            "get",
            "get_by_source",
            "get_all_source_paths",
            "get_stats",
            "count",
            "clear",
            "health_check",
        }

        oci_store = OCIVectorStore()
        qdrant_store = QdrantStore()

        # Check OCI has all methods
        for method_name in oci_methods:
            assert hasattr(oci_store, method_name), f"OCI missing {method_name}"
            assert callable(getattr(oci_store, method_name))

        # Check Qdrant has all methods
        for method_name in oci_methods:
            assert hasattr(qdrant_store, method_name), f"Qdrant missing {method_name}"
            assert callable(getattr(qdrant_store, method_name))

    @pytest.mark.anyio
    async def test_both_stubs_raise_not_implemented(self):
        """Test both stubs raise NotImplementedError for all methods."""
        oci_store = OCIVectorStore()
        qdrant_store = QdrantStore()

        # Test OCI store
        with pytest.raises(NotImplementedError):
            await oci_store.add([])
        with pytest.raises(NotImplementedError):
            await oci_store.search([0.1])
        with pytest.raises(NotImplementedError):
            await oci_store.delete([])
        with pytest.raises(NotImplementedError):
            await oci_store.delete_by_document("doc")
        with pytest.raises(NotImplementedError):
            await oci_store.delete_by_source("path")
        with pytest.raises(NotImplementedError):
            await oci_store.get([])
        with pytest.raises(NotImplementedError):
            await oci_store.get_by_source("path")
        with pytest.raises(NotImplementedError):
            await oci_store.get_all_source_paths()
        with pytest.raises(NotImplementedError):
            await oci_store.get_stats()
        with pytest.raises(NotImplementedError):
            await oci_store.count()
        with pytest.raises(NotImplementedError):
            await oci_store.clear()
        with pytest.raises(NotImplementedError):
            await oci_store.health_check()

        # Test Qdrant store
        with pytest.raises(NotImplementedError):
            await qdrant_store.add([])
        with pytest.raises(NotImplementedError):
            await qdrant_store.search([0.1])
        with pytest.raises(NotImplementedError):
            await qdrant_store.delete([])
        with pytest.raises(NotImplementedError):
            await qdrant_store.delete_by_document("doc")
        with pytest.raises(NotImplementedError):
            await qdrant_store.delete_by_source("path")
        with pytest.raises(NotImplementedError):
            await qdrant_store.get([])
        with pytest.raises(NotImplementedError):
            await qdrant_store.get_by_source("path")
        with pytest.raises(NotImplementedError):
            await qdrant_store.get_all_source_paths()
        with pytest.raises(NotImplementedError):
            await qdrant_store.get_stats()
        with pytest.raises(NotImplementedError):
            await qdrant_store.count()
        with pytest.raises(NotImplementedError):
            await qdrant_store.clear()
        with pytest.raises(NotImplementedError):
            await qdrant_store.health_check()
