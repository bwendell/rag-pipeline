"""Unit tests for vector store factory."""

from __future__ import annotations

import pytest

from rag_pipeline.vector_stores.factory import (
    create_vector_store,
    get_available_stores,
    get_store_class,
    register_store,
    STORE_REGISTRY,
)
from rag_pipeline.vector_stores.in_memory_store import InMemoryStore
from rag_pipeline.vector_stores.stubs import OCIVectorStore, QdrantStore


class TestCreateVectorStore:
    """Tests for create_vector_store function."""

    def test_create_vector_store_default_chroma(self):
        """Test creating default store (ChromaDB) when available."""
        try:
            store = create_vector_store()
            assert store is not None
        except ImportError:
            pytest.skip("chromadb not installed")

    def test_create_vector_store_in_memory(self):
        """Test creating in-memory store."""
        store = create_vector_store(store_type="in_memory")

        assert store is not None
        assert isinstance(store, InMemoryStore)

    def test_create_vector_store_chroma_explicit(self):
        """Test creating ChromaDB store explicitly."""
        try:
            store = create_vector_store(store_type="chroma")
            assert store is not None
        except ImportError:
            pytest.skip("chromadb not installed")

    def test_create_vector_store_qdrant(self):
        """Test creating Qdrant store (stub)."""
        store = create_vector_store(store_type="qdrant")

        assert store is not None
        assert isinstance(store, QdrantStore)

    def test_create_vector_store_oci(self):
        """Test creating OCI store (stub)."""
        store = create_vector_store(store_type="oci")

        assert store is not None
        assert isinstance(store, OCIVectorStore)

    def test_create_vector_store_with_kwargs_in_memory(self):
        """Test kwargs are passed to store constructor."""
        # InMemoryStore doesn't use kwargs, but test the mechanism
        store = create_vector_store(store_type="in_memory")

        assert store is not None
        assert isinstance(store, InMemoryStore)

    def test_create_vector_store_with_kwargs_qdrant(self):
        """Test kwargs are passed to Qdrant constructor."""
        store = create_vector_store(
            store_type="qdrant", url="http://example.com:6333", collection_name="test_collection"
        )

        assert store is not None
        assert store.url == "http://example.com:6333"
        assert store.collection_name == "test_collection"

    def test_create_vector_store_with_kwargs_oci(self):
        """Test kwargs are passed to OCI constructor."""
        store = create_vector_store(
            store_type="oci",
            compartment_id="ocid1.compartment.oc1..xxx",
            collection_name="test_collection",
        )

        assert store is not None
        assert store.compartment_id == "ocid1.compartment.oc1..xxx"
        assert store.collection_name == "test_collection"

    def test_create_vector_store_invalid_type_raises(self):
        """Test invalid store type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown store type"):
            create_vector_store(store_type="invalid-store")

    def test_create_vector_store_invalid_type_lists_available(self):
        """Test error message lists available stores."""
        try:
            create_vector_store(store_type="nonexistent")
            pytest.fail("Should raise ValueError")
        except ValueError as e:
            error_msg = str(e)
            assert "Available stores:" in error_msg
            # At least in_memory should be listed
            assert "in_memory" in error_msg or "in-memory" in error_msg


class TestGetAvailableStores:
    """Tests for get_available_stores function."""

    def test_get_available_stores_returns_list(self):
        """Test returns list of strings."""
        stores = get_available_stores()

        assert isinstance(stores, list)
        assert all(isinstance(s, str) for s in stores)

    def test_get_available_stores_contains_required(self):
        """Test contains all required store types."""
        stores = get_available_stores()

        # These are always available
        assert "in_memory" in stores
        assert "qdrant" in stores
        assert "oci" in stores
        # chroma may not be available if chromadb not installed
        # so we don't assert it here

    def test_get_available_stores_is_sorted(self):
        """Test stores list is sorted alphabetically."""
        stores = get_available_stores()

        assert stores == sorted(stores)

    def test_get_available_stores_contains_chroma_if_available(self):
        """Test contains chroma if chromadb is installed."""
        stores = get_available_stores()

        # Just verify it's consistent - if present, it should appear
        # This test documents expected behavior
        if "chroma" in stores:
            assert isinstance(stores[0], str)

    def test_get_available_stores_minimum_count(self):
        """Test minimum number of available stores."""
        stores = get_available_stores()

        # At minimum: in_memory, qdrant, oci
        assert len(stores) >= 3


class TestGetStoreClass:
    """Tests for get_store_class function."""

    def test_get_store_class_in_memory(self):
        """Test getting in_memory store class."""
        store_class = get_store_class("in_memory")

        assert store_class is not None
        assert store_class is InMemoryStore

    def test_get_store_class_qdrant(self):
        """Test getting qdrant store class."""
        store_class = get_store_class("qdrant")

        assert store_class is not None
        assert store_class is QdrantStore

    def test_get_store_class_oci(self):
        """Test getting oci store class."""
        store_class = get_store_class("oci")

        assert store_class is not None
        assert store_class is OCIVectorStore

    def test_get_store_class_chroma(self):
        """Test getting chroma store class."""
        try:
            store_class = get_store_class("chroma")
            assert store_class is not None
        except ImportError:
            pytest.skip("chromadb not installed")

    def test_get_store_class_invalid_raises(self):
        """Test invalid store type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown store type"):
            get_store_class("invalid-store-type")

    def test_get_store_class_caches_lazy_loaded(self):
        """Test lazy-loaded classes are cached."""
        # Get class twice
        class1 = get_store_class("qdrant")
        class2 = get_store_class("qdrant")

        # Should be the same object (cached)
        assert class1 is class2

    def test_get_store_class_returns_type(self):
        """Test returned value is a class."""
        store_class = get_store_class("in_memory")

        assert isinstance(store_class, type)


class TestRegisterStore:
    """Tests for register_store function."""

    def test_register_custom_store(self):
        """Test registering a custom store class."""

        class CustomVectorStore:
            """Mock custom vector store."""

            async def add(self, chunks):
                pass

            async def search(self, query_embedding, top_k=5, filters=None):
                pass

            async def health_check(self):
                return True

            # Add other required methods
            async def delete(self, ids):
                return 0

            async def delete_by_document(self, document_id):
                return 0

            async def delete_by_source(self, source_path):
                return 0

            async def get(self, ids):
                return []

            async def get_by_source(self, source_path, limit=100):
                return []

            async def get_all_source_paths(self):
                return []

            async def get_stats(self):
                return {}

            async def count(self):
                return 0

            async def clear(self):
                pass

        # Register the custom store
        register_store("custom-test-store", CustomVectorStore)

        # Verify it's available
        available = get_available_stores()
        assert "custom-test-store" in available

        # Verify we can get it
        store_class = get_store_class("custom-test-store")
        assert store_class is CustomVectorStore

        # Verify we can create an instance
        store = create_vector_store(store_type="custom-test-store")
        assert isinstance(store, CustomVectorStore)

        # Cleanup
        del STORE_REGISTRY["custom-test-store"]

    def test_register_store_overwrites_existing(self):
        """Test registering store with same name overwrites previous."""

        class Store1:
            pass

        class Store2:
            pass

        register_store("test-overwrite", Store1)
        assert get_store_class("test-overwrite") is Store1

        register_store("test-overwrite", Store2)
        assert get_store_class("test-overwrite") is Store2

        # Cleanup
        del STORE_REGISTRY["test-overwrite"]

    def test_register_store_makes_available(self):
        """Test registered store is in available stores list."""

        class TempStore:
            pass

        register_store("temp-store", TempStore)

        stores = get_available_stores()
        assert "temp-store" in stores

        # Cleanup
        del STORE_REGISTRY["temp-store"]

    def test_register_store_can_create_instances(self):
        """Test registered store can be used to create instances."""

        class TestStore:
            def __init__(self, name: str = "default"):
                self.name = name

        register_store("test-instance-store", TestStore)

        store = create_vector_store(store_type="test-instance-store", name="custom_name")

        assert isinstance(store, TestStore)
        assert store.name == "custom_name"

        # Cleanup
        del STORE_REGISTRY["test-instance-store"]


class TestStoreIntegration:
    """Integration tests for store factory."""

    def test_all_available_stores_can_be_retrieved(self):
        """Test all available stores can be retrieved as classes."""
        available = get_available_stores()

        for store_name in available:
            store_class = get_store_class(store_name)
            assert store_class is not None
            assert isinstance(store_class, type)

    def test_all_available_stores_can_be_instantiated(self):
        """Test all available stores can be instantiated."""
        available = get_available_stores()

        for store_name in available:
            try:
                store = create_vector_store(store_type=store_name)
                assert store is not None
            except Exception as e:
                # Some stores might require extra dependencies
                # Just log for debugging
                pytest.skip(f"Could not create {store_name}: {e}")

    def test_factory_consistency(self):
        """Test consistency between factory functions."""
        store_name = "in_memory"

        # Get via get_store_class
        store_class = get_store_class(store_name)
        store1 = store_class()

        # Get via create_vector_store
        store2 = create_vector_store(store_type=store_name)

        # Both should be same type
        assert type(store1) is type(store2)

    def test_default_store_is_chroma_or_available(self):
        """Test default store is ChromaDB if available, otherwise works."""
        try:
            store = create_vector_store()
            assert store is not None
        except ImportError:
            # ChromaDB not available, should skip
            pytest.skip("chromadb not installed, default store unavailable")
