"""Tests for custom exception hierarchy."""

import pytest

from rag_pipeline.core import (
    RAGPipelineError,
    ConfigurationError,
    IngestionError,
    DocumentSourceError,
    ChunkingError,
    EmbeddingError,
    VectorStoreError,
    LLMProviderError,
    QueryError,
    TriggerError,
    RetryableError,
    RetrievalError,
    GenerationError,
)


class TestExceptionHierarchy:
    def test_all_inherit_from_base(self):
        exceptions = [
            ConfigurationError,
            IngestionError,
            DocumentSourceError,
            ChunkingError,
            EmbeddingError,
            VectorStoreError,
            LLMProviderError,
            QueryError,
            TriggerError,
            RetryableError,
            RetrievalError,
            GenerationError,
        ]

        for exc_class in exceptions:
            assert issubclass(exc_class, RAGPipelineError)

    def test_ingestion_subclasses(self):
        assert issubclass(DocumentSourceError, IngestionError)
        assert issubclass(ChunkingError, IngestionError)


class TestRAGPipelineError:
    def test_basic_creation(self):
        err = RAGPipelineError("Something went wrong")

        assert err.message == "Something went wrong"
        assert err.details == {}
        assert str(err) == "Something went wrong"

    def test_with_details(self):
        err = RAGPipelineError("Operation failed", details={"id": "123", "type": "test"})

        assert err.details["id"] == "123"
        assert err.details["type"] == "test"
        assert "id='123'" in str(err)
        assert "type='test'" in str(err)

    def test_repr(self):
        err = RAGPipelineError("test", details={"key": "value"})

        repr_str = repr(err)
        assert "RAGPipelineError" in repr_str
        assert "test" in repr_str
        assert "key" in repr_str

    def test_can_be_caught_as_exception(self):
        with pytest.raises(Exception):
            raise RAGPipelineError("test")


class TestSpecificExceptions:
    def test_configuration_error(self):
        err = ConfigurationError("Invalid setting", details={"setting": "chunk_size", "value": -1})

        assert "Invalid setting" in str(err)
        assert err.details["setting"] == "chunk_size"

    def test_document_source_error(self):
        err = DocumentSourceError("File not found", details={"path": "/missing/file.py"})

        assert isinstance(err, IngestionError)
        assert err.details["path"] == "/missing/file.py"

    def test_chunking_error(self):
        err = ChunkingError("Parse error", details={"file": "broken.py", "line": 42})

        assert isinstance(err, IngestionError)
        assert err.details["line"] == 42

    def test_embedding_error(self):
        err = EmbeddingError("Model not found", details={"model": "nonexistent-model"})

        assert err.details["model"] == "nonexistent-model"

    def test_vector_store_error(self):
        err = VectorStoreError(
            "Connection failed", details={"store": "chroma", "host": "localhost"}
        )

        assert err.details["store"] == "chroma"

    def test_llm_provider_error(self):
        err = LLMProviderError("Timeout", details={"provider": "ollama", "timeout": 30})

        assert err.details["timeout"] == 30

    def test_query_error(self):
        err = QueryError("Query processing failed", details={"stage": "retrieval"})

        assert err.details["stage"] == "retrieval"

    def test_trigger_error(self):
        err = TriggerError("Webhook failed", details={"trigger": "slack"})

        assert err.details["trigger"] == "slack"

    def test_retrieval_error(self):
        err = RetrievalError("Vector search failed", details={"store": "chroma", "query": "auth"})

        assert err.details["store"] == "chroma"
        assert err.details["query"] == "auth"

    def test_generation_error(self):
        err = GenerationError("LLM failed", details={"model": "mistral", "reason": "timeout"})

        assert err.details["model"] == "mistral"
        assert err.details["reason"] == "timeout"


class TestRetryableError:
    def test_default_retry_after(self):
        err = RetryableError("Service unavailable")

        assert err.retry_after == 1.0

    def test_custom_retry_after(self):
        err = RetryableError("Rate limited", details={"limit": 100}, retry_after=5.0)

        assert err.retry_after == 5.0
        assert err.details["limit"] == 100

    def test_inherits_from_base(self):
        err = RetryableError("test")

        assert isinstance(err, RAGPipelineError)


class TestExceptionCatching:
    def test_catch_specific_exception(self):
        try:
            raise VectorStoreError("test")
        except VectorStoreError as e:
            assert e.message == "test"
        except RAGPipelineError:
            pytest.fail("Should have caught VectorStoreError specifically")

    def test_catch_base_exception(self):
        exceptions_to_test = [
            ConfigurationError("config"),
            EmbeddingError("embed"),
            VectorStoreError("store"),
            LLMProviderError("llm"),
        ]

        for exc in exceptions_to_test:
            try:
                raise exc
            except RAGPipelineError as e:
                assert e.message is not None
            except Exception:
                pytest.fail(f"Should have caught {type(exc).__name__} as RAGPipelineError")

    def test_catch_ingestion_hierarchy(self):
        try:
            raise ChunkingError("chunk failed")
        except IngestionError as e:
            assert "chunk failed" in e.message
        except RAGPipelineError:
            pytest.fail("Should have caught as IngestionError")
