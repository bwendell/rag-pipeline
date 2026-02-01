"""Tests for Protocol definitions and runtime checking."""

import pytest
from typing import Any, AsyncIterator
from datetime import datetime

from rag_pipeline.core import (
    Chunk,
    Chunker,
    Document,
    DocumentSource,
    DocumentType,
    EmbeddingProvider,
    LLMProvider,
    Message,
    MessageRole,
    Metadata,
    SearchResult,
    VectorStore,
    Trigger,
    QueryRequest,
    QueryResponse,
)


class TestVectorStoreProtocol:
    def test_isinstance_check_positive(self):
        class MockVectorStore:
            async def add(self, chunks: list[Chunk]) -> list[str]:
                return [c.id for c in chunks]

            async def search(
                self,
                query_embedding: list[float],
                top_k: int = 5,
                filters: dict[str, Any] | None = None,
            ) -> list[SearchResult]:
                return []

            async def delete(self, ids: list[str]) -> int:
                return 0

            async def delete_by_document(self, document_id: str) -> int:
                return 0

            async def delete_by_source(self, source_path: str) -> int:
                return 0

            async def get(self, ids: list[str]) -> list[Chunk]:
                return []

            async def get_by_source(self, source_path: str, limit: int = 100) -> list[Chunk]:
                return []

            async def get_all_source_paths(self) -> list[str]:
                return []

            async def get_stats(self) -> dict[str, Any]:
                return {"total_chunks": 0}

            async def count(self) -> int:
                return 0

            async def clear(self) -> None:
                pass

            async def health_check(self) -> bool:
                return True

        store = MockVectorStore()
        assert isinstance(store, VectorStore)

    def test_isinstance_check_negative(self):
        class NotAVectorStore:
            def add(self, chunks: list):
                pass

        assert not isinstance(NotAVectorStore(), VectorStore)


class TestLLMProviderProtocol:
    def test_isinstance_check_positive(self):
        class MockLLMProvider:
            async def generate(
                self,
                prompt: str,
                *,
                system_prompt: str | None = None,
                temperature: float = 0.7,
                max_tokens: int | None = None,
                stop_sequences: list[str] | None = None,
                **kwargs: Any,
            ) -> str:
                return "response"

            async def generate_stream(
                self,
                prompt: str,
                *,
                system_prompt: str | None = None,
                temperature: float = 0.7,
                max_tokens: int | None = None,
                stop_sequences: list[str] | None = None,
                **kwargs: Any,
            ) -> AsyncIterator[str]:
                yield "chunk"

            async def generate_with_context(
                self,
                query: str,
                context: str,
                *,
                system_prompt: str | None = None,
                temperature: float = 0.3,
                max_tokens: int | None = None,
                **kwargs: Any,
            ) -> str:
                return "response"

            def get_model_info(self) -> dict[str, Any]:
                return {"name": "test", "provider": "mock"}

            async def health_check(self) -> bool:
                return True

        provider = MockLLMProvider()
        assert isinstance(provider, LLMProvider)


class TestEmbeddingProviderProtocol:
    def test_isinstance_check_positive(self):
        class MockEmbeddingProvider:
            async def embed(self, text: str) -> list[float]:
                return [0.1, 0.2, 0.3]

            async def embed_batch(
                self,
                texts: list[str],
                *,
                batch_size: int | None = None,
                show_progress: bool = False,
            ) -> list[list[float]]:
                return [[0.1, 0.2, 0.3] for _ in texts]

            async def embed_query(self, query: str) -> list[float]:
                return [0.1, 0.2, 0.3]

            def get_embedding_dimension(self) -> int:
                return 3

            def get_model_info(self) -> dict[str, Any]:
                return {"name": "test"}

            async def health_check(self) -> bool:
                return True

        provider = MockEmbeddingProvider()
        assert isinstance(provider, EmbeddingProvider)


class TestDocumentSourceProtocol:
    def test_isinstance_check_positive(self):
        class MockDocumentSource:
            async def load(self) -> AsyncIterator[Document]:
                yield Document(content="test")

            async def load_changed(self, since: datetime) -> AsyncIterator[Document]:
                yield Document(content="changed")

            async def load_by_path(self, path: str) -> Document | None:
                return None

            async def list_paths(self) -> list[str]:
                return []

            async def get_changed_paths(self, since: datetime) -> list[str]:
                return []

            async def get_deleted_paths(self, known_paths: list[str]) -> list[str]:
                return []

            def get_source_info(self) -> dict[str, Any]:
                return {"type": "mock"}

            async def health_check(self) -> bool:
                return True

        source = MockDocumentSource()
        assert isinstance(source, DocumentSource)


class TestChunkerProtocol:
    def test_isinstance_check_positive(self):
        class MockChunker:
            def chunk(self, document: Document) -> list[Chunk]:
                return [Chunk(content=document.content, document_id=document.id)]

            def get_supported_types(self) -> list[str]:
                return ["code"]

            def get_config(self) -> dict[str, Any]:
                return {"chunk_size": 1000}

        chunker = MockChunker()
        assert isinstance(chunker, Chunker)


class TestTriggerProtocol:
    def test_isinstance_check_positive(self):
        class MockTrigger:
            async def start(self) -> None:
                pass

            async def stop(self) -> None:
                pass

            def on_query(self, callback) -> None:
                pass

            def get_trigger_info(self) -> dict[str, Any]:
                return {"type": "mock"}

            async def health_check(self) -> bool:
                return True

        trigger = MockTrigger()
        assert isinstance(trigger, Trigger)


class TestQueryRequestResponse:
    def test_query_request_defaults(self):
        req = QueryRequest(query="How does it work?")

        assert req.query == "How does it work?"
        assert req.request_id is not None
        assert req.user_id is None
        assert req.channel_id is None
        assert isinstance(req.timestamp, datetime)
        assert req.metadata == {}

    def test_query_request_with_values(self):
        req = QueryRequest(
            query="test",
            user_id="user123",
            channel_id="C12345",
            metadata={"source": "slack"},
        )

        assert req.user_id == "user123"
        assert req.channel_id == "C12345"
        assert req.metadata["source"] == "slack"

    def test_query_response_defaults(self):
        resp = QueryResponse(answer="The answer is 42", request_id="req-123")

        assert resp.answer == "The answer is 42"
        assert resp.request_id == "req-123"
        assert resp.sources == []
        assert resp.confidence is None
        assert resp.processing_time_ms == 0
        assert resp.error is None
        assert resp.is_error is False

    def test_query_response_error(self):
        resp = QueryResponse(answer="", request_id="req-123", error="Something went wrong")

        assert resp.is_error is True
        assert resp.error == "Something went wrong"


class TestMessageTypes:
    def test_message_role_values(self):
        assert MessageRole.SYSTEM.value == "system"
        assert MessageRole.USER.value == "user"
        assert MessageRole.ASSISTANT.value == "assistant"

    def test_message_creation(self):
        msg = Message(role=MessageRole.USER, content="Hello")

        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"

    def test_message_to_dict(self):
        msg = Message(role=MessageRole.SYSTEM, content="You are helpful")

        assert msg.to_dict() == {"role": "system", "content": "You are helpful"}

    def test_message_all_roles(self):
        for role in MessageRole:
            msg = Message(role=role, content="test")
            result = msg.to_dict()

            assert result["role"] == role.value
            assert result["content"] == "test"
