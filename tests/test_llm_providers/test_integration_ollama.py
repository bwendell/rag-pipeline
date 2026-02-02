"""Integration tests with a live Ollama server."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from rag_pipeline.llm_providers import ChatMessage, OllamaProvider

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

pytestmark = pytest.mark.integration


@pytest.fixture
async def live_provider() -> AsyncIterator[OllamaProvider]:
    provider = OllamaProvider()
    await provider.start()

    if not await provider.health_check():
        await provider.stop()
        pytest.skip("Ollama not running or model not available")

    yield provider

    await provider.stop()


@pytest.mark.asyncio
async def test_simple_generation(live_provider: OllamaProvider) -> None:
    response = await live_provider.generate("Say hello in one word.", temperature=0.0)

    assert response


@pytest.mark.asyncio
async def test_streaming_generation(live_provider: OllamaProvider) -> None:
    chunks: list[str] = []
    async for chunk in live_provider.generate_stream("Count from 1 to 3.", temperature=0.0):
        chunks.append(chunk)

    assert "".join(chunks)


@pytest.mark.asyncio
async def test_chat_multi_turn(live_provider: OllamaProvider) -> None:
    messages = [
        ChatMessage(role="system", content="You are concise."),
        ChatMessage(role="user", content="Say hello"),
        ChatMessage(role="assistant", content="Hello"),
        ChatMessage(role="user", content="Now say goodbye"),
    ]

    response = await live_provider.chat(messages, temperature=0.0)

    assert response


@pytest.mark.asyncio
async def test_list_models(live_provider: OllamaProvider) -> None:
    models = await live_provider.list_models()

    assert isinstance(models, list)
    assert models
