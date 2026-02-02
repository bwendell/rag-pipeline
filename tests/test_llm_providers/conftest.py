"""Fixtures for LLM provider tests."""

from __future__ import annotations

from typing import Any

import pytest

from rag_pipeline.llm_providers import OllamaConfig


@pytest.fixture
def ollama_config() -> OllamaConfig:
    """Default Ollama configuration."""
    return OllamaConfig(
        base_url="http://localhost:11434",
        model="mistral",
        timeout=30.0,
        max_retries=2,
    )


@pytest.fixture
def mock_ollama_response() -> dict[str, Any]:
    """Mock successful chat response from Ollama."""
    return {
        "model": "mistral",
        "message": {
            "role": "assistant",
            "content": "Hello! How can I help you today?",
        },
        "done": True,
    }


@pytest.fixture
def mock_stream_chunks() -> list[dict[str, Any]]:
    """Mock streaming chunks from Ollama."""
    return [
        {"message": {"content": "Hello"}, "done": False},
        {"message": {"content": " there"}, "done": False},
        {"message": {"content": "!"}, "done": False},
        {"message": {"content": ""}, "done": True},
    ]


@pytest.fixture
def mock_models_response() -> dict[str, Any]:
    """Mock models list response."""
    return {
        "models": [
            {"name": "mistral:latest"},
            {"name": "llama2:latest"},
            {"name": "codellama:latest"},
        ]
    }
