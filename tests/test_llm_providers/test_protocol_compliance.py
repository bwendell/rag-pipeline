"""Runtime protocol compliance tests for LLM providers.

These tests verify that implementations satisfy Protocol definitions
at runtime using isinstance checks and attribute presence.
"""

from __future__ import annotations

import pytest

from rag_pipeline.core.base_llm_provider import ChatLLMProvider, LLMProvider
from rag_pipeline.core.base_llm_provider import ChatMessage as ChatMessageProtocol
from rag_pipeline.llm_providers import OllamaProvider
from rag_pipeline.llm_providers.ollama_provider import ChatMessage
from rag_pipeline.llm_providers.stubs import (
    AnthropicProvider,
    OCIGenAIProvider,
    OpenAIProvider,
)


class TestProtocolCompliance:
    """Verify providers implement protocols correctly at runtime."""

    def test_ollama_provider_is_llm_provider(self) -> None:
        """OllamaProvider must satisfy LLMProvider protocol."""
        provider = OllamaProvider()

        assert isinstance(provider, LLMProvider)

    def test_ollama_provider_is_chat_llm_provider(self) -> None:
        """OllamaProvider must satisfy ChatLLMProvider protocol."""
        provider = OllamaProvider()

        assert isinstance(provider, ChatLLMProvider)

    def test_chat_message_satisfies_protocol(self) -> None:
        """ChatMessage dataclass must satisfy ChatMessage protocol."""
        message = ChatMessage(role="user", content="test")

        assert isinstance(message, ChatMessageProtocol)

    def test_chat_message_has_required_properties(self) -> None:
        """ChatMessage must expose role and content properties."""
        message = ChatMessage(role="system", content="hello")

        assert message.role == "system"
        assert message.content == "hello"


class TestStubProtocolCompliance:
    """Verify stubs also satisfy protocol structure."""

    @pytest.mark.parametrize(
        "provider_cls",
        [OCIGenAIProvider, OpenAIProvider, AnthropicProvider],
    )
    def test_stub_providers_have_protocol_methods(self, provider_cls: type[object]) -> None:
        """Stub providers should implement protocol method names."""
        provider = provider_cls()
        required_methods = [
            "generate",
            "generate_stream",
            "generate_with_context",
            "get_model_info",
            "health_check",
            "chat",
            "chat_stream",
        ]

        for method_name in required_methods:
            assert hasattr(provider, method_name), f"Missing {method_name}"
