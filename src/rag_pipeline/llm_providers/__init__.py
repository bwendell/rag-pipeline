"""LLM provider implementations.

This module provides LLM integrations for text generation:
- OllamaProvider: Local LLM via Ollama (implemented)
- OCIGenAIProvider: OCI Generative AI Service (stub)
- OpenAIProvider: OpenAI API (stub)
- AnthropicProvider: Anthropic Claude API (stub)

Example:
    >>> from rag_pipeline.llm_providers import OllamaProvider, LLMProviderFactory
    >>>
    >>> # Direct usage
    >>> async with OllamaProvider() as llm:
    ...     response = await llm.generate("Hello!")
    >>>
    >>> # Factory usage
    >>> factory = LLMProviderFactory()
    >>> llm = factory.create_provider("ollama", model="mistral")
"""

from rag_pipeline.llm_providers.factory import (
    LLMProviderFactory,
    create_llm_provider,
)
from rag_pipeline.llm_providers.ollama_provider import (
    ChatMessage,
    OllamaConfig,
    OllamaProvider,
)
from rag_pipeline.llm_providers.stubs import (
    AnthropicConfig,
    AnthropicProvider,
    OCIGenAIConfig,
    OCIGenAIProvider,
    OpenAIConfig,
    OpenAIProvider,
)

__all__ = [
    # Main provider
    "OllamaProvider",
    "OllamaConfig",
    "ChatMessage",
    # Factory
    "LLMProviderFactory",
    "create_llm_provider",
    # Stubs
    "OCIGenAIProvider",
    "OCIGenAIConfig",
    "OpenAIProvider",
    "OpenAIConfig",
    "AnthropicProvider",
    "AnthropicConfig",
]
