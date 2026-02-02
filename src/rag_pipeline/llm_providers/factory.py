"""Factory for creating LLM providers.

Provides centralized provider creation and management based on
configuration. Supports registration of custom providers.

Example:
    >>> from rag_pipeline.llm_providers import LLMProviderFactory
    >>>
    >>> # Create with default (Ollama)
    >>> provider = factory.create_provider("ollama")
    >>>
    >>> # Use factory singleton
    >>> factory = LLMProviderFactory.get_instance()
    >>> provider = factory.get_provider("ollama")
"""

from __future__ import annotations

from typing import Any, Type

from rag_pipeline.core.exceptions import ConfigurationError
from rag_pipeline.llm_providers.ollama_provider import OllamaConfig, OllamaProvider
from rag_pipeline.llm_providers.stubs.anthropic_provider import (
    AnthropicConfig,
    AnthropicProvider,
)
from rag_pipeline.llm_providers.stubs.oci_genai import OCIGenAIConfig, OCIGenAIProvider
from rag_pipeline.llm_providers.stubs.openai_provider import OpenAIConfig, OpenAIProvider


# Type alias for provider classes
ProviderType = Type[OllamaProvider | OCIGenAIProvider | OpenAIProvider | AnthropicProvider]


class LLMProviderFactory:
    """Factory for creating LLM provider instances.

    Maintains a registry of provider types and creates instances
    based on configuration. Supports custom provider registration.

    Example:
        >>> factory = LLMProviderFactory()
        >>>
        >>> # Create Ollama provider
        >>> provider = factory.create_provider(
        ...     provider_type="ollama",
        ...     model="mistral",
        ... )
        >>>
        >>> # With config object
        >>> config = OllamaConfig(model="codellama")
        >>> provider = factory.create_provider("ollama", config=config)
    """

    # Provider registry
    _registry: dict[str, tuple[ProviderType, type]] = {
        "ollama": (OllamaProvider, OllamaConfig),
        "oci_genai": (OCIGenAIProvider, OCIGenAIConfig),
        "oci": (OCIGenAIProvider, OCIGenAIConfig),  # Alias
        "openai": (OpenAIProvider, OpenAIConfig),
        "anthropic": (AnthropicProvider, AnthropicConfig),
        "claude": (AnthropicProvider, AnthropicConfig),  # Alias
    }

    # Singleton instance
    _instance: LLMProviderFactory | None = None

    def __init__(self) -> None:
        """Initialize the factory."""
        self._provider_cache: dict[str, Any] = {}

    @classmethod
    def get_instance(cls) -> LLMProviderFactory:
        """Get the singleton factory instance.

        Returns:
            The global factory instance.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def create_provider(
        self,
        provider_type: str,
        config: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """Create a new provider instance.

        Args:
            provider_type: Provider type name (e.g., "ollama", "openai").
            config: Optional configuration object.
            **kwargs: Configuration options (if config not provided).

        Returns:
            New provider instance (not started).

        Raises:
            ConfigurationError: If provider type is unknown.
        """
        provider_type = provider_type.lower()

        if provider_type not in self._registry:
            available = list(self._registry.keys())
            raise ConfigurationError(
                f"Unknown LLM provider type: {provider_type}",
                details={
                    "requested": provider_type,
                    "available": available,
                },
            )

        provider_cls, config_cls = self._registry[provider_type]

        # Create config if not provided
        if config is None and kwargs:
            config = config_cls(**kwargs)
        elif config is None:
            config = config_cls()

        return provider_cls(config=config)

    def get_provider(
        self,
        provider_type: str,
        cache: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Get a provider instance, optionally cached.

        Args:
            provider_type: Provider type name.
            cache: Whether to cache the instance.
            **kwargs: Configuration options.

        Returns:
            Provider instance.
        """
        cache_key = f"{provider_type}:{hash(frozenset(kwargs.items()))}"

        if cache and cache_key in self._provider_cache:
            return self._provider_cache[cache_key]

        provider = self.create_provider(provider_type, **kwargs)

        if cache:
            self._provider_cache[cache_key] = provider

        return provider

    @classmethod
    def register(
        cls,
        name: str,
        provider_cls: ProviderType,
        config_cls: type,
    ) -> None:
        """Register a custom provider type.

        Args:
            name: Provider type name.
            provider_cls: Provider class.
            config_cls: Configuration class.
        """
        cls._registry[name.lower()] = (provider_cls, config_cls)

    @classmethod
    def list_providers(cls) -> list[str]:
        """List all registered provider types.

        Returns:
            List of provider type names.
        """
        return list(cls._registry.keys())

    def clear_cache(self) -> None:
        """Clear the provider cache."""
        self._provider_cache.clear()


# Convenience function
def create_llm_provider(provider_type: str = "ollama", **kwargs: Any) -> Any:
    """Create an LLM provider instance.

    Convenience function using the global factory.

    Args:
        provider_type: Provider type name.
        **kwargs: Configuration options.

    Returns:
        New provider instance.

    Example:
        >>> provider = create_llm_provider("ollama", model="mistral")
    """
    factory = LLMProviderFactory.get_instance()
    return factory.create_provider(provider_type, **kwargs)
