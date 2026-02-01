"""LLMProvider Protocol definition.

This module defines the abstract interface for LLM provider implementations.
All LLM providers (Ollama, OpenAI, OCI GenAI, etc.) must conform to this protocol.

Example usage:
    >>> from rag_pipeline.core import LLMProvider
    >>>
    >>> class MyLLMProvider:
    ...     async def generate(self, prompt: str, **kwargs) -> str:
    ...         # Generate response from LLM
    ...         ...
    >>>
    >>> # Type checking: isinstance works at runtime
    >>> assert isinstance(MyLLMProvider(), LLMProvider)
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class MessageRole(str, Enum):
    """Role of a message in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """A message in a conversation.

    Used for structured prompt construction in the generation pipeline.

    Attributes:
        role: The role of the message sender.
        content: The message content.

    Example:
        >>> messages = [
        ...     Message(role=MessageRole.SYSTEM, content="You are helpful."),
        ...     Message(role=MessageRole.USER, content="Hello!"),
        ... ]
    """

    role: MessageRole
    content: str

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary format for API calls."""
        return {"role": self.role.value, "content": self.content}


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM provider implementations.

    LLM providers are responsible for:
    1. Sending prompts to language models
    2. Receiving and formatting responses
    3. Supporting both streaming and non-streaming generation
    4. Handling rate limiting and retries

    All generation methods are async to support non-blocking I/O.
    Implementations should handle connection pooling and retries.

    Implementations:
        - OllamaProvider: Local Ollama server
        - OpenAIProvider (stub): OpenAI API
        - OCIGenAIProvider (stub): OCI Generative AI Service
        - AnthropicProvider (stub): Anthropic API
    """

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
        """Generate a response from the LLM.

        Sends a prompt to the language model and returns the complete
        response as a string.

        Args:
            prompt: The user prompt/query to send to the model.
            system_prompt: Optional system prompt to set context.
            temperature: Sampling temperature (0.0 = deterministic,
                1.0 = more creative). Default: 0.7
            max_tokens: Maximum tokens to generate. None = model default.
            stop_sequences: Sequences that stop generation when encountered.
            **kwargs: Additional provider-specific parameters.

        Returns:
            The generated text response from the model.

        Raises:
            LLMProviderError: If generation fails.
            RetryableError: If the error is transient (rate limit, etc.).

        Example:
            >>> response = await llm.generate(
            ...     prompt="Explain this code: def foo(): pass",
            ...     system_prompt="You are a helpful coding assistant.",
            ...     temperature=0.3
            ... )
            >>> print(response)
        """
        ...

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
        """Generate a streaming response from the LLM.

        Sends a prompt to the language model and yields response
        chunks as they're generated. Useful for real-time display.

        Args:
            prompt: The user prompt/query to send to the model.
            system_prompt: Optional system prompt to set context.
            temperature: Sampling temperature (0.0 = deterministic,
                1.0 = more creative). Default: 0.7
            max_tokens: Maximum tokens to generate. None = model default.
            stop_sequences: Sequences that stop generation when encountered.
            **kwargs: Additional provider-specific parameters.

        Yields:
            String chunks of the response as they're generated.

        Raises:
            LLMProviderError: If generation fails.
            RetryableError: If the error is transient.

        Example:
            >>> async for chunk in llm.generate_stream(
            ...     prompt="Write a story about a robot",
            ...     temperature=0.9
            ... ):
            ...     print(chunk, end="", flush=True)
        """
        ...

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
        """Generate a response using RAG context.

        Convenience method that formats the prompt with retrieved
        context for RAG use cases.

        Args:
            query: The user's question.
            context: Retrieved context from vector search.
            system_prompt: Optional system prompt override.
            temperature: Sampling temperature. Default: 0.3 (more focused).
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional provider-specific parameters.

        Returns:
            The generated response grounded in the provided context.

        Raises:
            LLMProviderError: If generation fails.

        Example:
            >>> response = await llm.generate_with_context(
            ...     query="How does authentication work?",
            ...     context="[1] Source: auth.py\\ndef authenticate(user)..."
            ... )
        """
        ...

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the current model.

        Returns:
            Dictionary containing model information:
                - name: Model name (e.g., "mistral", "gpt-4")
                - provider: Provider name (e.g., "ollama", "openai")
                - context_length: Maximum context window size
                - capabilities: List of capabilities (e.g., ["chat", "code"])

        Example:
            >>> info = llm.get_model_info()
            >>> print(f"Using {info['name']} via {info['provider']}")
        """
        ...

    async def health_check(self) -> bool:
        """Check if the LLM provider is healthy and accessible.

        Returns:
            True if the provider is operational, False otherwise.

        Example:
            >>> if await llm.health_check():
            ...     print("LLM is ready")
            ... else:
            ...     print("LLM is not responding")
        """
        ...


@runtime_checkable
class ChatMessage(Protocol):
    """Protocol for chat message structure.

    Used for multi-turn conversations with chat models.
    """

    @property
    def role(self) -> str:
        """Message role: 'system', 'user', or 'assistant'."""
        ...

    @property
    def content(self) -> str:
        """Message content."""
        ...


@runtime_checkable
class ChatLLMProvider(LLMProvider, Protocol):
    """Extended protocol for chat-based LLM providers.

    Adds support for multi-turn conversations on top of the
    base LLMProvider protocol.
    """

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate a response from a conversation history.

        Args:
            messages: List of chat messages in conversation order.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional provider-specific parameters.

        Returns:
            The assistant's response message content.

        Raises:
            LLMProviderError: If generation fails.
        """
        ...

    async def chat_stream(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a response from a conversation history.

        Args:
            messages: List of chat messages in conversation order.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional provider-specific parameters.

        Yields:
            String chunks of the response as they're generated.
        """
        ...
