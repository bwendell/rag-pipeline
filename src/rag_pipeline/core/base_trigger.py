"""Trigger Protocol definition.

This module defines the abstract interface for trigger implementations.
Triggers are entry points that receive queries and invoke the RAG pipeline.

Examples of triggers:
- HTTP API endpoints (FastAPI)
- Slack bot
- Microsoft Teams bot
- CLI interface

Example usage:
    >>> from rag_pipeline.core import Trigger
    >>>
    >>> class MyTrigger:
    ...     async def start(self) -> None:
    ...         # Start listening for queries
    ...         ...
    ...
    ...     async def stop(self) -> None:
    ...         # Stop listening
    ...         ...
    >>>
    >>> # Type checking: isinstance works at runtime
    >>> assert isinstance(MyTrigger(), Trigger)
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4


@dataclass
class QueryRequest:
    """Incoming query request from a trigger.

    Represents a user's question along with context about
    where the request came from.

    Attributes:
        query: The user's question.
        request_id: Unique identifier for this request.
        user_id: Identifier for the user (if available).
        channel_id: Channel/conversation ID (for chat triggers).
        timestamp: When the request was received.
        metadata: Additional trigger-specific data.

    Example:
        >>> request = QueryRequest(
        ...     query="How does authentication work?",
        ...     user_id="user123",
        ...     channel_id="C12345"
        ... )
    """

    query: str
    request_id: str = field(default_factory=lambda: str(uuid4()))
    user_id: str | None = None
    channel_id: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResponse:
    """Response to a query request.

    Contains the generated answer along with sources and metadata.

    Attributes:
        answer: The generated response text.
        request_id: ID of the original request.
        sources: List of source documents used.
        confidence: Optional confidence score (0-1).
        processing_time_ms: Time to process in milliseconds.
        metadata: Additional response metadata.

    Example:
        >>> response = QueryResponse(
        ...     answer="Authentication uses JWT tokens...",
        ...     request_id=request.request_id,
        ...     sources=["auth.py:42", "README.md:10"],
        ...     processing_time_ms=1234
        ... )
    """

    answer: str
    request_id: str
    sources: list[str] = field(default_factory=list)
    confidence: float | None = None
    processing_time_ms: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def is_error(self) -> bool:
        """Check if this response represents an error."""
        return self.error is not None


# Callback types
QueryCallback = Callable[[QueryRequest], Awaitable[QueryResponse]]
"""Callback invoked when a query is received."""

StreamingCallback = Callable[[QueryRequest], Awaitable[None]]
"""Callback for streaming responses (chunks sent via response handler)."""


@runtime_checkable
class Trigger(Protocol):
    """Protocol for trigger implementations.

    Triggers are responsible for:
    1. Receiving queries from external sources (HTTP, Slack, etc.)
    2. Formatting requests into QueryRequest objects
    3. Invoking the RAG pipeline via callbacks
    4. Formatting and sending responses back

    Triggers should handle their own lifecycle (start/stop) and
    manage connections to external services.

    Implementations:
        - HTTPTrigger: FastAPI-based HTTP API
        - SlackTrigger (stub): Slack bot integration
        - TeamsTrigger (stub): Microsoft Teams integration
    """

    async def start(self) -> None:
        """Start the trigger and begin listening for queries.

        This method should:
        1. Initialize connections to external services
        2. Start listening for incoming requests
        3. Return once the trigger is ready to receive queries

        The trigger should run until stop() is called.

        Raises:
            TriggerError: If starting fails.

        Example:
            >>> await trigger.start()
            >>> print("Trigger is now listening")
        """
        ...

    async def stop(self) -> None:
        """Stop the trigger and clean up resources.

        This method should:
        1. Stop accepting new queries
        2. Wait for in-flight queries to complete (with timeout)
        3. Close connections and release resources

        Raises:
            TriggerError: If stopping fails.

        Example:
            >>> await trigger.stop()
            >>> print("Trigger stopped")
        """
        ...

    def on_query(self, callback: QueryCallback) -> None:
        """Register a callback for incoming queries.

        The callback will be invoked for each incoming query.
        Only one callback can be registered at a time.

        Args:
            callback: Async function that processes queries.

        Example:
            >>> async def handle_query(request: QueryRequest) -> QueryResponse:
            ...     answer = await rag_pipeline.query(request.query)
            ...     return QueryResponse(answer=answer, request_id=request.request_id)
            >>> trigger.on_query(handle_query)
        """
        ...

    def get_trigger_info(self) -> dict[str, Any]:
        """Get information about the trigger.

        Returns:
            Dictionary containing trigger information:
                - type: Trigger type (e.g., "http", "slack")
                - status: Current status ("running", "stopped")
                - endpoint: Endpoint/URL if applicable

        Example:
            >>> info = trigger.get_trigger_info()
            >>> print(f"Trigger: {info['type']} at {info.get('endpoint')}")
        """
        ...

    async def health_check(self) -> bool:
        """Check if the trigger is healthy.

        Returns:
            True if the trigger is operational, False otherwise.
        """
        ...


@runtime_checkable
class StreamingTrigger(Trigger, Protocol):
    """Extended protocol for triggers that support streaming responses.

    Adds support for streaming tokens as they're generated,
    which is useful for chat interfaces.
    """

    def on_query_streaming(
        self,
        callback: StreamingCallback,
        token_handler: "TokenHandler"
    ) -> None:
        """Register a callback for streaming query handling.

        Args:
            callback: Async function that processes queries.
            token_handler: Handler for streaming tokens back.

        Example:
            >>> async def handle_streaming(request: QueryRequest):
            ...     async for token in rag_pipeline.query_stream(request.query):
            ...         await token_handler.send(token)
            ...     await token_handler.complete()
            >>> trigger.on_query_streaming(handle_streaming, handler)
        """
        ...


class TokenHandler(Protocol):
    """Handler for streaming tokens back to the client."""

    async def send(self, token: str) -> None:
        """Send a token to the client."""
        ...

    async def complete(self) -> None:
        """Signal that the response is complete."""
        ...

    async def error(self, message: str) -> None:
        """Signal an error occurred."""
        ...
