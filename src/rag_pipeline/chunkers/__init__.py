"""Chunkers module for content-aware document chunking.

This module provides chunker implementations for splitting documents into
semantically meaningful chunks for embedding and retrieval. Each chunker
is specialized for a specific content type:

- CodeChunker: AST-aware chunking for code using tree-sitter
- MarkdownChunker: Header-based hierarchical chunking for markdown
- GenericChunker: Recursive text splitting for plain text

The factory functions provide automatic chunker selection based on document type.

Example usage:
    >>> from rag_pipeline.chunkers import (
    ...     create_chunker,
    ...     get_chunker_for_document,
    ...     CodeChunker,
    ...     MarkdownChunker,
    ...     GenericChunker,
    ... )
    >>> from rag_pipeline.core.types import Document, DocumentType
    >>>
    >>> # Use factory for automatic selection
    >>> doc = Document(content="def hello(): pass", doc_type=DocumentType.CODE)
    >>> chunker = get_chunker_for_document(doc)
    >>> chunks = chunker.chunk(doc)
    >>>
    >>> # Or instantiate directly
    >>> from rag_pipeline.core.base_chunker import ChunkingConfig
    >>> config = ChunkingConfig(chunk_size=500, chunk_overlap=50)
    >>> code_chunker = CodeChunker(config)
"""

from rag_pipeline.chunkers.base import (
    AbstractChunker,
    byte_offset_to_char_offset,
    calculate_line_numbers,
    char_offset_to_byte_offset,
)
from rag_pipeline.chunkers.code_chunker import CodeChunker
from rag_pipeline.chunkers.factory import (
    CHUNKER_REGISTRY,
    EXTENSION_TYPE_MAP,
    create_chunker,
    get_chunker_class,
    get_chunker_for_document,
    get_supported_types,
    register_chunker,
)
from rag_pipeline.chunkers.generic_chunker import GenericChunker
from rag_pipeline.chunkers.markdown_chunker import MarkdownChunker

__all__ = [
    # Registry for extension
    "CHUNKER_REGISTRY",
    "EXTENSION_TYPE_MAP",
    # Base class
    "AbstractChunker",
    # Concrete implementations
    "CodeChunker",
    "GenericChunker",
    "MarkdownChunker",
    # Utility functions
    "byte_offset_to_char_offset",
    "calculate_line_numbers",
    "char_offset_to_byte_offset",
    # Factory functions
    "create_chunker",
    "get_chunker_class",
    "get_chunker_for_document",
    "get_supported_types",
    "register_chunker",
]
