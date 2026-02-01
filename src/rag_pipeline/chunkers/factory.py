"""Chunker factory for automatic chunker selection based on document type.

This module provides factory functions to instantiate the appropriate chunker
for a given document or document type. It supports automatic selection based on
DocumentType or file extension, with fallback to GenericChunker for unknown types.

Example usage:
    >>> from rag_pipeline.chunkers.factory import create_chunker, get_chunker_for_document
    >>> from rag_pipeline.core.types import Document, DocumentType
    >>>
    >>> # Get chunker by document type
    >>> chunker = create_chunker(DocumentType.CODE)
    >>>
    >>> # Get chunker for a specific document (auto-detect)
    >>> doc = Document(content="...", doc_type=DocumentType.MARKDOWN)
    >>> chunker = get_chunker_for_document(doc)
    >>> chunks = chunker.chunk(doc)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag_pipeline.chunkers.base import AbstractChunker
from rag_pipeline.chunkers.code_chunker import CodeChunker
from rag_pipeline.chunkers.generic_chunker import GenericChunker
from rag_pipeline.chunkers.markdown_chunker import MarkdownChunker
from rag_pipeline.core.types import DocumentType

if TYPE_CHECKING:
    from rag_pipeline.core.base_chunker import ChunkingConfig
    from rag_pipeline.core.types import Document


# Mapping of DocumentType to chunker classes
CHUNKER_REGISTRY: dict[DocumentType, type[AbstractChunker]] = {
    DocumentType.CODE: CodeChunker,
    DocumentType.MARKDOWN: MarkdownChunker,
    DocumentType.RUNBOOK: MarkdownChunker,  # Runbooks are typically markdown
    DocumentType.TEXT: GenericChunker,
    DocumentType.UNKNOWN: GenericChunker,
}

# File extension to DocumentType mapping for auto-detection
EXTENSION_TYPE_MAP: dict[str, DocumentType] = {
    # Code files
    ".py": DocumentType.CODE,
    ".js": DocumentType.CODE,
    ".jsx": DocumentType.CODE,
    ".ts": DocumentType.CODE,
    ".tsx": DocumentType.CODE,
    ".java": DocumentType.CODE,
    ".go": DocumentType.CODE,
    ".rs": DocumentType.CODE,
    ".c": DocumentType.CODE,
    ".cpp": DocumentType.CODE,
    ".h": DocumentType.CODE,
    ".hpp": DocumentType.CODE,
    ".cs": DocumentType.CODE,
    ".rb": DocumentType.CODE,
    ".php": DocumentType.CODE,
    ".swift": DocumentType.CODE,
    ".kt": DocumentType.CODE,
    ".scala": DocumentType.CODE,
    # Markdown files
    ".md": DocumentType.MARKDOWN,
    ".markdown": DocumentType.MARKDOWN,
    ".mdx": DocumentType.MARKDOWN,
    # Text files
    ".txt": DocumentType.TEXT,
    ".text": DocumentType.TEXT,
    ".log": DocumentType.TEXT,
    ".csv": DocumentType.TEXT,
}


def create_chunker(
    doc_type: DocumentType | str,
    config: ChunkingConfig | None = None,
) -> AbstractChunker:
    """Create a chunker instance for the given document type.

    Instantiates the appropriate chunker class based on the document type.
    Falls back to GenericChunker for unknown or unsupported types.

    Args:
        doc_type: The document type (DocumentType enum or string value).
        config: Optional chunking configuration. Uses defaults if not provided.

    Returns:
        An AbstractChunker instance appropriate for the document type.

    Example:
        >>> chunker = create_chunker(DocumentType.CODE)
        >>> chunks = chunker.chunk(code_document)
    """
    # Convert string to DocumentType if needed
    if isinstance(doc_type, str):
        try:
            doc_type = DocumentType(doc_type)
        except ValueError:
            doc_type = DocumentType.UNKNOWN

    # Get the chunker class from registry
    chunker_class = CHUNKER_REGISTRY.get(doc_type, GenericChunker)

    return chunker_class(config)


def get_chunker_for_document(
    document: Document,
    config: ChunkingConfig | None = None,
) -> AbstractChunker:
    """Get the appropriate chunker for a specific document.

    Determines the best chunker based on the document's type. If the document
    type is UNKNOWN, attempts to infer from the file extension in metadata.

    Args:
        document: The document to get a chunker for.
        config: Optional chunking configuration. Uses defaults if not provided.

    Returns:
        An AbstractChunker instance appropriate for the document.

    Example:
        >>> doc = Document(content="...", doc_type=DocumentType.MARKDOWN)
        >>> chunker = get_chunker_for_document(doc)
        >>> chunks = chunker.chunk(doc)
    """
    doc_type = document.doc_type

    # If type is unknown, try to infer from file extension
    if doc_type == DocumentType.UNKNOWN and document.metadata.file_extension:
        extension = document.metadata.file_extension.lower()
        if not extension.startswith("."):
            extension = f".{extension}"
        doc_type = EXTENSION_TYPE_MAP.get(extension, DocumentType.UNKNOWN)

    return create_chunker(doc_type, config)


def register_chunker(
    doc_type: DocumentType,
    chunker_class: type[AbstractChunker],
) -> None:
    """Register a custom chunker class for a document type.

    Allows extending the factory with custom chunker implementations.
    The registered chunker will be used for the specified document type.

    Args:
        doc_type: The document type to register the chunker for.
        chunker_class: The chunker class to register. Must be a subclass of AbstractChunker.

    Raises:
        TypeError: If chunker_class is not a subclass of AbstractChunker.

    Example:
        >>> class CustomChunker(AbstractChunker):
        ...     def chunk(self, document: Document) -> list[Chunk]:
        ...         # Custom implementation
        ...         pass
        ...     def get_supported_types(self) -> list[str]:
        ...         return ["custom"]
        >>>
        >>> register_chunker(DocumentType.TEXT, CustomChunker)
    """
    if not issubclass(chunker_class, AbstractChunker):
        raise TypeError(f"chunker_class must be a subclass of AbstractChunker, got {chunker_class}")
    CHUNKER_REGISTRY[doc_type] = chunker_class


def get_supported_types() -> list[DocumentType]:
    """Get all document types that have registered chunkers.

    Returns:
        List of DocumentType values with registered chunkers.
    """
    return list(CHUNKER_REGISTRY.keys())


def get_chunker_class(doc_type: DocumentType) -> type[AbstractChunker]:
    """Get the chunker class registered for a document type.

    Args:
        doc_type: The document type to look up.

    Returns:
        The chunker class registered for this type, or GenericChunker if not found.
    """
    return CHUNKER_REGISTRY.get(doc_type, GenericChunker)
