"""Integration tests for the full chunking pipeline.

Tests the complete workflow from Document creation through chunking
to final Chunk output, verifying all components work together.
"""

from __future__ import annotations

from rag_pipeline.chunkers import (
    CodeChunker,
    GenericChunker,
    MarkdownChunker,
    create_chunker,
    get_chunker_for_document,
    get_supported_types,
)
from rag_pipeline.config import RAGSettings
from rag_pipeline.core.base_chunker import ChunkingConfig
from rag_pipeline.core.types import Chunk, Document, DocumentType, Metadata, SourceType

PYTHON_CODE = '''
"""Example Python module."""

import os
from pathlib import Path


def hello_world():
    """Print hello world."""
    print("Hello, World!")


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


class Calculator:
    """A simple calculator."""

    def __init__(self):
        self.result = 0

    def add(self, value: int) -> int:
        self.result += value
        return self.result

    def subtract(self, value: int) -> int:
        self.result -= value
        return self.result
'''


MARKDOWN_DOC = """# Project Documentation

This is the introduction to our project.

## Installation

Install the package using pip:

```bash
pip install mypackage
```

## Usage

### Basic Usage

Import and use the package:

```python
from mypackage import hello
hello()
```

### Advanced Usage

For advanced scenarios, configure options:

```python
from mypackage import configure
configure(debug=True)
```

## API Reference

### Functions

#### hello()

Prints a greeting message.

#### configure(debug=False)

Configures the package settings.
"""


PLAIN_TEXT = """
This is a sample text document that contains multiple paragraphs.
Each paragraph discusses different topics and should be chunked appropriately.

The second paragraph continues with more information about the subject matter.
It provides additional context and details that help explain the main points.

Finally, the third paragraph concludes the document with summary points.
This helps readers understand the key takeaways from the content.
"""


class TestEndToEndChunking:
    def test_python_code_end_to_end(self) -> None:
        document = Document(
            content=PYTHON_CODE,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/src/example.py",
                source_type=SourceType.FILESYSTEM,
                language="python",
                file_extension=".py",
            ),
        )

        chunker = get_chunker_for_document(document)
        chunks = chunker.chunk(document)

        assert len(chunks) >= 3
        assert all(isinstance(c, Chunk) for c in chunks)
        assert all(c.document_id == document.id for c in chunks)
        assert all(c.content.strip() for c in chunks)

        function_chunks = [c for c in chunks if "def " in c.content]
        assert len(function_chunks) >= 2

        class_chunks = [c for c in chunks if "class Calculator" in c.content]
        assert len(class_chunks) >= 1

    def test_markdown_end_to_end(self) -> None:
        document = Document(
            content=MARKDOWN_DOC,
            doc_type=DocumentType.MARKDOWN,
            metadata=Metadata(
                source_path="/docs/README.md",
                source_type=SourceType.FILESYSTEM,
                file_extension=".md",
            ),
        )

        chunker = get_chunker_for_document(document)
        chunks = chunker.chunk(document)

        assert len(chunks) >= 5
        assert all(isinstance(c, Chunk) for c in chunks)

        headers_found = set()
        for chunk in chunks:
            if "# Project Documentation" in chunk.content:
                headers_found.add("h1")
            if "## Installation" in chunk.content:
                headers_found.add("installation")
            if "## Usage" in chunk.content:
                headers_found.add("usage")
            if "## API Reference" in chunk.content:
                headers_found.add("api")

        assert len(headers_found) >= 3

    def test_plain_text_end_to_end(self) -> None:
        document = Document(
            content=PLAIN_TEXT,
            doc_type=DocumentType.TEXT,
            metadata=Metadata(
                source_path="/docs/notes.txt",
                source_type=SourceType.FILESYSTEM,
                file_extension=".txt",
            ),
        )

        chunker = get_chunker_for_document(document)
        chunks = chunker.chunk(document)

        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)

        combined = " ".join(c.content for c in chunks)
        assert "sample text document" in combined
        assert "multiple paragraphs" in combined


class TestFactoryIntegration:
    def test_factory_creates_correct_chunker_types(self) -> None:
        code_chunker = create_chunker(DocumentType.CODE)
        assert isinstance(code_chunker, CodeChunker)

        md_chunker = create_chunker(DocumentType.MARKDOWN)
        assert isinstance(md_chunker, MarkdownChunker)

        text_chunker = create_chunker(DocumentType.TEXT)
        assert isinstance(text_chunker, GenericChunker)

        runbook_chunker = create_chunker(DocumentType.RUNBOOK)
        assert isinstance(runbook_chunker, MarkdownChunker)

    def test_factory_with_string_type(self) -> None:
        code_chunker = create_chunker("code")
        assert isinstance(code_chunker, CodeChunker)

        md_chunker = create_chunker("markdown")
        assert isinstance(md_chunker, MarkdownChunker)

    def test_get_chunker_for_document_infers_from_extension(self) -> None:
        py_doc = Document(
            content="def foo(): pass",
            doc_type=DocumentType.UNKNOWN,
            metadata=Metadata(
                source_path="/src/test.py",
                file_extension=".py",
            ),
        )
        chunker = get_chunker_for_document(py_doc)
        assert isinstance(chunker, CodeChunker)

        md_doc = Document(
            content="# Title",
            doc_type=DocumentType.UNKNOWN,
            metadata=Metadata(
                source_path="/docs/readme.md",
                file_extension=".md",
            ),
        )
        chunker = get_chunker_for_document(md_doc)
        assert isinstance(chunker, MarkdownChunker)

    def test_all_supported_types_have_working_chunkers(self) -> None:
        supported = get_supported_types()

        for doc_type in supported:
            chunker = create_chunker(doc_type)
            assert chunker is not None
            assert hasattr(chunker, "chunk")
            assert hasattr(chunker, "get_supported_types")


class TestConfigIntegration:
    def test_settings_to_chunking_config(self) -> None:
        settings = RAGSettings(
            chunk_size=500,
            chunk_overlap=100,
            min_chunk_size=50,
            max_chunk_size=1000,
        )

        config = settings.to_chunking_config()

        assert isinstance(config, ChunkingConfig)
        assert config.chunk_size == 500
        assert config.chunk_overlap == 100
        assert config.min_chunk_size == 50
        assert config.max_chunk_size == 1000

    def test_settings_config_applies_to_chunker(self) -> None:
        settings = RAGSettings(chunk_size=250, chunk_overlap=50)
        config = settings.to_chunking_config()

        chunker = GenericChunker(config)

        assert chunker.config.chunk_size == 250
        assert chunker.config.chunk_overlap == 50

    def test_factory_accepts_settings_based_config(self) -> None:
        settings = RAGSettings(chunk_size=300, chunk_overlap=75)
        config = settings.to_chunking_config()

        chunker = create_chunker(DocumentType.TEXT, config)

        assert chunker.config.chunk_size == 300
        assert chunker.config.chunk_overlap == 75


class TestMetadataPreservation:
    def test_chunk_preserves_source_metadata(self) -> None:
        document = Document(
            content=PYTHON_CODE,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/project/src/module.py",
                source_type=SourceType.GIT,
                language="python",
                file_extension=".py",
                custom={"repo": "myrepo", "branch": "main"},
            ),
        )

        chunker = get_chunker_for_document(document)
        chunks = chunker.chunk(document)

        for chunk in chunks:
            assert chunk.metadata.source_path == "/project/src/module.py"
            assert chunk.metadata.source_type == SourceType.GIT
            assert chunk.metadata.custom.get("repo") == "myrepo"
            assert chunk.metadata.custom.get("branch") == "main"

    def test_chunk_adds_position_metadata(self) -> None:
        document = Document(
            content=MARKDOWN_DOC,
            doc_type=DocumentType.MARKDOWN,
            metadata=Metadata(source_path="/docs/readme.md"),
        )

        chunker = get_chunker_for_document(document)
        chunks = chunker.chunk(document)

        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
            assert "chunk_index" in chunk.metadata.custom
            assert "start_index" in chunk.metadata.custom
            assert "end_index" in chunk.metadata.custom


class TestMultiDocumentWorkflow:
    def test_batch_processing_multiple_document_types(self) -> None:
        documents = [
            Document(
                content=PYTHON_CODE,
                doc_type=DocumentType.CODE,
                metadata=Metadata(source_path="/src/code.py", language="python"),
            ),
            Document(
                content=MARKDOWN_DOC,
                doc_type=DocumentType.MARKDOWN,
                metadata=Metadata(source_path="/docs/readme.md"),
            ),
            Document(
                content=PLAIN_TEXT,
                doc_type=DocumentType.TEXT,
                metadata=Metadata(source_path="/notes/text.txt"),
            ),
        ]

        all_chunks = []
        for doc in documents:
            chunker = get_chunker_for_document(doc)
            chunks = chunker.chunk(doc)
            all_chunks.extend(chunks)

        assert len(all_chunks) >= 10

        doc_ids = {c.document_id for c in all_chunks}
        assert len(doc_ids) == 3

    def test_consistent_chunking_across_runs(self) -> None:
        document = Document(
            content=PYTHON_CODE,
            doc_type=DocumentType.CODE,
            metadata=Metadata(source_path="/src/code.py", language="python"),
        )

        chunker1 = get_chunker_for_document(document)
        chunks1 = chunker1.chunk(document)

        chunker2 = get_chunker_for_document(document)
        chunks2 = chunker2.chunk(document)

        assert len(chunks1) == len(chunks2)

        for c1, c2 in zip(chunks1, chunks2, strict=False):
            assert c1.id == c2.id
            assert c1.content == c2.content
            assert c1.start_index == c2.start_index
            assert c1.end_index == c2.end_index


class TestEdgeCasesIntegration:
    def test_empty_document_handling(self) -> None:
        document = Document(
            content="   \n\t  ",
            doc_type=DocumentType.TEXT,
            metadata=Metadata(source_path="/empty.txt"),
        )

        chunker = get_chunker_for_document(document)
        chunks = chunker.chunk(document)

        assert chunks == []

    def test_unicode_content_processing(self) -> None:
        unicode_content = """
# 日本語のドキュメント

これはテストです。

## セクション1

日本語のテキストを含むセクションです。

## セクション2 🎉

絵文字も含まれています！
"""  # noqa: RUF001
        document = Document(
            content=unicode_content,
            doc_type=DocumentType.MARKDOWN,
            metadata=Metadata(source_path="/docs/japanese.md"),
        )

        chunker = get_chunker_for_document(document)
        chunks = chunker.chunk(document)

        assert len(chunks) >= 1
        combined = "".join(c.content for c in chunks)
        assert "日本語" in combined
        assert "🎉" in combined

    def test_very_long_document(self) -> None:
        long_content = "# Main Title\n\n" + "\n\n".join(
            f"## Section {i}\n\nThis is content for section {i}. " * 10 for i in range(50)
        )

        document = Document(
            content=long_content,
            doc_type=DocumentType.MARKDOWN,
            metadata=Metadata(source_path="/docs/long.md"),
        )

        chunker = get_chunker_for_document(document)
        chunks = chunker.chunk(document)

        assert len(chunks) >= 20
        assert all(c.content for c in chunks)
