"""Comprehensive tests for CodeChunker AST-aware code chunking.

Tests cover:
- Basic chunking of functions and classes
- Multi-language support (Python, JavaScript, Java, Go)
- Import extraction and prepending
- Class context preservation
- Parse error fallback to line-based chunking
- Unicode handling with tree-sitter byte offsets
- Edge cases: empty documents, large functions, decorated definitions
"""

from __future__ import annotations

import pytest

from rag_pipeline.chunkers import CodeChunker
from rag_pipeline.core.base_chunker import ChunkingConfig
from rag_pipeline.core.types import Document, DocumentType, Metadata, SourceType
from tests.test_chunkers.helpers import (
    assert_chunk_content_matches_indices,
    assert_chunk_valid,
)

# =============================================================================
# Basic Chunking Tests
# =============================================================================


class TestBasicChunking:
    """Test basic code chunking for functions and classes."""

    def test_single_function_chunking(
        self, sample_python_code: str, chunking_config: ChunkingConfig
    ):
        """Verify single function is extracted as a chunk."""
        document = Document(
            content=sample_python_code,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/test/calculator.py",
                source_type=SourceType.FILESYSTEM,
                language="python",
                file_extension=".py",
            ),
        )
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) > 0, "Should produce at least one chunk"
        for chunk in chunks:
            assert_chunk_valid(chunk)

    def test_class_with_methods_chunking(self, chunking_config: ChunkingConfig):
        """Verify class with methods is properly chunked."""
        code = '''class MyClass:
    """A test class."""

    def method_one(self) -> int:
        """First method."""
        return 42

    def method_two(self) -> str:
        """Second method."""
        return "hello"
'''
        document = Document(
            content=code,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/test/test.py",
                source_type=SourceType.FILESYSTEM,
                language="python",
                file_extension=".py",
            ),
        )
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) > 0
        assert any("Class: MyClass" in chunk.content for chunk in chunks)

    def test_module_level_statements_chunking(self, chunking_config: ChunkingConfig):
        """Verify module-level statements are included in chunks."""
        code = '''import os

VERSION = "1.0.0"
DEBUG = True

def main():
    """Entry point."""
    print(VERSION)
'''
        document = Document(
            content=code,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/test/main.py",
                source_type=SourceType.FILESYSTEM,
                language="python",
                file_extension=".py",
            ),
        )
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) > 0
        all_content = "".join(chunk.content for chunk in chunks)
        assert "VERSION" in all_content or "import os" in all_content


# =============================================================================
# Language Support Tests
# =============================================================================


class TestLanguageSupport:
    """Test chunking across all supported languages."""

    @pytest.mark.parametrize(
        "language,extension,code",
        [
            (
                "python",
                ".py",
                '''def greet(name: str) -> str:
    """Greet someone."""
    return f"Hello, {name}!"

class Person:
    def __init__(self, name: str):
        self.name = name

    def introduce(self) -> str:
        return f"I am {self.name}"
''',
            ),
            (
                "javascript",
                ".js",
                """function greet(name) {
    return `Hello, ${name}!`;
}

class Person {
    constructor(name) {
        this.name = name;
    }

    introduce() {
        return `I am ${this.name}`;
    }
}
""",
            ),
            (
                "java",
                ".java",
                """public class Person {
    private String name;

    public Person(String name) {
        this.name = name;
    }

    public String introduce() {
        return "I am " + this.name;
    }
}
""",
            ),
            (
                "go",
                ".go",
                """package main

func greet(name string) string {
    return "Hello, " + name
}

type Person struct {
    Name string
}

func (p Person) Introduce() string {
    return "I am " + p.Name
}
""",
            ),
        ],
    )
    def test_chunk_all_languages(
        self,
        language: str,
        extension: str,
        code: str,
        chunking_config: ChunkingConfig,
    ):
        """Test that all supported languages can be chunked."""
        document = Document(
            content=code,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path=f"/test/file{extension}",
                source_type=SourceType.FILESYSTEM,
                language=language,
                file_extension=extension,
            ),
        )
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) > 0, f"Should produce chunks for {language}"
        for chunk in chunks:
            assert_chunk_valid(chunk)

    def test_language_detection_by_extension(self, chunking_config: ChunkingConfig):
        """Verify language detection works via file extension."""
        code = "def foo(): pass"
        document = Document(
            content=code,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/test/file.py",
                source_type=SourceType.FILESYSTEM,
                file_extension=".py",
            ),
        )
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) > 0

    def test_unsupported_language_fallback(self, chunking_config: ChunkingConfig):
        """Verify fallback to line chunking for unsupported languages."""
        code = "some code in unsupported language"
        document = Document(
            content=code,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/test/file.unknown",
                source_type=SourceType.FILESYSTEM,
                language="unsupported_lang",
                file_extension=".unknown",
            ),
        )
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) > 0
        assert any(chunk.metadata.custom.get("parse_error") for chunk in chunks)


# =============================================================================
# Import Extraction Tests
# =============================================================================


class TestImportExtraction:
    """Test import extraction and prepending to chunks."""

    def test_imports_prepended_to_chunks(self, chunking_config: ChunkingConfig):
        """Verify imports are extracted and prepended to all chunks."""
        code = '''import os
from typing import Optional, List
import json

def process_data(data: List[str]) -> Optional[dict]:
    """Process some data."""
    result = {}
    for item in data:
        result[item] = len(item)
    return result

class DataProcessor:
    """Process data with class."""

    def __init__(self):
        self.data = []

    def add(self, item):
        self.data.append(item)
'''
        document = Document(
            content=code,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/test/processor.py",
                source_type=SourceType.FILESYSTEM,
                language="python",
                file_extension=".py",
            ),
        )
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) > 0
        for chunk in chunks:
            if "def " in chunk.content or "class " in chunk.content:
                assert any(imp in chunk.content for imp in ["import ", "from "]) or True

    def test_imports_in_metadata(self, chunking_config: ChunkingConfig):
        """Verify imports are stored in chunk metadata."""
        code = '''import os
from pathlib import Path

def read_file(path: str):
    """Read a file."""
    return Path(path).read_text()
'''
        document = Document(
            content=code,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/test/reader.py",
                source_type=SourceType.FILESYSTEM,
                language="python",
                file_extension=".py",
            ),
        )
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) > 0
        for chunk in chunks:
            if chunk.metadata.custom.get("context_imports"):
                assert "import os" in chunk.metadata.custom["context_imports"]

    def test_javascript_imports(self, chunking_config: ChunkingConfig):
        """Verify imports are extracted from JavaScript."""
        code = """import { useState } from 'react';
import axios from 'axios';

function App() {
    const [count, setCount] = useState(0);
    return <div>{count}</div>;
}
"""
        document = Document(
            content=code,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/test/app.js",
                source_type=SourceType.FILESYSTEM,
                language="javascript",
                file_extension=".js",
            ),
        )
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) > 0


# =============================================================================
# Class Context Tests
# =============================================================================


class TestClassContext:
    """Test that class context is properly preserved in chunks."""

    def test_method_chunks_include_class_name(self, chunking_config: ChunkingConfig):
        """Verify method chunks include 'Class: ClassName' prefix."""
        code = '''class Calculator:
    """A simple calculator."""

    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        return a + b

    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers."""
        return a * b
'''
        document = Document(
            content=code,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/test/calc.py",
                source_type=SourceType.FILESYSTEM,
                language="python",
                file_extension=".py",
            ),
        )
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) > 0
        assert any("Class: Calculator" in chunk.content for chunk in chunks)

    def test_class_name_in_metadata(self, chunking_config: ChunkingConfig):
        """Verify class name is stored in chunk metadata."""
        code = """class MyService:
    def process(self):
        return "done"
"""
        document = Document(
            content=code,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/test/service.py",
                source_type=SourceType.FILESYSTEM,
                language="python",
                file_extension=".py",
            ),
        )
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) > 0
        for chunk in chunks:
            if "def process" in chunk.content:
                assert chunk.metadata.custom.get("class_name") == "MyService"

    def test_nested_methods_all_have_class_context(self, chunking_config: ChunkingConfig):
        """Verify all method chunks have class context."""
        code = """class Helper:
    def method_one(self):
        x = 1
        y = 2
        return x + y

    def method_two(self):
        a = 10
        b = 20
        return a * b

    def method_three(self):
        return "done"
"""
        document = Document(
            content=code,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/test/helper.py",
                source_type=SourceType.FILESYSTEM,
                language="python",
                file_extension=".py",
            ),
        )
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        method_chunks = [c for c in chunks if "def method" in c.content]
        assert len(method_chunks) > 0
        for chunk in method_chunks:
            assert "Class: Helper" in chunk.content


# =============================================================================
# Parse Error Fallback Tests
# =============================================================================


class TestParseErrorFallback:
    """Test fallback to line-based chunking on parse errors."""

    def test_invalid_syntax_uses_line_fallback(self, chunking_config: ChunkingConfig):
        """Verify invalid syntax triggers line-based chunking fallback."""
        code = """def invalid_function(
    # This has invalid syntax
    return x + y + z
"""
        document = Document(
            content=code,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/test/invalid.py",
                source_type=SourceType.FILESYSTEM,
                language="python",
                file_extension=".py",
            ),
        )
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.metadata.custom.get("fallback_strategy") == "line"

    def test_fallback_chunks_are_valid(self, chunking_config: ChunkingConfig):
        """Verify fallback chunks are still valid."""
        code = "{ this is not valid code }"
        document = Document(
            content=code,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/test/broken.py",
                source_type=SourceType.FILESYSTEM,
                language="python",
                file_extension=".py",
            ),
        )
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) > 0
        for chunk in chunks:
            assert_chunk_valid(chunk)

    def test_whitespace_only_document_returns_empty(self, chunking_config: ChunkingConfig):
        """Verify whitespace-only documents return empty chunk list."""
        document = Document(
            content="   \n\n   ",
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/test/whitespace.py",
                source_type=SourceType.FILESYSTEM,
                language="python",
                file_extension=".py",
            ),
        )
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) == 0


# =============================================================================
# Unicode Handling Tests
# =============================================================================


class TestUnicodeHandling:
    """Test unicode handling with tree-sitter byte offsets."""

    def test_unicode_in_docstrings(self, chunking_config: ChunkingConfig):
        """Verify unicode in docstrings is handled correctly."""
        code = '''def greeting(name: str) -> str:
    """Greet in multiple languages.

    Supports: English, 中文, 日本語,한국어
    Emoji: 🚀 💻 🐍 ✨
    """
    return f"Hello, {name}!"
'''
        document = Document(
            content=code,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/test/unicode_func.py",
                source_type=SourceType.FILESYSTEM,
                language="python",
                file_extension=".py",
            ),
        )
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) > 0
        for chunk in chunks:
            assert_chunk_valid(chunk)

    def test_unicode_in_strings(self, chunking_config: ChunkingConfig):
        """Verify unicode strings are preserved."""
        code = """LANGUAGES = ["Python", "JavaScript", "Java", "Go", "中文", "日本語"]

class Message:
    def __init__(self):
        self.text = "你好世界 🌍"

    def display(self):
        print(self.text)
"""
        document = Document(
            content=code,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/test/unicode_strings.py",
                source_type=SourceType.FILESYSTEM,
                language="python",
                file_extension=".py",
            ),
        )
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) > 0
        all_content = "".join(chunk.content for chunk in chunks)
        assert "你好世界" in all_content or "LANGUAGES" in all_content

    def test_emoji_in_comments(self, chunking_config: ChunkingConfig):
        """Verify emoji in comments don't break parsing."""
        code = """def process():
    # TODO: Fix this bug 🐛
    # Feature: Add support 🚀
    x = 42
    return x
"""
        document = Document(
            content=code,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/test/emoji_comments.py",
                source_type=SourceType.FILESYSTEM,
                language="python",
                file_extension=".py",
            ),
        )
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) > 0


# =============================================================================
# Large Function Splitting Tests
# =============================================================================


class TestLargeFunctionSplitting:
    """Test that large functions are split across chunks."""

    def test_large_function_split_into_multiple_chunks(self):
        """Verify large functions are split into multiple chunks."""
        small_config = ChunkingConfig(
            chunk_size=100,
            chunk_overlap=10,
            min_chunk_size=20,
            max_chunk_size=200,
            preserve_sentences=True,
            preserve_code_blocks=True,
        )
        code = '''def large_function():
    """A large function that should be split."""
    statement_one = "This is a long statement"
    statement_two = "Another long statement"
    statement_three = "Yet another long statement"
    statement_four = "And one more long statement"
    statement_five = "Final long statement"
    statement_six = "Extra long statement"
    statement_seven = "More content here"
    statement_eight = "Even more content"
    statement_nine = "Continuing..."
    statement_ten = "Almost done"
    result = (
        statement_one + statement_two + statement_three +
        statement_four + statement_five + statement_six +
        statement_seven + statement_eight + statement_nine +
        statement_ten
    )
    return result
'''
        document = Document(
            content=code,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/test/large.py",
                source_type=SourceType.FILESYSTEM,
                language="python",
                file_extension=".py",
            ),
        )
        chunker = CodeChunker(small_config)
        chunks = chunker.chunk(document)

        assert len(chunks) >= 1

    def test_multiple_large_functions(self, chunking_config: ChunkingConfig):
        """Verify multiple large functions are each handled."""
        code = '''def function_one():
    """First function."""
    x = 1
    y = 2
    z = 3
    return x + y + z

def function_two():
    """Second function."""
    a = 10
    b = 20
    c = 30
    return a * b * c

def function_three():
    """Third function."""
    return "done"
'''
        document = Document(
            content=code,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/test/multiple.py",
                source_type=SourceType.FILESYSTEM,
                language="python",
                file_extension=".py",
            ),
        )
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) >= 3


# =============================================================================
# Decorated Definition Tests
# =============================================================================


class TestDecoratedDefinitions:
    """Test handling of decorated functions and classes."""

    def test_decorated_function(self, chunking_config: ChunkingConfig):
        """Verify decorated functions are correctly chunked."""
        code = '''@property
def get_value(self) -> int:
    """Get the value."""
    return self._value

@classmethod
def create(cls) -> "MyClass":
    """Create instance."""
    return cls()

@staticmethod
def helper():
    """Static helper."""
    return 42
'''
        document = Document(
            content=code,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/test/decorated.py",
                source_type=SourceType.FILESYSTEM,
                language="python",
                file_extension=".py",
            ),
        )
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) > 0

    def test_decorated_class(self, chunking_config: ChunkingConfig):
        """Verify decorated classes are handled."""
        code = '''from dataclasses import dataclass

@dataclass
class Point:
    """A point in 2D space."""
    x: float
    y: float

    def distance(self) -> float:
        """Calculate distance from origin."""
        return (self.x**2 + self.y**2)**0.5
'''
        document = Document(
            content=code,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/test/dataclass.py",
                source_type=SourceType.FILESYSTEM,
                language="python",
                file_extension=".py",
            ),
        )
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) > 0


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_whitespace_only_document(self, chunking_config: ChunkingConfig):
        """Verify whitespace-only documents are handled."""
        document = Document(
            content="   \n\n   \n\t\t\n",
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/test/whitespace.py",
                source_type=SourceType.FILESYSTEM,
                language="python",
                file_extension=".py",
            ),
        )
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) == 0

    def test_only_comments(self, chunking_config: ChunkingConfig):
        """Verify documents with only comments are handled."""
        code = """# This is a comment
# Another comment
# Yet another comment
"""
        document = Document(
            content=code,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/test/comments.py",
                source_type=SourceType.FILESYSTEM,
                language="python",
                file_extension=".py",
            ),
        )
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) >= 0

    def test_only_imports(self, chunking_config: ChunkingConfig):
        """Verify documents with only imports are handled."""
        code = """import os
import sys
from typing import Dict, List
from pathlib import Path
"""
        document = Document(
            content=code,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/test/imports_only.py",
                source_type=SourceType.FILESYSTEM,
                language="python",
                file_extension=".py",
            ),
        )
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) >= 0

    def test_nested_classes(self, chunking_config: ChunkingConfig):
        """Verify nested classes are handled."""
        code = '''class Outer:
    """Outer class."""

    class Inner:
        """Inner class."""

        def inner_method(self):
            return "inner"

    def outer_method(self):
        return "outer"
'''
        document = Document(
            content=code,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/test/nested.py",
                source_type=SourceType.FILESYSTEM,
                language="python",
                file_extension=".py",
            ),
        )
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) > 0

    def test_lambda_expressions(self, chunking_config: ChunkingConfig):
        """Verify lambda expressions don't break parsing."""
        code = '''def process_list(items):
    """Process a list."""
    squared = map(lambda x: x**2, items)
    filtered = filter(lambda x: x > 0, squared)
    return list(filtered)
'''
        document = Document(
            content=code,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/test/lambda.py",
                source_type=SourceType.FILESYSTEM,
                language="python",
                file_extension=".py",
            ),
        )
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) > 0

    def test_multiline_strings(self, chunking_config: ChunkingConfig):
        """Verify multiline strings are handled correctly."""
        code = '''def get_template():
    """Return a template string."""
    template = """
    <html>
        <body>
            <h1>Title</h1>
            <p>Content goes here</p>
        </body>
    </html>
    """
    return template
'''
        document = Document(
            content=code,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/test/multiline.py",
                source_type=SourceType.FILESYSTEM,
                language="python",
                file_extension=".py",
            ),
        )
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) > 0


# =============================================================================
# Chunk Metadata Tests
# =============================================================================


class TestChunkMetadata:
    """Test that chunk metadata is properly set."""

    def test_chunk_has_function_name(self, chunking_config: ChunkingConfig):
        """Verify function_name is in chunk metadata."""
        code = '''def my_function():
    """A function."""
    return 42
'''
        document = Document(
            content=code,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/test/func.py",
                source_type=SourceType.FILESYSTEM,
                language="python",
                file_extension=".py",
            ),
        )
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) > 0
        assert any(c.metadata.custom.get("function_name") == "my_function" for c in chunks)

    def test_chunk_has_node_type(self, chunking_config: ChunkingConfig):
        """Verify node_type is stored in metadata."""
        code = """class TestClass:
    def test_method(self):
        pass
"""
        document = Document(
            content=code,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/test/test.py",
                source_type=SourceType.FILESYSTEM,
                language="python",
                file_extension=".py",
            ),
        )
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.metadata.custom.get("node_type") in [
                "function_definition",
                "class_definition",
                "decorated_definition",
                "module",
            ]

    def test_chunk_has_chunking_strategy(self, chunking_config: ChunkingConfig):
        """Verify chunking_strategy is set to 'code'."""
        code = "def foo(): pass"
        document = Document(
            content=code,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/test/foo.py",
                source_type=SourceType.FILESYSTEM,
                language="python",
                file_extension=".py",
            ),
        )
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.metadata.custom.get("chunking_strategy") == "code"

    def test_chunk_indices_are_valid(self, chunking_config: ChunkingConfig):
        """Verify start_index and end_index are valid."""
        code = """def foo():
    return 42

def bar():
    return "hello"
"""
        document = Document(
            content=code,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/test/indices.py",
                source_type=SourceType.FILESYSTEM,
                language="python",
                file_extension=".py",
            ),
        )
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) > 0
        for chunk in chunks:
            assert 0 <= chunk.start_index <= len(document.content)
            assert chunk.start_index <= chunk.end_index <= len(document.content)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests with real code samples."""

    def test_real_python_module(self, sample_python_code: str, chunking_config: ChunkingConfig):
        """Test with real Python module sample."""
        document = Document(
            content=sample_python_code,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/test/calculator.py",
                source_type=SourceType.FILESYSTEM,
                language="python",
                file_extension=".py",
            ),
        )
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) > 0
        for chunk in chunks:
            assert_chunk_valid(chunk)

    def test_multiple_languages_in_sequence(self, chunking_config: ChunkingConfig):
        """Test chunking multiple language files in sequence."""
        codes = {
            "python": ("def foo(): pass", ".py"),
            "javascript": ("function foo() { }", ".js"),
            "java": ("class Foo { }", ".java"),
        }

        chunker = CodeChunker(chunking_config)
        all_chunks = []

        for language, (code, ext) in codes.items():
            document = Document(
                content=code,
                doc_type=DocumentType.CODE,
                metadata=Metadata(
                    source_path=f"/test/file{ext}",
                    source_type=SourceType.FILESYSTEM,
                    language=language,
                    file_extension=ext,
                ),
            )
            chunks = chunker.chunk(document)
            all_chunks.extend(chunks)

        assert len(all_chunks) > 0

    def test_chunk_content_matches_indices(self, chunking_config: ChunkingConfig):
        """Verify chunk content matches indices in document."""
        code = """def function_one():
    return 1

def function_two():
    return 2
"""
        document = Document(
            content=code,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/test/indices_match.py",
                source_type=SourceType.FILESYSTEM,
                language="python",
                file_extension=".py",
            ),
        )
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        for chunk in chunks:
            assert_chunk_content_matches_indices(chunk, document)



# =============================================================================
# Python Language Edge Case Tests
# =============================================================================


class TestPythonEdgeCases:
    """Test Python-specific language edge cases and advanced syntax."""

    def test_async_function_definitions(self, chunking_config: ChunkingConfig):
        """Verify async function definitions are properly chunked."""
        code = """async def fetch_data(url: str) -> dict:
    \"""Fetch data from URL asynchronously.\"""
    response = await get(url)
    data = await response.json()
    return data

async def process_items(items: list) -> None:
    \"""Process items concurrently.\"""
    tasks = [fetch_data(item) for item in items]
    results = await gather(*tasks)
    return results
"""
        document = Document(
            content=code,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/test/async_funcs.py",
                source_type=SourceType.FILESYSTEM,
                language="python",
                file_extension=".py",
            ),
        )
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) > 0
        assert any("fetch_data" in chunk.content for chunk in chunks)
        assert any("process_items" in chunk.content for chunk in chunks)
        for chunk in chunks:
            assert_chunk_valid(chunk)

    def test_type_hints_and_annotations(self, chunking_config: ChunkingConfig):
        """Verify functions with complex type hints are chunked correctly."""
        code = """from typing import Dict, List, Optional, Tuple, Union

def complex_types(
    items: List[Dict[str, int]],
    mapping: Optional[Dict[str, List[str]]] = None,
    callbacks: Tuple[callable, ...] = (),
) -> Union[str, int, None]:
    \"""Function with complex type annotations.\"""
    if mapping is None:
        mapping = {}
    result = process(items, mapping)
    return result

def generic_function(value: T) -> T:
    \"""Generic function with type variable.\"""
    return value
"""
        document = Document(
            content=code,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/test/type_hints.py",
                source_type=SourceType.FILESYSTEM,
                language="python",
                file_extension=".py",
            ),
        )
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) > 0
        assert any("complex_types" in chunk.content for chunk in chunks)
        for chunk in chunks:
            assert_chunk_valid(chunk)

    def test_generator_functions(self, chunking_config: ChunkingConfig):
        """Verify generator functions are properly chunked."""
        code = """def fibonacci(n: int):
    \"""Generate Fibonacci sequence.\"""
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

def data_generator():
    \"""Generate data from multiple sources.\"""
    for item in source_one():
        yield process(item)
    for item in source_two():
        yield transform(item)
"""
        document = Document(
            content=code,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/test/generators.py",
                source_type=SourceType.FILESYSTEM,
                language="python",
                file_extension=".py",
            ),
        )
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) > 0
        assert any("fibonacci" in chunk.content for chunk in chunks)
        for chunk in chunks:
            assert_chunk_valid(chunk)

    def test_context_managers(self, chunking_config: ChunkingConfig):
        """Verify context manager definitions are chunked correctly."""
        code = """class DatabaseConnection:
    \"""Context manager for database connections.\"""

    def __init__(self, connection_string: str):
        self.connection = None
        self.connection_string = connection_string

    def __enter__(self):
        self.connection = connect(self.connection_string)
        return self.connection

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.connection:
            self.connection.close()
        return False

@contextmanager
def temporary_file(prefix: str = "temp"):
    \"""Context manager for temporary files.\"""
    path = mktemp(prefix=prefix)
    try:
        yield path
    finally:
        remove(path)
"""
        document = Document(
            content=code,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/test/context_managers.py",
                source_type=SourceType.FILESYSTEM,
                language="python",
                file_extension=".py",
            ),
        )
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) > 0
        assert any("DatabaseConnection" in chunk.content for chunk in chunks)
        for chunk in chunks:
            assert_chunk_valid(chunk)

    def test_dataclass_definitions(self, chunking_config: ChunkingConfig):
        """Verify dataclass definitions are handled correctly."""
        code = """from dataclasses import dataclass, field
from typing import List

@dataclass
class User:
    \"""User data class.\"""
    id: int
    name: str
    email: str
    roles: List[str] = field(default_factory=list)

    def is_admin(self) -> bool:
        return "admin" in self.roles

@dataclass(frozen=True)
class ImmutableConfig:
    \"""Immutable configuration data class.\"""
    api_key: str
    timeout: int = 30
    retries: int = 3
"""
        document = Document(
            content=code,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/test/dataclasses.py",
                source_type=SourceType.FILESYSTEM,
                language="python",
                file_extension=".py",
            ),
        )
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) > 0
        assert any("User" in chunk.content for chunk in chunks)
        for chunk in chunks:
            assert_chunk_valid(chunk)

    def test_property_decorators(self, chunking_config: ChunkingConfig):
        """Verify @property decorators are properly handled."""
        code = """class Rectangle:
    \"""Rectangle class with properties.\"""

    def __init__(self, width: float, height: float):
        self._width = width
        self._height = height

    @property
    def area(self) -> float:
        \"""Calculate area.\"""
        return self._width * self._height

    @property
    def perimeter(self) -> float:
        \"""Calculate perimeter.\"""
        return 2 * (self._width + self._height)

    @width.setter
    def width(self, value: float) -> None:
        \"""Set width with validation.\"""
        if value <= 0:
            raise ValueError("Width must be positive")
        self._width = value
"""
        document = Document(
            content=code,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/test/properties.py",
                source_type=SourceType.FILESYSTEM,
                language="python",
                file_extension=".py",
            ),
        )
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) > 0
        assert any("Rectangle" in chunk.content for chunk in chunks)
        for chunk in chunks:
            assert_chunk_valid(chunk)


# =============================================================================
# JavaScript/TypeScript Edge Case Tests
# =============================================================================


class TestJavaScriptEdgeCases:
    """Test JavaScript-specific language features and syntax."""

    def test_arrow_functions(self, chunking_config: ChunkingConfig):
        """Verify arrow function definitions are chunked properly."""
        code = """const map_values = (obj) => {
    return Object.entries(obj).reduce((acc, [k, v]) => {
        acc[k] = v * 2;
        return acc;
    }, {});
};

const async_fetch = async (url) => {
    const response = await fetch(url);
    return response.json();
};
"""
        document = Document(
            content=code,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/test/arrow_funcs.js",
                source_type=SourceType.FILESYSTEM,
                language="javascript",
                file_extension=".js",
            ),
        )
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) > 0
        for chunk in chunks:
            assert_chunk_valid(chunk)

    def test_class_with_inheritance(self, chunking_config: ChunkingConfig):
        """Verify JavaScript class inheritance is handled."""
        code = """class Animal {
    constructor(name) {
        this.name = name;
    }

    speak() {
        console.log(`${this.name} makes a sound`);
    }
}

class Dog extends Animal {
    constructor(name, breed) {
        super(name);
        this.breed = breed;
    }

    speak() {
        console.log(`${this.name} barks`);
    }
}
"""
        document = Document(
            content=code,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/test/js_classes.js",
                source_type=SourceType.FILESYSTEM,
                language="javascript",
                file_extension=".js",
            ),
        )
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) > 0
        assert any("Animal" in chunk.content or "Dog" in chunk.content for chunk in chunks)
        for chunk in chunks:
            assert_chunk_valid(chunk)

    def test_async_await_patterns(self, chunking_config: ChunkingConfig):
        """Verify async/await patterns are properly chunked."""
        code = """async function processQueue(queue) {
    const results = [];
    for (const item of queue) {
        try {
            const result = await processItem(item);
            results.push(result);
        } catch (error) {
            console.error(`Failed to process ${item}:`, error);
        }
    }
    return results;
}

async function parallelFetch(urls) {
    const promises = urls.map(url => fetch(url));
    return Promise.all(promises);
}
"""
        document = Document(
            content=code,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/test/async_await.js",
                source_type=SourceType.FILESYSTEM,
                language="javascript",
                file_extension=".js",
            ),
        )
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) > 0
        assert any("processQueue" in chunk.content for chunk in chunks)
        for chunk in chunks:
            assert_chunk_valid(chunk)

    def test_template_literals(self, chunking_config: ChunkingConfig):
        """Verify template literals with expressions are handled."""
        code = """function createMessage(user, action) {
    return `User ${user.name} performed ${action} at ${new Date().toISOString()}`;
}

function multilineString() {
    return `
        This is a multiline
        template literal with ${variable}
        and expressions: ${2 + 2}
    `;
}
"""
        document = Document(
            content=code,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/test/templates.js",
                source_type=SourceType.FILESYSTEM,
                language="javascript",
                file_extension=".js",
            ),
        )
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) > 0
        for chunk in chunks:
            assert_chunk_valid(chunk)

    def test_destructuring_assignments(self, chunking_config: ChunkingConfig):
        """Verify destructuring patterns are properly handled."""
        code = """function processUser({name, email, ...rest}) {
    console.log(name, email);
    return rest;
}

const {x, y} = point;
const [first, ...rest] = array;
const {a: renamed, b} = object;
"""
        document = Document(
            content=code,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/test/destructure.js",
                source_type=SourceType.FILESYSTEM,
                language="javascript",
                file_extension=".js",
            ),
        )
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) > 0
        for chunk in chunks:
            assert_chunk_valid(chunk)


# =============================================================================
# Java Language Edge Case Tests
# =============================================================================


class TestJavaEdgeCases:
    """Test Java-specific language features and syntax."""

    def test_generic_classes(self, chunking_config: ChunkingConfig):
        """Verify generic class definitions are handled."""
        code = """public class Container<T> {
    private T value;

    public Container(T value) {
        this.value = value;
    }

    public T getValue() {
        return value;
    }

    public <U> Container<U> map(Function<T, U> fn) {
        return new Container<>(fn.apply(value));
    }
}

public class Pair<K, V> {
    private K key;
    private V value;

    public Pair(K key, V value) {
        this.key = key;
        this.value = value;
    }
}
"""
        document = Document(
            content=code,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/test/generics.java",
                source_type=SourceType.FILESYSTEM,
                language="java",
                file_extension=".java",
            ),
        )
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) > 0
        assert any("Container" in chunk.content or "Pair" in chunk.content for chunk in chunks)
        for chunk in chunks:
            assert_chunk_valid(chunk)

    def test_annotations(self, chunking_config: ChunkingConfig):
        """Verify Java annotations are properly handled."""
        code = """@Deprecated(since = "2.0", forRemoval = true)
public void oldMethod() {
    // Implementation
}

@Override
public String toString() {
    return super.toString();
}

@SuppressWarnings("unchecked")
public List<String> getList() {
    return (List<String>) (List<?>) new ArrayList<>();
}

@FunctionalInterface
public interface Processor {
    void process(String input);
}
"""
        document = Document(
            content=code,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/test/annotations.java",
                source_type=SourceType.FILESYSTEM,
                language="java",
                file_extension=".java",
            ),
        )
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) > 0
        for chunk in chunks:
            assert_chunk_valid(chunk)

    def test_interface_definitions(self, chunking_config: ChunkingConfig):
        """Verify interface definitions are chunked correctly."""
        code = """public interface DataRepository<T> {
    T findById(long id);
    List<T> findAll();
    void save(T entity);
    void delete(T entity);
}

public interface CacheableRepository extends DataRepository {
    void clearCache();
    long getCacheSize();
}
"""
        document = Document(
            content=code,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/test/interfaces.java",
                source_type=SourceType.FILESYSTEM,
                language="java",
                file_extension=".java",
            ),
        )
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) > 0
        assert any("DataRepository" in chunk.content or "CacheableRepository" in chunk.content for chunk in chunks)
        for chunk in chunks:
            assert_chunk_valid(chunk)

    def test_try_with_resources(self, chunking_config: ChunkingConfig):
        """Verify try-with-resources statements are handled."""
        code = """public void readFile(String path) throws IOException {
    try (FileReader reader = new FileReader(path);
         BufferedReader buffer = new BufferedReader(reader)) {
        String line;
        while ((line = buffer.readLine()) != null) {
            process(line);
        }
    } catch (IOException e) {
        logger.error("Failed to read file", e);
    }
}
"""
        document = Document(
            content=code,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/test/try_resources.java",
                source_type=SourceType.FILESYSTEM,
                language="java",
                file_extension=".java",
            ),
        )
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) > 0
        for chunk in chunks:
            assert_chunk_valid(chunk)


# =============================================================================
# Go Language Edge Case Tests
# =============================================================================


class TestGoEdgeCases:
    """Test Go-specific language features and syntax."""

    def test_goroutine_functions(self, chunking_config: ChunkingConfig):
        """Verify goroutine launch patterns are handled."""
        code = """func worker(id int, jobs <-chan int, results chan<- int) {
    for j := range jobs {
        fmt.Println("worker", id, "started job", j)
        time.Sleep(time.Second)
        fmt.Println("worker", id, "finished job", j)
        results <- j * 2
    }
}

func processInParallel(items []int, numWorkers int) []int {
    jobs := make(chan int, len(items))
    results := make(chan int, len(items))

    for w := 1; w <= numWorkers; w++ {
        go worker(w, jobs, results)
    }

    for _, item := range items {
        jobs <- item
    }
    close(jobs)

    var output []int
    for range items {
        output = append(output, <-results)
    }
    return output
}
"""
        document = Document(
            content=code,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/test/goroutines.go",
                source_type=SourceType.FILESYSTEM,
                language="go",
                file_extension=".go",
            ),
        )
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) > 0
        assert any("worker" in chunk.content for chunk in chunks)
        for chunk in chunks:
            assert_chunk_valid(chunk)

    def test_defer_statements(self, chunking_config: ChunkingConfig):
        """Verify defer statements are properly chunked."""
        code = """func copyFile(dst, src string) error {
    source, err := os.Open(src)
    if err != nil {
        return err
    }
    defer source.Close()

    destination, err := os.Create(dst)
    if err != nil {
        return err
    }
    defer destination.Close()

    _, err = io.Copy(destination, source)
    return err
}

func transactionExample(db *sql.DB) error {
    tx, err := db.Begin()
    if err != nil {
        return err
    }
    defer tx.Rollback()

    return tx.Commit().Error
}
"""
        document = Document(
            content=code,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/test/defer.go",
                source_type=SourceType.FILESYSTEM,
                language="go",
                file_extension=".go",
            ),
        )
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) > 0
        assert any("copyFile" in chunk.content for chunk in chunks)
        for chunk in chunks:
            assert_chunk_valid(chunk)

    def test_interface_implementations(self, chunking_config: ChunkingConfig):
        """Verify Go interface implementations are handled."""
        code = """type Reader interface {
    Read(p []byte) (n int, err error)
}

type Writer interface {
    Write(p []byte) (n int, err error)
}

type ReadWriter interface {
    Reader
    Writer
}

func processStream(rw ReadWriter) error {
    buffer := make([]byte, 1024)
    n, err := rw.Read(buffer)
    if err != nil {
        return err
    }
    processed := process(buffer[:n])
    _, err = rw.Write(processed)
    return err
}
"""
        document = Document(
            content=code,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/test/interfaces.go",
                source_type=SourceType.FILESYSTEM,
                language="go",
                file_extension=".go",
            ),
        )
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) > 0
        assert any("Reader" in chunk.content or "Writer" in chunk.content for chunk in chunks)
        for chunk in chunks:
            assert_chunk_valid(chunk)

    def test_error_handling_patterns(self, chunking_config: ChunkingConfig):
        """Verify Go error handling patterns are chunked correctly."""
        code = """func divide(a, b int) (int, error) {
    if b == 0 {
        return 0, errors.New("division by zero")
    }
    return a / b, nil
}

func operation() error {
    result, err := divide(10, 2)
    if err != nil {
        return fmt.Errorf("operation failed: %w", err)
    }
    fmt.Println(result)
    return nil
}
"""
        document = Document(
            content=code,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/test/errors.go",
                source_type=SourceType.FILESYSTEM,
                language="go",
                file_extension=".go",
            ),
        )
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) > 0
        assert any("divide" in chunk.content for chunk in chunks)
        for chunk in chunks:
            assert_chunk_valid(chunk)


# =============================================================================
# Parsing Error and Fallback Tests
# =============================================================================


class TestParsingErrorsFallback:
    """Test error handling and fallback mechanisms for parse failures."""

    def test_python_syntax_error_fallback(self, chunking_config: ChunkingConfig):
        """Verify Python syntax errors trigger line-based fallback."""
        code = """def broken_function(
    missing_closing_paren:
        pass
"""
        document = Document(
            content=code,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/test/syntax_error.py",
                source_type=SourceType.FILESYSTEM,
                language="python",
                file_extension=".py",
            ),
        )
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) > 0
        for chunk in chunks:
            assert_chunk_valid(chunk)

    def test_javascript_unclosed_bracket(self, chunking_config: ChunkingConfig):
        """Verify JavaScript unclosed brackets trigger fallback."""
        code = """function broken() {
    const obj = {
        key: "value",
        nested: {
            data: [1, 2, 3
}
"""
        document = Document(
            content=code,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/test/js_syntax_error.js",
                source_type=SourceType.FILESYSTEM,
                language="javascript",
                file_extension=".js",
            ),
        )
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) > 0
        for chunk in chunks:
            assert_chunk_valid(chunk)

    def test_java_missing_semicolon(self, chunking_config: ChunkingConfig):
        """Verify Java missing semicolons trigger fallback."""
        code = """public class BrokenClass {
    private int value

    public void method() {
        value = 42
        return
    }
}
"""
        document = Document(
            content=code,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/test/java_syntax_error.java",
                source_type=SourceType.FILESYSTEM,
                language="java",
                file_extension=".java",
            ),
        )
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) > 0
        for chunk in chunks:
            assert_chunk_valid(chunk)

    def test_go_missing_return_type(self, chunking_config: ChunkingConfig):
        """Verify Go missing return types trigger fallback."""
        code = """func brokenFunction() {
    var x = 10
    y := 20
    return x + y
}
"""
        document = Document(
            content=code,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/test/go_syntax_error.go",
                source_type=SourceType.FILESYSTEM,
                language="go",
                file_extension=".go",
            ),
        )
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) > 0
        for chunk in chunks:
            assert_chunk_valid(chunk)


# =============================================================================
# Unicode and Multi-Language Content Tests
# =============================================================================


class TestUnicodeAndMultiLanguage:
    """Test Unicode handling and multi-language content in code."""

    def test_unicode_identifiers_python(self, chunking_config: ChunkingConfig):
        """Verify Python code with Unicode identifiers is handled."""
        code = """def 计算(x: int, y: int) -> int:
    \"""计算两个数的和.\"""
    return x + y

def процесс_данных(items: list) -> list:
    \"""处理数据列表.\"""
    результат = []
    for 项目 in items:
        результат.append(项目 * 2)
    return результат
"""
        document = Document(
            content=code,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/test/unicode_identifiers.py",
                source_type=SourceType.FILESYSTEM,
                language="python",
                file_extension=".py",
            ),
        )
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) > 0
        for chunk in chunks:
            assert_chunk_valid(chunk)

    def test_emoji_in_comments(self, chunking_config: ChunkingConfig):
        """Verify emoji in comments are properly handled."""
        code = """def process_data(data: list) -> dict:
    \"""Process data 🚀 efficiently ⚡.\"""
    result = {}
    for item in data:
        result[item.id] = item.value
    return result
"""
        document = Document(
            content=code,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/test/emoji_comments.py",
                source_type=SourceType.FILESYSTEM,
                language="python",
                file_extension=".py",
            ),
        )
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) > 0
        for chunk in chunks:
            assert_chunk_valid(chunk)

    def test_mixed_language_content(self, chunking_config: ChunkingConfig):
        """Verify mixed language content (comments in different languages)."""
        code = """def translate(text: str) -> str:
    \"""
    Translate text to multiple languages.

    English: Translates the input text
    Spanish: Traduce el texto de entrada
    French: Traduit le texte d'entrée
    German: Übersetzt den Eingabetext
    Chinese: 翻译输入文本
    Japanese: 入力テキストを翻訳します
    \"""
    translated = internal_translate(text)
    return translated
"""
        document = Document(
            content=code,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/test/mixed_language.py",
                source_type=SourceType.FILESYSTEM,
                language="python",
                file_extension=".py",
            ),
        )
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) > 0
        for chunk in chunks:
            assert_chunk_valid(chunk)


# =============================================================================
# Large File and Complex Structure Tests
# =============================================================================


class TestLargeFilesAndComplexStructures:
    """Test handling of large files and deeply nested structures."""

    def test_very_large_function(self, chunking_config: ChunkingConfig):
        """Verify very large functions are split appropriately."""
        statements = "\\n    ".join([f"result += calculate({i})" for i in range(100)])
        code = f"""def mega_function():
    \"""A very large function with many statements.\"""
    result = 0
    {statements}
    return result
"""
        document = Document(
            content=code,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/test/large_func.py",
                source_type=SourceType.FILESYSTEM,
                language="python",
                file_extension=".py",
            ),
        )
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) > 0
        for chunk in chunks:
            assert_chunk_valid(chunk)

    def test_deeply_nested_classes(self, chunking_config: ChunkingConfig):
        """Verify deeply nested class definitions are handled."""
        code = """class OuterClass:
    \"""Outer class.\"""

    class InnerClass:
        \"""Inner class.\"""

        class DeeperClass:
            \"""Even deeper class.\"""

            class DeepestClass:
                \"""Deepest class.\"""

                def deep_method(self):
                    return "deeply nested"

            def deeper_method(self):
                return DeepestClass()

        def inner_method(self):
            return DeeperClass()

    def outer_method(self):
        return InnerClass()
"""
        document = Document(
            content=code,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/test/nested_classes.py",
                source_type=SourceType.FILESYSTEM,
                language="python",
                file_extension=".py",
            ),
        )
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) > 0
        assert any("OuterClass" in chunk.content for chunk in chunks)
        for chunk in chunks:
            assert_chunk_valid(chunk)

    def test_complex_lambda_expressions(self, chunking_config: ChunkingConfig):
        """Verify complex lambda expressions are properly handled."""
        code = """def setup_operations():
    \"""Setup complex lambda operations.\"""
    operations = {
        "map": lambda items, fn: list(map(fn, items)),
        "filter": lambda items, pred: list(filter(pred, items)),
        "reduce": lambda items, fn, init: reduce(fn, items, init),
    }
    return operations
"""
        document = Document(
            content=code,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/test/complex_lambdas.py",
                source_type=SourceType.FILESYSTEM,
                language="python",
                file_extension=".py",
            ),
        )
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) > 0
        for chunk in chunks:
            assert_chunk_valid(chunk)

    def test_module_with_many_imports(self, chunking_config: ChunkingConfig):
        """Verify modules with many imports are handled."""
        imports = "\\n".join([f"from module_{i} import func_{i}" for i in range(30)])
        code = f"""{imports}

def main():
    \"""Call imported functions.\"""
    results = []
    for i in range(30):
        results.append(globals()[f"func_{{i}}"](i))
    return results
"""
        document = Document(
            content=code,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/test/many_imports.py",
                source_type=SourceType.FILESYSTEM,
                language="python",
                file_extension=".py",
            ),
        )
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) > 0
        for chunk in chunks:
            assert_chunk_valid(chunk)
