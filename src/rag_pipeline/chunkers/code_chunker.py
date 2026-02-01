"""Code chunker implementation using tree-sitter for AST-aware splitting.

This module provides the CodeChunker class that parses source code with
tree-sitter and extracts chunks for:
- Functions
- Classes
- Module-level statements

The chunker preserves important context by prepending imports and class names
to chunk content, while also storing that context in metadata.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, TypedDict

import tree_sitter_go as tsgo
import tree_sitter_java as tsjava
import tree_sitter_javascript as tsjavascript
import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser, Tree

from rag_pipeline.chunkers.base import AbstractChunker, byte_offset_to_char_offset

if TYPE_CHECKING:
    from rag_pipeline.core.base_chunker import ChunkingConfig
    from rag_pipeline.core.types import Chunk, Document


class LanguageConfig(TypedDict):
    """Configuration for a tree-sitter language."""

    language: Language
    function_types: set[str]
    class_types: set[str]
    import_types: set[str]
    statement_types: set[str]


LANGUAGE_CONFIG: dict[str, LanguageConfig] = {
    "python": {
        "language": Language(tspython.language()),
        "function_types": {
            "function_definition",
            "async_function_definition",
            "decorated_definition",
        },
        "class_types": {"class_definition", "decorated_definition"},
        "import_types": {"import_statement", "import_from_statement"},
        "statement_types": {
            "assignment",
            "augmented_assignment",
            "return_statement",
            "expression_statement",
            "if_statement",
            "for_statement",
            "while_statement",
            "with_statement",
            "try_statement",
            "raise_statement",
            "assert_statement",
            "pass_statement",
            "break_statement",
            "continue_statement",
            "global_statement",
            "nonlocal_statement",
            "import_statement",
            "import_from_statement",
            "function_definition",
            "async_function_definition",
            "class_definition",
            "decorated_definition",
        },
    },
    "javascript": {
        "language": Language(tsjavascript.language()),
        "function_types": {"function_declaration", "generator_function_declaration"},
        "class_types": {"class_declaration"},
        "import_types": {"import_statement"},
        "statement_types": {
            "expression_statement",
            "return_statement",
            "if_statement",
            "for_statement",
            "for_in_statement",
            "for_of_statement",
            "while_statement",
            "do_statement",
            "switch_statement",
            "try_statement",
            "throw_statement",
            "break_statement",
            "continue_statement",
            "variable_declaration",
            "lexical_declaration",
            "function_declaration",
            "class_declaration",
            "method_definition",
        },
    },
    "java": {
        "language": Language(tsjava.language()),
        "function_types": {"method_declaration", "constructor_declaration"},
        "class_types": {"class_declaration", "interface_declaration", "enum_declaration"},
        "import_types": {"import_declaration"},
        "statement_types": {
            "expression_statement",
            "return_statement",
            "if_statement",
            "for_statement",
            "enhanced_for_statement",
            "while_statement",
            "do_statement",
            "switch_statement",
            "try_statement",
            "throw_statement",
            "break_statement",
            "continue_statement",
            "local_variable_declaration",
            "method_declaration",
            "constructor_declaration",
            "field_declaration",
            "class_declaration",
            "interface_declaration",
            "enum_declaration",
        },
    },
    "go": {
        "language": Language(tsgo.language()),
        "function_types": {"function_declaration", "method_declaration"},
        "class_types": {"type_declaration"},
        "import_types": {"import_declaration"},
        "statement_types": {
            "declaration",
            "expression_statement",
            "return_statement",
            "if_statement",
            "for_statement",
            "switch_statement",
            "type_switch_statement",
            "assignment_statement",
            "short_var_declaration",
            "go_statement",
            "defer_statement",
            "function_declaration",
            "method_declaration",
            "type_declaration",
        },
    },
}


FILE_EXTENSION_LANGUAGE_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".java": "java",
    ".go": "go",
}


class CodeChunker(AbstractChunker):
    """AST-aware code chunker using tree-sitter."""

    DEFAULT_CONTEXT_SEPARATOR: ClassVar[str] = "\n\n"

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        """Initialize the code chunker with configuration."""
        super().__init__(config)
        self._parsers: dict[str, Parser] = {}

    def get_supported_types(self) -> list[str]:
        """Get the document types this chunker supports."""
        return ["code"]

    def chunk(self, document: Document) -> list[Chunk]:
        """Split a code document into chunks using tree-sitter parsing."""
        if not document.content:
            return []

        language = self._resolve_language(document)
        if language is None:
            return self._chunk_by_lines(document, parse_error=True)

        parser = self._get_parser(language)
        if parser is None:
            return self._chunk_by_lines(document, parse_error=True)

        try:
            tree = parser.parse(document.content.encode("utf-8"))
        except Exception:
            return self._chunk_by_lines(document, parse_error=True)

        if tree.root_node.has_error:
            return self._chunk_by_lines(document, parse_error=True)

        chunks = self._extract_chunks(document, tree, language)
        if not chunks:
            return self._chunk_by_lines(document, parse_error=False)
        return chunks

    def _resolve_language(self, document: Document) -> str | None:
        """Resolve a supported language from document metadata."""
        if document.metadata.language:
            language = document.metadata.language.strip().lower()
        else:
            extension = (document.metadata.file_extension or "").strip().lower()
            language = FILE_EXTENSION_LANGUAGE_MAP.get(extension, extension.lstrip("."))

        if language in {"py"}:
            language = "python"
        if language in {"js"}:
            language = "javascript"
        if language in {"golang"}:
            language = "go"

        if language in LANGUAGE_CONFIG:
            return language
        return None

    def _get_parser(self, language: str | None) -> Parser | None:
        """Get or create a tree-sitter parser for a language."""
        if language is None:
            return None

        config = LANGUAGE_CONFIG.get(language)
        if config is None:
            return None

        if language in self._parsers:
            return self._parsers[language]

        parser = Parser()
        language_obj = config["language"]
        set_language = getattr(parser, "set_language", None)
        if callable(set_language):
            set_language(language_obj)
        else:
            parser.language = language_obj
        self._parsers[language] = parser
        return parser

    def _extract_chunks(self, document: Document, tree: Tree, language: str) -> list[Chunk]:
        """Extract chunks from a parsed tree-sitter AST."""
        config = LANGUAGE_CONFIG[language]
        content = document.content
        root = tree.root_node

        imports_text = self._extract_imports(root, content, config)

        chunks: list[Chunk] = []
        chunk_index = 0
        module_statements: list[Node] = []

        for child in root.children:
            if not child.is_named:
                continue

            if child.type in config["import_types"]:
                continue

            if child.type == "decorated_definition":
                definition_node = self._unwrap_decorated_definition(child, config)
                if definition_node is None:
                    module_statements.append(child)
                    continue

                if definition_node.type in config["class_types"]:
                    class_name = self._get_node_name(definition_node, content)
                    node_chunks = self._process_node(
                        node=child,
                        document=document,
                        content=content,
                        config=config,
                        imports_text=imports_text,
                        class_name=class_name,
                        chunk_index_start=chunk_index,
                        definition_node=definition_node,
                    )
                    chunks.extend(node_chunks)
                    chunk_index += len(node_chunks)
                    continue

                if definition_node.type in config["function_types"]:
                    node_chunks = self._process_node(
                        node=child,
                        document=document,
                        content=content,
                        config=config,
                        imports_text=imports_text,
                        class_name=None,
                        chunk_index_start=chunk_index,
                        definition_node=definition_node,
                    )
                    chunks.extend(node_chunks)
                    chunk_index += len(node_chunks)
                    continue

                module_statements.append(child)
                continue

            if child.type in config["class_types"]:
                class_name = self._get_node_name(child, content)
                node_chunks = self._process_node(
                    node=child,
                    document=document,
                    content=content,
                    config=config,
                    imports_text=imports_text,
                    class_name=class_name,
                    chunk_index_start=chunk_index,
                )
                chunks.extend(node_chunks)
                chunk_index += len(node_chunks)
                continue

            if child.type in config["function_types"]:
                node_chunks = self._process_node(
                    node=child,
                    document=document,
                    content=content,
                    config=config,
                    imports_text=imports_text,
                    class_name=None,
                    chunk_index_start=chunk_index,
                )
                chunks.extend(node_chunks)
                chunk_index += len(node_chunks)
                continue

            module_statements.append(child)

        if module_statements:
            module_metadata = self._build_context_metadata(imports_text, None)
            module_metadata.update({"chunking_strategy": "code", "node_type": "module"})
            module_chunks = self._chunk_statement_nodes(
                nodes=module_statements,
                document=document,
                content=content,
                context_prefix=self._build_context_prefix(imports_text, None),
                extra_metadata=module_metadata,
                chunk_index_start=chunk_index,
            )
            chunks.extend(module_chunks)
            chunk_index += len(module_chunks)

        return chunks

    def _process_node(
        self,
        node: Node,
        document: Document,
        content: str,
        config: LanguageConfig,
        imports_text: str,
        class_name: str | None,
        chunk_index_start: int,
        definition_node: Node | None = None,
    ) -> list[Chunk]:
        """Process a function or class AST node into chunk(s)."""
        node_text, node_start, node_end = self._get_node_text(node, content)
        if not node_text.strip():
            return []

        target_node = definition_node or node
        function_name = None
        if target_node.type in config["function_types"]:
            function_name = self._get_node_name(target_node, content)

        if target_node.type in config["class_types"] and class_name is None:
            class_name = self._get_node_name(target_node, content)

        base_metadata = self._build_context_metadata(imports_text, class_name)
        base_metadata.update(
            {
                "chunking_strategy": "code",
                "node_type": target_node.type,
            }
        )

        if function_name:
            base_metadata["function_name"] = function_name

        if len(node_text) <= self.config.max_chunk_size:
            context_prefix = self._build_context_prefix(imports_text, class_name)
            chunk_content = f"{context_prefix}{node_text}" if context_prefix else node_text
            chunk = self._create_chunk(
                content=chunk_content,
                document=document,
                chunk_index=chunk_index_start,
                start_index=node_start,
                end_index=node_end,
                extra_metadata=base_metadata,
            )
            return [chunk] if chunk is not None else []

        body_node = self._get_body_node(target_node)
        if body_node is None:
            return self._chunk_text_by_lines(
                text=node_text,
                document=document,
                base_start=node_start,
                context_prefix=self._build_context_prefix(imports_text, class_name),
                extra_metadata=base_metadata,
                chunk_index_start=chunk_index_start,
            )

        statement_nodes = [
            child
            for child in body_node.children
            if child.is_named and child.type in config["statement_types"]
        ]

        if not statement_nodes:
            return self._chunk_text_by_lines(
                text=node_text,
                document=document,
                base_start=node_start,
                context_prefix=self._build_context_prefix(imports_text, class_name),
                extra_metadata=base_metadata,
                chunk_index_start=chunk_index_start,
            )

        header_text = self._get_header_text(node, target_node, content)
        context_prefix = self._build_context_prefix(imports_text, class_name, header_text)
        metadata = {
            **base_metadata,
            **self._build_context_metadata(imports_text, class_name, header_text),
        }

        return self._chunk_statement_nodes(
            nodes=statement_nodes,
            document=document,
            content=content,
            context_prefix=context_prefix,
            extra_metadata=metadata,
            chunk_index_start=chunk_index_start,
        )

    def _extract_imports(self, root: Node, content: str, config: LanguageConfig) -> str:
        """Extract import statements from the root node."""
        import_lines: list[str] = []
        for child in root.children:
            if not child.is_named or child.type not in config["import_types"]:
                continue

            node_text, _, _ = self._get_node_text(child, content)
            node_text = node_text.strip()
            if node_text:
                import_lines.append(node_text)

        return "\n".join(import_lines)

    def _chunk_by_lines(self, document: Document, parse_error: bool = False) -> list[Chunk]:
        """Fallback line-based chunking for parse errors."""
        extra_metadata: dict[str, object] = {
            "chunking_strategy": "code",
            "fallback_strategy": "line",
            "parse_error": parse_error,
        }

        return self._chunk_text_by_lines(
            text=document.content,
            document=document,
            base_start=0,
            context_prefix="",
            extra_metadata=extra_metadata,
            chunk_index_start=0,
        )

    def _chunk_text_by_lines(
        self,
        text: str,
        document: Document,
        base_start: int,
        context_prefix: str,
        extra_metadata: dict[str, object],
        chunk_index_start: int,
    ) -> list[Chunk]:
        """Chunk text by line boundaries with optional context prefix."""
        lines = text.splitlines(keepends=True)
        if not lines:
            return []

        chunks: list[Chunk] = []
        buffer: str = ""
        buffer_start = base_start
        chunk_index = chunk_index_start
        current_offset = base_start
        context_length = len(context_prefix)

        for line in lines:
            line_start = current_offset
            line_end = current_offset + len(line)

            if buffer and len(buffer) + len(line) + context_length > self.config.max_chunk_size:
                chunk = self._create_chunk(
                    content=f"{context_prefix}{buffer}" if context_prefix else buffer,
                    document=document,
                    chunk_index=chunk_index,
                    start_index=buffer_start,
                    end_index=buffer_start + len(buffer),
                    extra_metadata=extra_metadata,
                )
                if chunk is not None:
                    chunks.append(chunk)
                    chunk_index += 1

                if self.config.chunk_overlap > 0:
                    overlap_text = buffer[-self.config.chunk_overlap :]
                    chunk_end = buffer_start + len(buffer)
                    buffer = overlap_text
                    buffer_start = chunk_end - len(overlap_text)
                else:
                    buffer = ""

            if not buffer:
                buffer_start = line_start

            buffer += line
            current_offset = line_end

        if buffer.strip():
            chunk = self._create_chunk(
                content=f"{context_prefix}{buffer}" if context_prefix else buffer,
                document=document,
                chunk_index=chunk_index,
                start_index=buffer_start,
                end_index=buffer_start + len(buffer),
                extra_metadata=extra_metadata,
            )
            if chunk is not None:
                chunks.append(chunk)

        return chunks

    def _chunk_statement_nodes(
        self,
        nodes: list[Node],
        document: Document,
        content: str,
        context_prefix: str,
        extra_metadata: dict[str, object],
        chunk_index_start: int,
    ) -> list[Chunk]:
        """Group statement nodes into chunks while respecting size limits."""
        if not nodes:
            return []

        groups: list[tuple[int, int]] = []
        current_start: int | None = None
        current_end: int | None = None
        context_length = len(context_prefix)

        for node in nodes:
            node_start = byte_offset_to_char_offset(content, node.start_byte)
            node_end = byte_offset_to_char_offset(content, node.end_byte)

            if current_start is None:
                current_start = node_start
                current_end = node_end
                continue

            tentative_end = node_end
            tentative_length = context_length + (tentative_end - current_start)

            if tentative_length > self.config.max_chunk_size and current_end is not None:
                groups.append((current_start, current_end))
                current_start = node_start
                current_end = node_end
            else:
                current_end = node_end

        if current_start is not None and current_end is not None:
            groups.append((current_start, current_end))

        chunks: list[Chunk] = []
        chunk_index = chunk_index_start
        for start, end in groups:
            slice_text = content[start:end]
            if len(slice_text) + context_length > self.config.max_chunk_size:
                split_chunks = self._chunk_text_by_lines(
                    text=slice_text,
                    document=document,
                    base_start=start,
                    context_prefix=context_prefix,
                    extra_metadata=extra_metadata,
                    chunk_index_start=chunk_index,
                )
                chunks.extend(split_chunks)
                chunk_index += len(split_chunks)
                continue

            chunk_content = f"{context_prefix}{slice_text}" if context_prefix else slice_text
            chunk = self._create_chunk(
                content=chunk_content,
                document=document,
                chunk_index=chunk_index,
                start_index=start,
                end_index=end,
                extra_metadata=extra_metadata,
            )
            if chunk is not None:
                chunks.append(chunk)
                chunk_index += 1

        return chunks

    def _get_node_text(self, node: Node, content: str) -> tuple[str, int, int]:
        """Get a node's text and character offsets from byte offsets."""
        start_char = byte_offset_to_char_offset(content, node.start_byte)
        end_char = byte_offset_to_char_offset(content, node.end_byte)
        return content[start_char:end_char], start_char, end_char

    def _get_body_node(self, node: Node) -> Node | None:
        """Get the body node for a definition if available."""
        return node.child_by_field_name("body")

    def _unwrap_decorated_definition(self, node: Node, config: LanguageConfig) -> Node | None:
        """Unwrap a decorated_definition node to its underlying definition."""
        if node.type != "decorated_definition":
            return None

        definition_node = node.child_by_field_name("definition")
        if definition_node is not None:
            return definition_node

        for child in node.children:
            if child.is_named and (
                child.type in config["function_types"] or child.type in config["class_types"]
            ):
                return child
        return None

    def _get_node_name(self, node: Node, content: str) -> str | None:
        """Extract the name of a function/class node when possible."""
        name_node = node.child_by_field_name("name")
        if name_node is None:
            for child in node.children:
                if child.is_named and child.type in {
                    "identifier",
                    "property_identifier",
                    "type_identifier",
                }:
                    name_node = child
                    break

        if name_node is None:
            return None

        name_text, _, _ = self._get_node_text(name_node, content)
        return name_text.strip() or None

    def _get_header_text(self, outer_node: Node, definition_node: Node, content: str) -> str | None:
        """Extract the header text up to the body for large node splitting."""
        body_node = definition_node.child_by_field_name("body")
        if body_node is None:
            return None

        start_char = byte_offset_to_char_offset(content, outer_node.start_byte)
        body_start_char = byte_offset_to_char_offset(content, body_node.start_byte)
        header_text = content[start_char:body_start_char].strip()
        return header_text or None

    def _build_context_prefix(
        self, imports_text: str, class_name: str | None, header_text: str | None = None
    ) -> str:
        """Build a context prefix with imports and class headers."""
        parts: list[str] = []
        if imports_text:
            parts.append(imports_text)
        if class_name:
            parts.append(f"Class: {class_name}")
        if header_text:
            parts.append(header_text)

        if not parts:
            return ""

        return self.DEFAULT_CONTEXT_SEPARATOR.join(parts) + self.DEFAULT_CONTEXT_SEPARATOR

    def _build_context_metadata(
        self, imports_text: str, class_name: str | None, header_text: str | None = None
    ) -> dict[str, object]:
        """Build metadata describing the context prefix."""
        metadata: dict[str, object] = {}
        if imports_text:
            metadata["context_imports"] = imports_text
            metadata["imports"] = [
                line.strip() for line in imports_text.splitlines() if line.strip()
            ]
        if class_name:
            metadata["class_name"] = class_name
        if header_text:
            metadata["context_header"] = header_text

        context_prefix = self._build_context_prefix(imports_text, class_name, header_text)
        if context_prefix:
            metadata["context_prefix"] = context_prefix

        return metadata
