"""Markdown chunker implementation with header-based splitting.

This module provides the MarkdownChunker class that implements intelligent
markdown chunking by splitting at header boundaries while preserving:
- Header hierarchy information in metadata
- Code blocks as atomic units (never split within)
- Tables and lists at natural boundaries
- Content before the first header
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from rag_pipeline.chunkers.base import AbstractChunker, calculate_line_numbers
from rag_pipeline.core.types import Chunk, Document, Metadata

if TYPE_CHECKING:
    from rag_pipeline.core.base_chunker import ChunkingConfig


@dataclass
class MarkdownSection:
    """Represents a section of a markdown document.

        A section is defined by a header and all content following it until
    the next header at the same or higher level (lower number).

        Attributes:
            level: Header level (1-6), 0 for content before first header
            title: Header text (or empty for pre-header content)
            content: Section content (without header line)
            start_index: Start position in original document
            end_index: End position in original document
            parent_titles: Parent header hierarchy for context
    """

    level: int
    title: str
    content: str
    start_index: int
    end_index: int
    parent_titles: list[str]


class MarkdownChunker(AbstractChunker):
    """Markdown-aware chunker with header-based splitting.

    This chunker splits markdown documents at header boundaries while:
    1. Preserving header hierarchy in metadata
    2. Keeping code blocks intact (atomic units)
    3. Handling content before the first header
    4. Gracefully handling malformed markdown

    Attributes:
        config: The chunking configuration controlling behavior.
        HEADER_PATTERN: Regex pattern for markdown headers
        CODE_BLOCK_PATTERN: Regex pattern for fenced code blocks
    """

    HEADER_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"^(#{1,6})\s+(.+?)$", re.MULTILINE)
    CODE_BLOCK_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"^(```|~~~)[^\n]*\n[\s\S]*?\n\1|^```[^\n]*$|^~~~[^\n]*$",
        re.MULTILINE,
    )

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        """Initialize the markdown chunker with configuration."""
        super().__init__(config)

    def get_supported_types(self) -> list[str]:
        """Get the document types this chunker supports."""
        return ["markdown", "runbook"]

    def chunk(self, document: Document) -> list[Chunk]:
        """Split a markdown document into chunks."""
        if not document.content:
            return []

        content = document.content
        code_blocks = self._find_code_blocks(content)
        sections = self._parse_sections(content, code_blocks)
        return self._sections_to_chunks(sections, document)

    def _find_code_blocks(self, content: str) -> list[dict[str, object]]:
        """Find all code blocks and their positions in the content.

        Returns a list of dictionaries containing:
            - start: Start position of code block
            - end: End position of code block
            - content: The code block content
        """
        code_blocks: list[dict[str, object]] = []
        for match in self.CODE_BLOCK_PATTERN.finditer(content):
            code_blocks.append(
                {
                    "start": match.start(),
                    "end": match.end(),
                    "content": match.group(0),
                }
            )
        return code_blocks

    def _is_in_code_block(self, pos: int, code_blocks: list[dict[str, object]]) -> bool:
        """Check if a position is inside any code block."""
        for block in code_blocks:
            start: int = block["start"]  # type: ignore[assignment]
            end: int = block["end"]  # type: ignore[assignment]
            if start <= pos < end:
                return True
        return False

    def _parse_sections(
        self, content: str, code_blocks: list[dict[str, object]]
    ) -> list[MarkdownSection]:
        """Parse markdown into sections based on headers.

        Splits the document at header boundaries (H1-H6), excluding headers
        that are inside code blocks. Creates sections with hierarchy info.
        """
        sections: list[MarkdownSection] = []
        header_matches = list(self.HEADER_PATTERN.finditer(content))

        effective_headers = [
            m for m in header_matches if not self._is_in_code_block(m.start(), code_blocks)
        ]

        if not effective_headers:
            if content.strip():
                sections.append(
                    MarkdownSection(
                        level=0,
                        title="",
                        content=content,
                        start_index=0,
                        end_index=len(content),
                        parent_titles=[],
                    )
                )
            return sections

        first_header_start = effective_headers[0].start()
        if first_header_start > 0:
            pre_header_content = content[:first_header_start]
            if pre_header_content.strip():
                sections.append(
                    MarkdownSection(
                        level=0,
                        title="",
                        content=pre_header_content,
                        start_index=0,
                        end_index=first_header_start,
                        parent_titles=[],
                    )
                )

        parent_stack: list[tuple[int, str]] = []

        for i, match in enumerate(effective_headers):
            hashes = match.group(1)
            title = match.group(2).strip()
            level = len(hashes)
            header_line_start = match.start()
            header_line_end = match.end()

            if i + 1 < len(effective_headers):
                next_header_start = effective_headers[i + 1].start()
            else:
                next_header_start = len(content)

            section_content_end = next_header_start

            while parent_stack and parent_stack[-1][0] >= level:
                parent_stack.pop()

            parent_titles = [p[1] for p in parent_stack]

            sections.append(
                MarkdownSection(
                    level=level,
                    title=title,
                    content=content[header_line_end:section_content_end],
                    start_index=header_line_start,
                    end_index=section_content_end,
                    parent_titles=parent_titles,
                )
            )

            parent_stack.append((level, title))

        return sections

    def _sections_to_chunks(
        self, sections: list[MarkdownSection], document: Document
    ) -> list[Chunk]:
        """Convert markdown sections to Chunk objects."""
        chunks: list[Chunk] = []

        for section_index, section in enumerate(sections):
            if section.level > 0:
                header_prefix = f"{'#' * section.level} {section.title}\n\n"
                chunk_content = header_prefix + section.content
            else:
                chunk_content = section.content

            if not chunk_content.strip():
                continue

            start_index = section.start_index
            end_index = section.end_index

            extra_metadata: dict[str, object] = {
                "header_level": section.level,
                "chunking_strategy": "markdown",
            }

            if section.title:
                extra_metadata["header_title"] = section.title

            if section.parent_titles:
                extra_metadata["parent_headers"] = section.parent_titles
                extra_metadata["header_path"] = " > ".join(section.parent_titles)

            start_line, end_line = calculate_line_numbers(document.content, start_index, end_index)

            chunk_id = self._generate_chunk_id(document.id, section_index, chunk_content)

            chunk_metadata = Metadata(
                source_path=document.metadata.source_path,
                source_type=document.metadata.source_type,
                language=document.metadata.language,
                file_extension=document.metadata.file_extension,
                created_at=document.metadata.created_at,
                updated_at=document.metadata.updated_at,
            )
            chunk_metadata.custom.update(document.metadata.custom)
            chunk_metadata.custom["chunk_index"] = section_index
            chunk_metadata.custom["start_index"] = start_index
            chunk_metadata.custom["end_index"] = end_index
            chunk_metadata.custom["start_line"] = start_line
            chunk_metadata.custom["end_line"] = end_line
            chunk_metadata.custom.update(extra_metadata)

            chunk = Chunk(
                id=chunk_id,
                content=chunk_content,
                document_id=document.id,
                metadata=chunk_metadata,
                start_index=start_index,
                end_index=end_index,
                chunk_index=section_index,
            )

            chunks.append(chunk)

        return chunks
