# AGENTS.md - AI Assistant Guidance

## Project Overview

This is a **RAG (Retrieval-Augmented Generation) Pipeline** that enables natural language queries against:
- Team codebase (Java, Python, Go)
- Operational runbooks
- Internal documentation

## Architecture Philosophy

### Core Principles

1. **Abstraction First**: Every external dependency has a Protocol-based abstraction
2. **Local by Default**: All implementations work locally without cloud credentials
3. **Stubs for Cloud**: OCI/cloud integrations are stubs until needed
4. **Factory Pattern**: Component selection via configuration, not code changes

### Component Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                         RAG Pipeline                            │
├─────────────────────────────────────────────────────────────────┤
│  Triggers (HTTP, Slack*, Teams*)                               │
├─────────────────────────────────────────────────────────────────┤
│  Pipeline Orchestration (Ingestion, Retrieval, Generation)     │
├───────────────┬───────────────┬───────────────┬─────────────────┤
│ Document      │ Chunkers      │ Vector        │ LLM             │
│ Sources       │               │ Stores        │ Providers       │
├───────────────┼───────────────┼───────────────┼─────────────────┤
│ FileSystem    │ CodeChunker   │ ChromaDB      │ Ollama          │
│ Git*          │ MarkdownChunk │ InMemory      │ OCI GenAI*      │
│ Confluence*   │ GenericChunk  │ OCI Vector*   │ OpenAI*         │
│ S3*           │               │ Qdrant*       │ Anthropic*      │
└───────────────┴───────────────┴───────────────┴─────────────────┘
                         * = Stub only
```

## Directory Structure

```
src/rag_pipeline/
├── core/           # Protocols, types, exceptions (FOUNDATION)
├── chunkers/       # Content chunking strategies
├── vector_stores/  # Vector database implementations
├── llm_providers/  # LLM API integrations
├── embedding_providers/  # Embedding model integrations
├── document_sources/    # Data ingestion sources
├── triggers/       # External trigger mechanisms
├── pipeline/       # Orchestration layer
└── api/            # FastAPI application
```

## Code Conventions

### Type Hints
- **Required** on all public function signatures
- Use `from __future__ import annotations` for forward references
- Prefer `list[X]` over `List[X]` (Python 3.11+)

### Async/Await
- All I/O operations are async
- Use `anyio` for async primitives (not `asyncio` directly)
- Sync operations wrapped with `anyio.to_thread.run_sync()`

### Error Handling
- Custom exceptions in `core/exceptions.py`
- Never catch generic `Exception` without re-raising
- Use `structlog` for logging, not `print` or `logging`

### Testing
- Tests mirror source structure in `tests/`
- Use `pytest-asyncio` for async tests
- Mock external services, never hit real APIs in tests
- Fixtures in `conftest.py` at appropriate levels

### Naming Conventions
- Files: `snake_case.py`
- Classes: `PascalCase`
- Functions/Variables: `snake_case`
- Constants: `SCREAMING_SNAKE_CASE`
- Protocols: Named after capability (e.g., `VectorStore`, not `IVectorStore`)

## Implementation Order

Phases must be completed sequentially:

1. **Phase 0**: Project Scaffolding (this setup)
2. **Phase 1**: Core Abstractions (Protocols in `core/`)
3. **Phase 2**: Chunking Engine
4. **Phase 3**: Embedding Provider
5. **Phase 4**: Vector Store
6. **Phase 5**: LLM Provider
7. **Phase 6**: Document Sources
8. **Phase 7**: Pipeline Orchestration
9. **Phase 8**: HTTP API & Triggers

## OCI Context

### Likely Internal Tools (Confirm During Onboarding)
| Category | Expected Tool |
|----------|--------------|
| Code Repos | Visual Builder Studio (VBS) or internal Git |
| Runbooks | Fleet Application Management or Confluence |
| Documentation | Confluence or internal wiki |
| CI/CD | OCI DevOps Build Pipelines |

### Future OCI Integrations
- **OCI GenAI**: LLM inference (Cohere Command, Llama)
- **Oracle Database 23ai**: Vector search with HNSW/IVF indexes
- **OCI Object Storage**: Runbook and document storage

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Language | Python 3.11+ | ML ecosystem, async support |
| Framework | FastAPI | Auto-docs, typing, async |
| Local Vector DB | ChromaDB | Zero config, embedded |
| Local LLM | Ollama | OpenAI-compatible API |
| Code Parsing | tree-sitter | Language-agnostic AST |
| Embeddings | sentence-transformers | Local, no API keys |

## Common Tasks

### Add a New Vector Store

1. Create `src/rag_pipeline/vector_stores/my_store.py`
2. Implement the `VectorStore` Protocol from `core/base_vector_store.py`
3. Register in `vector_stores/factory.py`
4. Add tests in `tests/test_vector_stores/test_my_store.py`
5. Add config options to `config.py`

### Add a New LLM Provider

1. Create `src/rag_pipeline/llm_providers/my_provider.py`
2. Implement the `LLMProvider` Protocol from `core/base_llm_provider.py`
3. Register in `llm_providers/factory.py`
4. Add tests in `tests/test_llm_providers/test_my_provider.py`
5. Add config options to `config.py`

### Run the Pipeline Locally

```bash
# Start Ollama (separate terminal)
ollama serve

# Pull a model
ollama pull mistral

# Install and run
pip install -e ".[dev]"
uvicorn rag_pipeline.main:app --reload
```

## Quality Gates

Before completing any phase:
- [ ] All tests pass (`pytest`)
- [ ] Type checking passes (`mypy src`)
- [ ] Linting passes (`ruff check src`)
- [ ] Coverage >80% for new code (`pytest --cov`)
