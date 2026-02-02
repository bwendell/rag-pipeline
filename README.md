# RAG Pipeline

A Retrieval-Augmented Generation (RAG) pipeline that enables natural language queries against your team's codebase, runbooks, and documentation.

## Overview

This pipeline:
- **Ingests** code, runbooks, and documentation from multiple sources
- **Chunks** content using intelligent, content-aware strategies
- **Embeds** chunks into vector representations
- **Stores** embeddings in a vector database for fast retrieval
- **Retrieves** relevant context based on user queries
- **Generates** answers using an LLM with the retrieved context

## Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai/) for local LLM inference

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd rag

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install with dev dependencies
pip install -e ".[dev]"

# Install Ollama and pull a model
ollama pull mistral
```

### Running the API

```bash
# Start the development server
uvicorn rag_pipeline.main:app --reload

# API is available at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

### Ingesting Documents

```bash
# Ingest a local directory
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{"source_path": "/path/to/your/codebase"}'
```

### Querying

```bash
# Ask a question
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "How does the authentication system work?"}'
```

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Triggers   │────▶│   Pipeline   │────▶│  LLM Provider│
│  HTTP/Slack  │     │ Orchestrator │     │   (Ollama)   │
└──────────────┘     └──────────────┘     └──────────────┘
                            │
                            ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Document   │────▶│   Chunkers   │────▶│ Vector Store │
│   Sources    │     │ Code/MD/Text │     │  (ChromaDB)  │
└──────────────┘     └──────────────┘     └──────────────┘
```

## Configuration

Configuration is managed via environment variables (with `RAG_` prefix):

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_VECTOR_STORE_TYPE` | `chroma` | Vector store backend |
| `RAG_LLM_PROVIDER_TYPE` | `ollama` | LLM provider |
| `RAG_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API URL |
| `RAG_OLLAMA_MODEL` | `mistral` | Ollama model to use |
| `RAG_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Embedding model |
| `RAG_CHUNK_SIZE` | `1000` | Target chunk size in characters |
| `RAG_CHUNK_OVERLAP` | `200` | Chunk overlap in characters |

## Project Structure

```
rag/
├── src/rag_pipeline/
│   ├── core/           # Protocols and shared types
│   ├── chunkers/       # Content chunking strategies
│   ├── vector_stores/  # Vector database implementations
│   ├── llm_providers/  # LLM integrations
│   ├── embedding_providers/  # Embedding models
│   ├── document_sources/    # Data sources
│   ├── pipeline/       # Orchestration
│   └── api/            # FastAPI endpoints
├── tests/              # Test suite
└── docs/               # Documentation
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=rag_pipeline --cov-report=html

# Run specific test file
pytest tests/test_chunkers/test_code_chunker.py
```

### Code Quality

```bash
# Type checking
mypy src

# Linting
ruff check src

# Format code
ruff format src
```

## Extending the Pipeline

The pipeline is designed with abstraction layers for easy extension:

- **Add a new vector store**: Implement the `VectorStore` protocol
- **Add a new LLM provider**: Implement the `LLMProvider` protocol
- **Add a new document source**: Implement the `DocumentSource` protocol
- **Add a new trigger**: Implement the `Trigger` protocol

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed extension guides.

## Roadmap

- [x] Phase 0: Project Scaffolding
- [x] Phase 1: Core Abstractions
- [x] Phase 2: Chunking Engine
- [x] Phase 3: Embedding Provider
- [x] Phase 4: Vector Store (ChromaDB)
- [x] Phase 5: LLM Provider (Ollama)
- [x] Phase 6: Document Sources (Filesystem)
- [ ] Phase 7: Pipeline Orchestration
- [ ] Phase 8: HTTP API

## Future Integrations

| Component | Integration |
|-----------|-------------|
| Vector Store | OCI Vector Search (Oracle 23ai) |
| LLM | OCI GenAI Service |
| Documents | Confluence, Git, OCI Object Storage |
| Triggers | Slack, Microsoft Teams |

## License

MIT License - see LICENSE file for details.
