# Architecture Documentation

## System Overview

The RAG Pipeline is a modular, extensible system for querying team knowledge bases using natural language. It follows a clean architecture pattern with clear separation between:

- **Core**: Domain models and protocols (no external dependencies)
- **Adapters**: Implementations of protocols for specific technologies
- **Application**: Pipeline orchestration and business logic
- **Interface**: API endpoints and triggers

## Component Architecture

### 1. Core Layer (`src/rag_pipeline/core/`)

The core layer defines the contracts that all components must follow.

#### 1.1 Data Types (`types.py`)

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

@dataclass
class Document:
    """A source document before chunking."""
    id: str
    content: str
    source_path: str
    source_type: str  # "code", "markdown", "runbook", "text"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class Chunk:
    """A chunk of a document, ready for embedding."""
    id: str
    document_id: str
    content: str
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # Position info
    start_line: int | None = None
    end_line: int | None = None

@dataclass
class SearchResult:
    """A search result from the vector store."""
    chunk: Chunk
    score: float
    distance: float
```

#### 1.2 Protocols

All external integrations are defined as Python Protocols:

| Protocol | Purpose | Key Methods |
|----------|---------|-------------|
| `VectorStore` | Store and retrieve embeddings | `add()`, `search()`, `delete()` |
| `LLMProvider` | Generate text completions | `complete()`, `complete_stream()` |
| `EmbeddingProvider` | Generate embeddings | `embed()`, `embed_batch()` |
| `DocumentSource` | Load documents | `scan()`, `load()` |
| `Chunker` | Split documents into chunks | `chunk()` |
| `Trigger` | Receive external events | `start()`, `stop()` |

### 2. Adapter Layer

Adapters implement protocols for specific technologies.

#### 2.1 Vector Stores (`vector_stores/`)

| Adapter | Status | Description |
|---------|--------|-------------|
| `ChromaDBStore` | Implemented | Local embedded vector database |
| `InMemoryStore` | Implemented | In-memory store for testing |
| `OCIVectorStore` | Stub | Oracle Database 23ai Vector Search |
| `QdrantStore` | Stub | Qdrant vector database |

**ChromaDB Configuration:**
- Persistence: Local filesystem (`./data/chroma`)
- Distance metric: Cosine similarity
- Index type: HNSW (default)

#### 2.2 LLM Providers (`llm_providers/`)

| Adapter | Status | Description |
|---------|--------|-------------|
| `OllamaProvider` | Implemented | Local LLM via Ollama |
| `OCIGenAIProvider` | Stub | OCI GenAI Service |
| `OpenAIProvider` | Stub | OpenAI API |
| `AnthropicProvider` | Stub | Anthropic API |

**Ollama Configuration:**
- API: OpenAI-compatible REST API
- Models: Mistral, Llama2, CodeLlama
- Streaming: Supported via SSE

#### 2.3 Embedding Providers (`embedding_providers/`)

| Adapter | Status | Description |
|---------|--------|-------------|
| `SentenceTransformersProvider` | Implemented | Local embedding model |
| `OCIEmbeddingProvider` | Stub | OCI GenAI Embeddings |
| `OpenAIEmbeddingProvider` | Stub | OpenAI Embeddings |

**Default Model:**
- Model: `all-MiniLM-L6-v2`
- Dimensions: 384
- Max tokens: 256

#### 2.4 Chunkers (`chunkers/`)

| Chunker | Use Case | Strategy |
|---------|----------|----------|
| `CodeChunker` | Source code | AST-based (tree-sitter) |
| `MarkdownChunker` | Documentation | Header-based hierarchical |
| `GenericChunker` | Plain text | Recursive text splitter |

**Chunking Strategy Details:**

**Code Chunking (tree-sitter):**
1. Parse code into AST
2. Extract top-level definitions (functions, classes)
3. If definition > max_size, recursively split
4. Preserve context (imports, class name for methods)

**Markdown Chunking:**
1. Split by headers (H1 → H4)
2. Preserve code blocks intact
3. Include parent headers in metadata for context
4. Configurable overlap between sections

#### 2.5 Document Sources (`document_sources/`)

| Source | Status | Description |
|--------|--------|-------------|
| `FileSystemSource` | Implemented | Local directory scanning |
| `GitSource` | Stub | Git repository cloning |
| `ConfluenceSource` | Stub | Confluence API |
| `S3Source` | Stub | OCI/AWS Object Storage |

### 3. Pipeline Layer (`pipeline/`)

#### 3.1 Ingestion Pipeline

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Document     │───▶│  Chunkers    │───▶│  Embedding   │───▶│ Vector       │
│ Sources      │    │              │    │  Provider    │    │ Store        │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
     scan()              chunk()            embed()              add()
```

**Flow:**
1. `DocumentSource.scan()` - Find all documents
2. `DocumentSource.load()` - Load document content
3. `Chunker.chunk()` - Split into chunks (selected by file type)
4. `EmbeddingProvider.embed_batch()` - Generate embeddings
5. `VectorStore.add()` - Store chunks with embeddings

#### 3.2 Retrieval Pipeline

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Query      │───▶│  Embedding   │───▶│ Vector       │
│              │    │  Provider    │    │ Store        │
└──────────────┘    └──────────────┘    └──────────────┘
                        embed()             search()
```

**Flow:**
1. Receive query string
2. `EmbeddingProvider.embed()` - Embed the query
3. `VectorStore.search()` - Find similar chunks
4. (Optional) Rerank results
5. Return ranked `SearchResult` list

#### 3.3 Generation Pipeline

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Context    │───▶│   Prompt     │───▶│ LLM          │
│   Assembly   │    │   Builder    │    │ Provider     │
└──────────────┘    └──────────────┘    └──────────────┘
```

**Flow:**
1. Assemble context from `SearchResult` chunks
2. Build prompt with system message, context, and query
3. `LLMProvider.complete()` - Generate response
4. Format response with source citations
5. (Optional) Stream response tokens

### 4. API Layer (`api/`)

#### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/query` | Query the RAG pipeline |
| `POST` | `/query/stream` | Streaming query response |
| `POST` | `/ingest` | Trigger document ingestion |
| `GET` | `/health` | Health check |
| `GET` | `/stats` | Vector store statistics |

#### Request/Response Models

```python
class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    include_sources: bool = True

class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceReference]
    metadata: dict[str, Any]

class SourceReference(BaseModel):
    file_path: str
    start_line: int | None
    end_line: int | None
    score: float
    snippet: str
```

## Configuration System

Configuration uses Pydantic Settings with environment variable support:

```python
class Settings(BaseSettings):
    # Vector Store
    vector_store_type: Literal["chroma", "memory", "oci", "qdrant"] = "chroma"
    chroma_persist_dir: Path = Path("./data/chroma")
    
    # LLM
    llm_provider_type: Literal["ollama", "oci", "openai", "anthropic"] = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "mistral"
    
    # Embeddings
    embedding_provider_type: Literal["sentence_transformers", "oci", "openai"] = "sentence_transformers"
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    model_config = SettingsConfigDict(env_prefix="RAG_")
```

## Data Flow

### Full Query Flow

```
User Query
    │
    ▼
┌───────────────────────────────────────────────────────────────────┐
│                         FastAPI Endpoint                          │
│                          POST /query                              │
└───────────────────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────────────────┐
│                        RAGPipeline.query()                        │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐   │
│  │ RetrievalPipe   │─▶│ GenerationPipe  │─▶│ Response Format │   │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘   │
└───────────────────────────────────────────────────────────────────┘
    │                         │
    ▼                         ▼
┌─────────────┐         ┌─────────────┐
│ Embedding   │         │ LLM         │
│ Provider    │         │ Provider    │
│ (local)     │         │ (Ollama)    │
└─────────────┘         └─────────────┘
    │
    ▼
┌─────────────┐
│ Vector      │
│ Store       │
│ (ChromaDB)  │
└─────────────┘
```

## Error Handling

### Exception Hierarchy

```
RAGPipelineError (base)
├── SourceError
│   ├── SourceNotFoundError
│   └── SourceAccessError
├── ChunkingError
│   ├── ParseError
│   └── ChunkSizeError
├── EmbeddingError
│   ├── ModelNotFoundError
│   └── EmbeddingDimensionError
├── VectorStoreError
│   ├── StoreConnectionError
│   └── SearchError
├── LLMError
│   ├── ModelNotAvailableError
│   ├── GenerationError
│   └── TokenLimitError
└── ConfigurationError
```

### Error Recovery Strategies

| Error Type | Strategy |
|------------|----------|
| Transient network | Retry with exponential backoff |
| Model not available | Fallback to default model |
| Token limit exceeded | Truncate context |
| Parse error | Skip file, log warning |

## Testing Strategy

### Test Categories

1. **Unit Tests**: Test individual components in isolation
   - Mock all external dependencies
   - Fast execution (<100ms per test)

2. **Integration Tests**: Test component interactions
   - Use in-memory implementations
   - May use local services (Ollama, ChromaDB)

3. **End-to-End Tests**: Test full pipeline
   - Marked with `@pytest.mark.e2e`
   - Require running services

### Test Fixtures

```python
# conftest.py
@pytest.fixture
def sample_code_file() -> str:
    return '''
def hello(name: str) -> str:
    """Say hello."""
    return f"Hello, {name}!"
'''

@pytest.fixture
def mock_embedding_provider():
    provider = Mock(spec=EmbeddingProvider)
    provider.embed.return_value = [0.1] * 384
    return provider

@pytest.fixture
def in_memory_store():
    return InMemoryStore()
```

## Performance Considerations

### Chunking
- Batch document loading (100 docs at a time)
- Parallel AST parsing with thread pool
- Cache parsed ASTs for large files

### Embedding
- Batch embedding requests (32 chunks per batch)
- Async embedding for non-blocking ingestion
- Model warm-up on startup

### Vector Search
- Use HNSW index for fast approximate search
- Metadata filtering before vector search
- Limit top_k to reduce memory usage

### LLM Generation
- Stream responses for better UX
- Truncate context to fit token limit
- Cache common queries (optional)
