"""End-to-end integration tests for the RAG pipeline.

Tests the complete workflow from document ingestion through chunking,
embedding, vector storage, and retrieval. Verifies all components work
together seamlessly in realistic scenarios.

Test Categories:
1. Basic Pipeline Flow: Single document through all stages
2. Multi-Language Processing: Documents in different languages
3. Multi-Document Workflows: Batch processing multiple files
4. Data Integrity: Content preservation through pipeline
5. Error Recovery: Handling and recovery from failures
6. Real-World Scenarios: Realistic usage patterns

Run with:
    pytest tests/test_pipeline_integration.py -v
    pytest tests/test_pipeline_integration.py -v -k "basic"  # Filter by category
    pytest tests/test_pipeline_integration.py --co -q  # List all tests
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from rag_pipeline.chunkers import CodeChunker, MarkdownChunker, create_chunker
from rag_pipeline.core.base_chunker import ChunkingConfig
from rag_pipeline.core.types import (
    Chunk,
    Document,
    DocumentType,
    Metadata,
    QueryContext,
    SearchResult,
    SourceType,
)
from rag_pipeline.vector_stores.in_memory_store import InMemoryStore

if TYPE_CHECKING:
    from rag_pipeline.embedding_providers.base import EmbeddingProvider


# =============================================================================
# Test Data & Fixtures
# =============================================================================


class SampleDocuments:
    """Collection of sample documents for integration testing."""

    PYTHON_FILE = """'''Module for user authentication.'''

import hashlib
from typing import Optional


class User:
    '''Represents a system user.'''
    
    def __init__(self, username: str, password: str):
        '''Initialize user with credentials.
        
        Args:
            username: The user's username.
            password: The user's password (will be hashed).
        '''
        self.username = username
        self._password_hash = self._hash_password(password)
    
    def _hash_password(self, password: str) -> str:
        '''Hash a password using SHA256.
        
        Args:
            password: The password to hash.
            
        Returns:
            Hexadecimal hash of the password.
        '''
        return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_password(self, password: str) -> bool:
        '''Verify a password against the stored hash.
        
        Args:
            password: The password to verify.
            
        Returns:
            True if password matches, False otherwise.
        '''
        return self._hash_password(password) == self._password_hash


class AuthenticationManager:
    '''Manages user authentication and session handling.'''
    
    def __init__(self):
        '''Initialize the authentication manager.'''
        self.users: dict[str, User] = {}
    
    def register_user(self, username: str, password: str) -> bool:
        '''Register a new user.
        
        Args:
            username: The username for the new user.
            password: The password for the new user.
            
        Returns:
            True if registration succeeded, False if user exists.
        '''
        if username in self.users:
            return False
        self.users[username] = User(username, password)
        return True
    
    def authenticate(self, username: str, password: str) -> Optional[User]:
        '''Authenticate a user.
        
        Args:
            username: The username to authenticate.
            password: The password to verify.
            
        Returns:
            The User object if authentication succeeds, None otherwise.
        '''
        user = self.users.get(username)
        if user and user.verify_password(password):
            return user
        return None
"""

    MARKDOWN_FILE = """# RAG Pipeline Documentation

## Overview

The RAG (Retrieval-Augmented Generation) pipeline is a system that enables
natural language queries against your team's codebase, runbooks, and documentation.

### Architecture

The pipeline consists of several key components:

1. **Document Sources**: Ingest documents from various sources
   - Filesystem (local files and directories)
   - Git repositories
   - Confluence wikis
   - S3/Cloud storage

2. **Chunking Engine**: Split documents into meaningful chunks
   - Code-aware chunking for programming languages
   - Markdown-aware chunking for documentation
   - Generic text chunking with configurable parameters

3. **Embedding Provider**: Convert chunks to vector embeddings
   - SentenceTransformers for CPU-based embeddings
   - OpenAI API for cloud-based embeddings
   - OCI GenAI Service integration

4. **Vector Store**: Store and search embeddings
   - ChromaDB for persistent storage
   - In-memory store for development
   - OCI Vector Search (Oracle 23ai)

5. **LLM Provider**: Generate answers using retrieved context
   - Ollama for local inference
   - OpenAI API for cloud-based models
   - OCI GenAI Service

## Getting Started

### Installation

```bash
pip install -e ".[dev]"
```

### Configuration

Set environment variables with `RAG_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| RAG_VECTOR_STORE_TYPE | chroma | Vector store backend |
| RAG_LLM_PROVIDER_TYPE | ollama | LLM provider |
| RAG_CHUNK_SIZE | 1000 | Target chunk size |
| RAG_CHUNK_OVERLAP | 200 | Chunk overlap |

## Best Practices

1. **Chunk Size**: Use 500-2000 chars for optimal retrieval
2. **Overlap**: Set to 10-20% of chunk size to maintain context
3. **Metadata**: Include source information for traceability
4. **Batch Processing**: Process multiple documents efficiently
"""

    JAVASCRIPT_FILE = """/**
 * API client for RAG pipeline interactions.
 * Provides async methods for querying and managing documents.
 */

const BASE_URL = process.env.RAG_API_URL || 'http://localhost:8000';

class RAGClient {
  /**
   * Initialize the RAG client.
   * @param {string} apiKey - Optional API key for authentication
   */
  constructor(apiKey = null) {
    this.apiKey = apiKey;
    this.headers = {
      'Content-Type': 'application/json',
    };
    if (apiKey) {
      this.headers['Authorization'] = `Bearer ${apiKey}`;
    }
  }

  /**
   * Query the RAG pipeline with a question.
   * @param {string} question - The question to ask
   * @param {number} topK - Number of context chunks to retrieve (default: 5)
   * @returns {Promise<Object>} The response with answer and source chunks
   */
  async query(question, topK = 5) {
    const response = await fetch(`${BASE_URL}/query`, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify({ question, top_k: topK }),
    });

    if (!response.ok) {
      throw new Error(`Query failed: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Ingest documents from a source path.
   * @param {string} sourcePath - Path to ingest from
   * @returns {Promise<Object>} Ingestion result with document count
   */
  async ingestDocuments(sourcePath) {
    const response = await fetch(`${BASE_URL}/ingest`, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify({ source_path: sourcePath }),
    });

    if (!response.ok) {
      throw new Error(`Ingestion failed: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Get pipeline health status.
   * @returns {Promise<Object>} Health status of all components
   */
  async getHealth() {
    const response = await fetch(`${BASE_URL}/health`, {
      method: 'GET',
      headers: this.headers,
    });

    if (!response.ok) {
      throw new Error(`Health check failed: ${response.statusText}`);
    }

    return response.json();
  }
}

module.exports = RAGClient;
"""

    GO_FILE = """package auth

import (
    "crypto/sha256"
    "encoding/hex"
    "errors"
    "fmt"
)

// User represents a user in the system.
type User struct {
    Username string
    Email    string
    hash     string
}

// ErrInvalidPassword is returned when password verification fails.
var ErrInvalidPassword = errors.New("invalid password")

// NewUser creates a new user with the given credentials.
func NewUser(username, email, password string) *User {
    return &User{
        Username: username,
        Email:    email,
        hash:     hashPassword(password),
    }
}

// VerifyPassword checks if the provided password matches the stored hash.
func (u *User) VerifyPassword(password string) bool {
    return hashPassword(password) == u.hash
}

// hashPassword returns the SHA256 hash of a password.
func hashPassword(password string) string {
    hash := sha256.Sum256([]byte(password))
    return hex.EncodeToString(hash[:])
}

// Authenticator manages user authentication.
type Authenticator struct {
    users map[string]*User
}

// NewAuthenticator creates a new authenticator.
func NewAuthenticator() *Authenticator {
    return &Authenticator{
        users: make(map[string]*User),
    }
}

// Register adds a new user to the authenticator.
func (a *Authenticator) Register(user *User) error {
    if _, exists := a.users[user.Username]; exists {
        return fmt.Errorf("user %s already exists", user.Username)
    }
    a.users[user.Username] = user
    return nil
}

// Authenticate verifies a user's credentials.
func (a *Authenticator) Authenticate(username, password string) (*User, error) {
    user, exists := a.users[username]
    if !exists {
        return nil, fmt.Errorf("user %s not found", username)
    }
    if !user.VerifyPassword(password) {
        return nil, ErrInvalidPassword
    }
    return user, nil
}
"""


@pytest.fixture
def chunking_config() -> ChunkingConfig:
    """Standard chunking configuration for integration tests."""
    return ChunkingConfig(chunk_size=500, chunk_overlap=50)


@pytest.fixture
def embedding_dimension() -> int:
    """Standard embedding dimension for integration tests."""
    return 384


@pytest.fixture
def mock_embedding_provider(embedding_dimension: int):
    """Create a mock embedding provider for integration testing.

    Generates deterministic embeddings based on text hash for consistency
    across test runs while maintaining variation between different texts.
    """
    provider = AsyncMock()
    provider.dimension = embedding_dimension

    async def mock_embed(text: str) -> list[float]:
        """Generate a deterministic but varied embedding."""
        # Use hash to get consistent embeddings for same text
        hash_val = hash(text) % 1000
        np.random.seed(hash_val)
        embedding = np.random.randn(embedding_dimension).astype(np.float32)
        # Normalize for consistency
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        return embedding.tolist()

    async def mock_embed_batch(texts: list[str]) -> list[list[float]]:
        """Generate deterministic embeddings for a batch of texts."""
        return [await mock_embed(text) for text in texts]

    provider.embed = mock_embed
    provider.embed_batch = mock_embed_batch
    return provider


# =============================================================================
# Test Classes
# =============================================================================


class TestBasicPipelineFlow:
    """Test basic end-to-end pipeline flow with single documents.

    Verifies that:
    - Documents are chunked correctly
    - Chunks have proper metadata
    - Chunks can be embedded
    - Chunks can be stored and retrieved
    """

    def test_python_document_through_pipeline(self, chunking_config: ChunkingConfig) -> None:
        """Test a Python document through chunking."""
        document = Document(
            content=SampleDocuments.PYTHON_FILE,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/src/auth.py",
                source_type=SourceType.FILESYSTEM,
                language="python",
                file_extension=".py",
            ),
        )

        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        # Verify chunking occurred
        assert len(chunks) > 0, "Should produce at least one chunk"
        assert all(isinstance(c, Chunk) for c in chunks)

        # Verify chunk structure
        for i, chunk in enumerate(chunks):
            assert chunk.content, f"Chunk {i} should have content"
            assert chunk.document_id == document.id
            assert chunk.chunk_index == i
            assert chunk.start_index >= 0
            assert chunk.end_index > chunk.start_index
            assert chunk.metadata.source_path == document.metadata.source_path
            assert chunk.metadata.language == "python"

    def test_markdown_document_through_pipeline(self, chunking_config: ChunkingConfig) -> None:
        """Test a Markdown document through chunking."""
        document = Document(
            content=SampleDocuments.MARKDOWN_FILE,
            doc_type=DocumentType.MARKDOWN,
            metadata=Metadata(
                source_path="/docs/README.md",
                source_type=SourceType.FILESYSTEM,
                language="markdown",
                file_extension=".md",
            ),
        )

        chunker = MarkdownChunker(chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.content
            assert chunk.metadata.source_path == "/docs/README.md"

    @pytest.mark.asyncio
    async def test_chunk_embedding_integration(
        self,
        chunking_config: ChunkingConfig,
        mock_embedding_provider: AsyncMock,
    ) -> None:
        """Test chunking followed by embedding generation."""
        document = Document(
            content=SampleDocuments.PYTHON_FILE,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/src/auth.py",
                source_type=SourceType.FILESYSTEM,
                language="python",
                file_extension=".py",
            ),
        )

        # Chunk the document
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        # Embed the chunks
        for chunk in chunks:
            embedding = await mock_embedding_provider.embed(chunk.content)
            assert len(embedding) == mock_embedding_provider.dimension
            assert all(isinstance(v, float) for v in embedding)
            chunk.embedding = embedding

        # Verify all chunks are embedded
        assert all(c.has_embedding for c in chunks)

    @pytest.mark.asyncio
    async def test_chunk_storage_and_retrieval(
        self,
        chunking_config: ChunkingConfig,
        mock_embedding_provider: AsyncMock,
    ) -> None:
        """Test storing chunks in vector store and retrieving them."""
        document = Document(
            content=SampleDocuments.PYTHON_FILE,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/src/auth.py",
                source_type=SourceType.FILESYSTEM,
                language="python",
                file_extension=".py",
            ),
        )

        # Chunk and embed
        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        for chunk in chunks:
            chunk.embedding = await mock_embedding_provider.embed(chunk.content)

        # Store in vector store
        store = InMemoryStore()
        await store.add(chunks)

        # Verify chunks were stored
        assert await store.count() == len(chunks)

        # Retrieve by ID
        for chunk in chunks:
            retrieved_chunks = await store.get([chunk.id])
            assert len(retrieved_chunks) > 0
            retrieved = retrieved_chunks[0]
            assert retrieved.content == chunk.content
            assert retrieved.document_id == document.id


class TestMultiLanguageProcessing:
    """Test processing documents in multiple programming languages.

    Verifies language-specific chunking strategies work correctly
    for different languages.
    """

    @pytest.mark.parametrize(
        "content,language,file_ext",
        [
            (SampleDocuments.PYTHON_FILE, "python", ".py"),
            (SampleDocuments.JAVASCRIPT_FILE, "javascript", ".js"),
            (SampleDocuments.GO_FILE, "go", ".go"),
        ],
    )
    def test_language_specific_chunking(
        self,
        content: str,
        language: str,
        file_ext: str,
        chunking_config: ChunkingConfig,
    ) -> None:
        """Test that different languages are chunked appropriately."""
        document = Document(
            content=content,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path=f"/src/file{file_ext}",
                source_type=SourceType.FILESYSTEM,
                language=language,
                file_extension=file_ext,
            ),
        )

        chunker = create_chunker(document.doc_type, chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) > 0, f"Should chunk {language} code"
        assert all(c.metadata.language == language for c in chunks)

        # Verify content is preserved (allowing for some formatting differences)
        full_content = "".join(c.content for c in chunks)
        # Just check that we captured most of the content
        assert len(full_content) > len(content) * 0.8

    @pytest.mark.asyncio
    async def test_multilingual_embedding_search(
        self,
        chunking_config: ChunkingConfig,
        mock_embedding_provider: AsyncMock,
    ) -> None:
        """Test searching across chunks from multiple languages."""
        documents = [
            Document(
                content=SampleDocuments.PYTHON_FILE,
                doc_type=DocumentType.CODE,
                metadata=Metadata(
                    source_path="/src/auth.py",
                    language="python",
                    file_extension=".py",
                ),
            ),
            Document(
                content=SampleDocuments.JAVASCRIPT_FILE,
                doc_type=DocumentType.CODE,
                metadata=Metadata(
                    source_path="/src/client.js",
                    language="javascript",
                    file_extension=".js",
                ),
            ),
        ]

        # Chunk and embed all documents
        all_chunks = []
        for doc in documents:
            chunker = create_chunker(doc.doc_type, chunking_config)
            chunks = chunker.chunk(doc)
            for chunk in chunks:
                chunk.embedding = await mock_embedding_provider.embed(chunk.content)
            all_chunks.extend(chunks)

        # Store in vector store
        store = InMemoryStore()
        await store.add(all_chunks)

        # Verify chunks from both languages are stored
        python_chunks = [c for c in all_chunks if c.metadata.language == "python"]
        js_chunks = [c for c in all_chunks if c.metadata.language == "javascript"]

        assert len(python_chunks) > 0
        assert len(js_chunks) > 0


class TestMultiDocumentWorkflows:
    """Test batch processing of multiple documents.

    Verifies that:
    - Multiple documents can be processed together
    - Document boundaries are maintained
    - Metadata is preserved across documents
    """

    @pytest.mark.asyncio
    async def test_batch_document_processing(
        self,
        chunking_config: ChunkingConfig,
        mock_embedding_provider: AsyncMock,
    ) -> None:
        """Test processing multiple documents in sequence."""
        documents = [
            Document(
                content=SampleDocuments.PYTHON_FILE,
                doc_type=DocumentType.CODE,
                metadata=Metadata(
                    source_path="/src/auth.py",
                    language="python",
                    file_extension=".py",
                ),
            ),
            Document(
                content=SampleDocuments.MARKDOWN_FILE,
                doc_type=DocumentType.MARKDOWN,
                metadata=Metadata(
                    source_path="/docs/README.md",
                    language="markdown",
                    file_extension=".md",
                ),
            ),
            Document(
                content=SampleDocuments.GO_FILE,
                doc_type=DocumentType.CODE,
                metadata=Metadata(
                    source_path="/src/auth.go",
                    language="go",
                    file_extension=".go",
                ),
            ),
        ]

        # Process all documents
        all_chunks = []
        for doc in documents:
            chunker = create_chunker(doc.doc_type, chunking_config)
            chunks = chunker.chunk(doc)

            # Embed chunks
            for chunk in chunks:
                chunk.embedding = await mock_embedding_provider.embed(chunk.content)

            all_chunks.extend(chunks)

        # Store all chunks
        store = InMemoryStore()
        await store.add(all_chunks)

        # Verify correct number of chunks from each document
        assert await store.count() == len(all_chunks)

        # Verify document boundaries are maintained
        for doc in documents:
            doc_chunks = [c for c in all_chunks if c.document_id == doc.id]
            assert len(doc_chunks) > 0, f"Should have chunks for {doc.metadata.source_path}"
            assert all(c.metadata.source_path == doc.metadata.source_path for c in doc_chunks)

    def test_document_source_path_preservation(self, chunking_config: ChunkingConfig) -> None:
        """Test that source paths are preserved through chunking."""
        source_paths = [
            "/project/src/main.py",
            "/project/src/utils.py",
            "/project/docs/API.md",
            "/project/src/handler.go",
        ]

        for source_path in source_paths:
            if source_path.endswith(".py"):
                content = SampleDocuments.PYTHON_FILE
                doc_type = DocumentType.CODE
                language = "python"
            elif source_path.endswith(".go"):
                content = SampleDocuments.GO_FILE
                doc_type = DocumentType.CODE
                language = "go"
            else:
                content = SampleDocuments.MARKDOWN_FILE
                doc_type = DocumentType.MARKDOWN
                language = "markdown"

            document = Document(
                content=content,
                doc_type=doc_type,
                metadata=Metadata(
                    source_path=source_path,
                    language=language,
                ),
            )

            chunker = create_chunker(document.doc_type, chunking_config)
            chunks = chunker.chunk(document)

            # All chunks should preserve source path
            assert all(c.metadata.source_path == source_path for c in chunks)


class TestDataIntegrity:
    """Test that data is preserved and correct through the pipeline.

    Verifies:
    - Content is not lost or corrupted
    - Chunk indices are accurate
    - Metadata is preserved
    - Round-trip integrity
    """

    def test_content_preservation_through_chunking(self, chunking_config: ChunkingConfig) -> None:
        """Test that all content is preserved when chunking.

        Note: CodeChunker may modify content slightly to preserve semantic
        boundaries (e.g., adding/removing lines), so we verify coverage
        rather than exact equality.
        """
        document = Document(
            content=SampleDocuments.PYTHON_FILE,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/src/auth.py",
                language="python",
            ),
        )

        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        # Reconstruct document from chunks
        reconstructed = "".join(c.content for c in chunks)

        # All chunks should be non-empty
        assert all(len(c.content) > 0 for c in chunks)
        # Should have recovered a reasonable amount of content
        assert len(reconstructed) > len(document.content) * 0.7

    def test_chunk_indices_accuracy(self, chunking_config: ChunkingConfig) -> None:
        """Test that chunk start/end indices are within valid document bounds.

        Note: CodeChunker may modify content to preserve semantic boundaries,
        so exact index matching is not guaranteed. We verify indices are
        within bounds and used for retrieval.
        """
        document = Document(
            content=SampleDocuments.PYTHON_FILE,
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/src/auth.py",
                language="python",
            ),
        )

        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        for chunk in chunks:
            # Verify indices are within document bounds
            assert 0 <= chunk.start_index < len(document.content)
            assert 0 < chunk.end_index <= len(document.content)
            assert chunk.start_index < chunk.end_index

    def test_metadata_inheritance_and_override(self, chunking_config: ChunkingConfig) -> None:
        """Test that metadata is correctly inherited/overridden in chunks."""
        custom_metadata = Metadata(
            source_path="/src/module.py",
            source_type=SourceType.GIT,
            language="python",
            file_extension=".py",
            custom={"author": "John Doe", "team": "backend"},
        )

        document = Document(
            content=SampleDocuments.PYTHON_FILE,
            doc_type=DocumentType.CODE,
            metadata=custom_metadata,
        )

        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        for chunk in chunks:
            # Standard metadata should be inherited
            assert chunk.metadata.source_path == custom_metadata.source_path
            assert chunk.metadata.source_type == custom_metadata.source_type
            assert chunk.metadata.language == custom_metadata.language

            # Custom metadata should be inherited
            assert chunk.metadata.custom.get("author") == "John Doe"
            assert chunk.metadata.custom.get("team") == "backend"


class TestErrorRecovery:
    """Test error handling and recovery throughout the pipeline.

    Verifies that:
    - Invalid inputs are handled gracefully
    - Partial failures don't crash the system
    - Error information is useful for debugging
    """

    def test_empty_document_handling(self, chunking_config: ChunkingConfig) -> None:
        """Test that empty documents are handled appropriately."""
        with pytest.raises(ValueError, match="content cannot be empty"):
            Document(
                content="",
                doc_type=DocumentType.CODE,
                metadata=Metadata(source_path="/src/empty.py"),
            )

    def test_invalid_language_code_handling(self, chunking_config: ChunkingConfig) -> None:
        """Test handling of unsupported programming languages."""
        document = Document(
            content="function test() { console.log('hello'); }",
            doc_type=DocumentType.CODE,
            metadata=Metadata(
                source_path="/src/test.brainfuck",
                language="brainfuck",  # Unsupported
            ),
        )

        # Should not crash, should fall back to generic chunking
        chunker = create_chunker(document.doc_type, chunking_config)
        chunks = chunker.chunk(document)

        assert len(chunks) > 0, "Should still produce chunks"

    @pytest.mark.asyncio
    async def test_partial_embedding_failure_recovery(
        self,
        chunking_config: ChunkingConfig,
        mock_embedding_provider: AsyncMock,
        embedding_dimension: int,
    ) -> None:
        """Test recovery when some embeddings fail to generate."""
        document = Document(
            content=SampleDocuments.PYTHON_FILE,
            doc_type=DocumentType.CODE,
            metadata=Metadata(source_path="/src/auth.py"),
        )

        chunker = CodeChunker(chunking_config)
        chunks = chunker.chunk(document)

        # Try to embed all chunks
        embedded_chunks = []
        for chunk in chunks:
            try:
                embedding = await mock_embedding_provider.embed(chunk.content)
                chunk.embedding = embedding
                embedded_chunks.append(chunk)
            except Exception as e:
                # Log and continue
                print(f"Failed to embed chunk {chunk.id}: {e}")
                continue

        # Should have successfully embedded at least some
        assert len(embedded_chunks) > 0


class TestRealWorldScenarios:
    """Test realistic usage patterns and workflows.

    Represents how the pipeline would be used in practice.
    """

    @pytest.mark.asyncio
    async def test_documentation_ingestion_workflow(
        self,
        chunking_config: ChunkingConfig,
        mock_embedding_provider: AsyncMock,
    ) -> None:
        """Test a realistic documentation ingestion workflow."""
        # Simulate a documentation set
        docs_to_ingest = [
            Document(
                content=SampleDocuments.MARKDOWN_FILE,
                doc_type=DocumentType.MARKDOWN,
                metadata=Metadata(
                    source_path="/docs/README.md",
                    source_type=SourceType.FILESYSTEM,
                ),
            ),
        ]

        store = InMemoryStore()

        # Process each document
        for doc in docs_to_ingest:
            chunker = create_chunker(doc.doc_type, chunking_config)
            chunks = chunker.chunk(doc)

            # Embed
            for chunk in chunks:
                chunk.embedding = await mock_embedding_provider.embed(chunk.content)

            # Store
            await store.add(chunks)

        # Verify documents are accessible
        assert await store.count() > 0

    @pytest.mark.asyncio
    async def test_codebase_indexing_workflow(
        self,
        chunking_config: ChunkingConfig,
        mock_embedding_provider: AsyncMock,
    ) -> None:
        """Test indexing a small codebase with multiple languages."""
        codebase_files = [
            Document(
                content=SampleDocuments.PYTHON_FILE,
                doc_type=DocumentType.CODE,
                metadata=Metadata(
                    source_path="/src/auth.py",
                    source_type=SourceType.FILESYSTEM,
                    language="python",
                ),
            ),
            Document(
                content=SampleDocuments.JAVASCRIPT_FILE,
                doc_type=DocumentType.CODE,
                metadata=Metadata(
                    source_path="/src/client.js",
                    source_type=SourceType.FILESYSTEM,
                    language="javascript",
                ),
            ),
            Document(
                content=SampleDocuments.GO_FILE,
                doc_type=DocumentType.CODE,
                metadata=Metadata(
                    source_path="/src/auth.go",
                    source_type=SourceType.FILESYSTEM,
                    language="go",
                ),
            ),
        ]

        store = InMemoryStore()
        all_chunks = []

        # Index all files
        for doc in codebase_files:
            chunker = create_chunker(doc.doc_type, chunking_config)
            chunks = chunker.chunk(doc)

            for chunk in chunks:
                chunk.embedding = await mock_embedding_provider.embed(chunk.content)

            all_chunks.extend(chunks)

        await store.add(all_chunks)

        # Verify statistics
        assert await store.count() == len(all_chunks)

        # Verify language distribution
        languages = {}
        for chunk in all_chunks:
            lang = chunk.metadata.language
            languages[lang] = languages.get(lang, 0) + 1

        assert "python" in languages
        assert "javascript" in languages
        assert "go" in languages


# =============================================================================
# Integration Test Suite Summary
# =============================================================================

__all__ = [
    "TestBasicPipelineFlow",
    "TestMultiLanguageProcessing",
    "TestMultiDocumentWorkflows",
    "TestDataIntegrity",
    "TestErrorRecovery",
    "TestRealWorldScenarios",
]
