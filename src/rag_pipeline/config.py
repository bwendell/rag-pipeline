"""Central configuration for the RAG pipeline using Pydantic Settings.

Environment variables with RAG_ prefix override defaults.
Example: RAG_CHUNK_SIZE=500 sets chunk_size to 500.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from rag_pipeline.core.base_chunker import ChunkingConfig


class RAGSettings(BaseSettings):
    """Configuration loaded from environment variables with RAG_ prefix."""

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    chunk_size: int = Field(default=1000, ge=100, le=10000)
    chunk_overlap: int = Field(default=200, ge=0, le=5000)
    min_chunk_size: int = Field(default=100, ge=10, le=1000)
    max_chunk_size: int = Field(default=2000, ge=500, le=20000)

    vector_store_type: Literal["chroma", "oci"] = "chroma"
    chroma_persist_directory: str = "./data/chroma"
    chroma_collection_name: str = "rag_documents"

    llm_provider_type: Literal["ollama", "oci", "openai"] = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "mistral"

    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    embedding_provider_type: Literal["sentence-transformers", "openai", "oci"] = (
        "sentence-transformers"
    )
    embedding_device: str | None = None  # None means auto-detect (cuda if available, else cpu)
    embedding_normalize: bool = True
    embedding_batch_size: int = Field(default=32, ge=1, le=512)

    top_k: int = Field(default=5, ge=1, le=50)
    similarity_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    def to_chunking_config(self) -> ChunkingConfig:
        """Create a ChunkingConfig from these settings."""
        return ChunkingConfig(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            min_chunk_size=self.min_chunk_size,
            max_chunk_size=self.max_chunk_size,
        )

    def to_embedding_config(self) -> dict[str, Any]:
        """Create embedding provider configuration from these settings."""
        return {
            "model_name": self.embedding_model,
            "device": self.embedding_device,
            "normalize_embeddings": self.embedding_normalize,
            "batch_size": self.embedding_batch_size,
        }


@lru_cache
def get_settings() -> RAGSettings:
    """Get cached settings instance. Use for application-wide config."""
    return RAGSettings()
