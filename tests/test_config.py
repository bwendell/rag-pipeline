"""Tests for RAG pipeline configuration management.

Tests cover:
1. Default Values: Verify all defaults match specification
2. Environment Variables: Test RAG_ prefix loading
3. Validation: Boundary conditions and constraints
4. Config Methods: to_chunking_config() and to_embedding_config()
5. Settings Caching: Verify lru_cache behavior
6. Type Conversion: Literal types and complex conversions
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from rag_pipeline.config import RAGSettings, get_settings
from rag_pipeline.core.base_chunker import ChunkingConfig


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def clear_settings_cache():
    """Clear the lru_cache for get_settings() to allow fresh instances."""
    yield
    get_settings.cache_clear()


@pytest.fixture
def clean_env():
    """Remove all RAG_ prefixed environment variables."""
    original_env = {}
    rag_keys = [k for k in os.environ.keys() if k.startswith("RAG_")]
    for key in rag_keys:
        original_env[key] = os.environ.pop(key)
    yield
    # Restore original environment
    for key, value in original_env.items():
        os.environ[key] = value


# =============================================================================
# Tests: Default Values
# =============================================================================


class TestDefaultValues:
    """Verify all default configuration values."""

    def test_default_chunk_size(self) -> None:
        """Test chunk_size default is 1000."""
        settings = RAGSettings()
        assert settings.chunk_size == 1000

    def test_default_chunk_overlap(self) -> None:
        """Test chunk_overlap default is 200."""
        settings = RAGSettings()
        assert settings.chunk_overlap == 200

    def test_default_min_chunk_size(self) -> None:
        """Test min_chunk_size default is 100."""
        settings = RAGSettings()
        assert settings.min_chunk_size == 100

    def test_default_max_chunk_size(self) -> None:
        """Test max_chunk_size default is 2000."""
        settings = RAGSettings()
        assert settings.max_chunk_size == 2000

    def test_default_vector_store_type(self) -> None:
        """Test vector_store_type default is 'chroma'."""
        settings = RAGSettings()
        assert settings.vector_store_type == "chroma"

    def test_default_chroma_persist_directory(self) -> None:
        """Test chroma_persist_directory default path."""
        settings = RAGSettings()
        assert settings.chroma_persist_directory == "./data/chroma"

    def test_default_chroma_collection_name(self) -> None:
        """Test chroma_collection_name default."""
        settings = RAGSettings()
        assert settings.chroma_collection_name == "rag_documents"

    def test_default_llm_provider_type(self) -> None:
        """Test llm_provider_type default is 'ollama'."""
        settings = RAGSettings()
        assert settings.llm_provider_type == "ollama"

    def test_default_ollama_base_url(self) -> None:
        """Test ollama_base_url default."""
        settings = RAGSettings()
        assert settings.ollama_base_url == "http://localhost:11434"

    def test_default_ollama_model(self) -> None:
        """Test ollama_model default is 'mistral'."""
        settings = RAGSettings()
        assert settings.ollama_model == "mistral"

    def test_default_embedding_model(self) -> None:
        """Test embedding_model default."""
        settings = RAGSettings()
        assert settings.embedding_model == "all-MiniLM-L6-v2"

    def test_default_embedding_dimension(self) -> None:
        """Test embedding_dimension default is 384."""
        settings = RAGSettings()
        assert settings.embedding_dimension == 384

    def test_default_embedding_provider_type(self) -> None:
        """Test embedding_provider_type default is 'sentence-transformers'."""
        settings = RAGSettings()
        assert settings.embedding_provider_type == "sentence-transformers"

    def test_default_embedding_device(self) -> None:
        """Test embedding_device default is None (auto-detect)."""
        settings = RAGSettings()
        assert settings.embedding_device is None

    def test_default_embedding_normalize(self) -> None:
        """Test embedding_normalize default is True."""
        settings = RAGSettings()
        assert settings.embedding_normalize is True

    def test_default_embedding_batch_size(self) -> None:
        """Test embedding_batch_size default is 32."""
        settings = RAGSettings()
        assert settings.embedding_batch_size == 32

    def test_default_top_k(self) -> None:
        """Test top_k default is 5."""
        settings = RAGSettings()
        assert settings.top_k == 5

    def test_default_similarity_threshold(self) -> None:
        """Test similarity_threshold default is 0.5."""
        settings = RAGSettings()
        assert settings.similarity_threshold == 0.5

    def test_default_log_level(self) -> None:
        """Test log_level default is 'INFO'."""
        settings = RAGSettings()
        assert settings.log_level == "INFO"


# =============================================================================
# Tests: Environment Variable Loading
# =============================================================================


class TestEnvironmentVariableLoading:
    """Test loading configuration from environment variables with RAG_ prefix."""

    def test_load_chunk_size_from_env(self, clean_env: None) -> None:
        """Test RAG_CHUNK_SIZE environment variable is loaded."""
        os.environ["RAG_CHUNK_SIZE"] = "500"
        settings = RAGSettings()
        assert settings.chunk_size == 500

    def test_load_chunk_overlap_from_env(self, clean_env: None) -> None:
        """Test RAG_CHUNK_OVERLAP environment variable is loaded."""
        os.environ["RAG_CHUNK_OVERLAP"] = "100"
        settings = RAGSettings()
        assert settings.chunk_overlap == 100

    def test_load_llm_provider_type_from_env(self, clean_env: None) -> None:
        """Test RAG_LLM_PROVIDER_TYPE environment variable is loaded."""
        os.environ["RAG_LLM_PROVIDER_TYPE"] = "openai"
        settings = RAGSettings()
        assert settings.llm_provider_type == "openai"

    def test_load_ollama_model_from_env(self, clean_env: None) -> None:
        """Test RAG_OLLAMA_MODEL environment variable is loaded."""
        os.environ["RAG_OLLAMA_MODEL"] = "llama2"
        settings = RAGSettings()
        assert settings.ollama_model == "llama2"

    def test_load_embedding_model_from_env(self, clean_env: None) -> None:
        """Test RAG_EMBEDDING_MODEL environment variable is loaded."""
        os.environ["RAG_EMBEDDING_MODEL"] = "all-mpnet-base-v2"
        settings = RAGSettings()
        assert settings.embedding_model == "all-mpnet-base-v2"

    def test_load_vector_store_type_from_env(self, clean_env: None) -> None:
        """Test RAG_VECTOR_STORE_TYPE environment variable is loaded."""
        os.environ["RAG_VECTOR_STORE_TYPE"] = "oci"
        settings = RAGSettings()
        assert settings.vector_store_type == "oci"

    def test_load_embedding_device_from_env(self, clean_env: None) -> None:
        """Test RAG_EMBEDDING_DEVICE environment variable is loaded."""
        os.environ["RAG_EMBEDDING_DEVICE"] = "cuda"
        settings = RAGSettings()
        assert settings.embedding_device == "cuda"

    def test_load_embedding_batch_size_from_env(self, clean_env: None) -> None:
        """Test RAG_EMBEDDING_BATCH_SIZE environment variable is loaded."""
        os.environ["RAG_EMBEDDING_BATCH_SIZE"] = "64"
        settings = RAGSettings()
        assert settings.embedding_batch_size == 64

    def test_load_top_k_from_env(self, clean_env: None) -> None:
        """Test RAG_TOP_K environment variable is loaded."""
        os.environ["RAG_TOP_K"] = "10"
        settings = RAGSettings()
        assert settings.top_k == 10

    def test_load_similarity_threshold_from_env(self, clean_env: None) -> None:
        """Test RAG_SIMILARITY_THRESHOLD environment variable is loaded."""
        os.environ["RAG_SIMILARITY_THRESHOLD"] = "0.7"
        settings = RAGSettings()
        assert settings.similarity_threshold == 0.7

    def test_load_log_level_from_env(self, clean_env: None) -> None:
        """Test RAG_LOG_LEVEL environment variable is loaded."""
        os.environ["RAG_LOG_LEVEL"] = "DEBUG"
        settings = RAGSettings()
        assert settings.log_level == "DEBUG"

    def test_multiple_env_vars_loaded(self, clean_env: None) -> None:
        """Test multiple environment variables are all loaded together."""
        os.environ["RAG_CHUNK_SIZE"] = "1500"
        os.environ["RAG_OLLAMA_MODEL"] = "neural-chat"
        os.environ["RAG_TOP_K"] = "3"
        settings = RAGSettings()
        assert settings.chunk_size == 1500
        assert settings.ollama_model == "neural-chat"
        assert settings.top_k == 3

    def test_non_rag_env_vars_ignored(self, clean_env: None) -> None:
        """Test that non-RAG_ prefixed environment variables are ignored."""
        os.environ["OTHER_VARIABLE"] = "value"
        os.environ["CHUNK_SIZE"] = "999"
        settings = RAGSettings()
        # Should use defaults, not the non-prefixed env vars
        assert settings.chunk_size == 1000


# =============================================================================
# Tests: Validation & Constraints
# =============================================================================


class TestValidation:
    """Test configuration validation and boundary constraints."""

    def test_chunk_size_minimum_boundary(self) -> None:
        """Test chunk_size minimum constraint (ge=100)."""
        with pytest.raises(ValidationError) as exc_info:
            RAGSettings(chunk_size=99)
        assert "greater than or equal to 100" in str(exc_info.value)

    def test_chunk_size_maximum_boundary(self) -> None:
        """Test chunk_size maximum constraint (le=10000)."""
        with pytest.raises(ValidationError) as exc_info:
            RAGSettings(chunk_size=10001)
        assert "less than or equal to 10000" in str(exc_info.value)

    def test_chunk_size_valid_minimum(self) -> None:
        """Test chunk_size at minimum boundary is valid."""
        settings = RAGSettings(chunk_size=100)
        assert settings.chunk_size == 100

    def test_chunk_size_valid_maximum(self) -> None:
        """Test chunk_size at maximum boundary is valid."""
        settings = RAGSettings(chunk_size=10000)
        assert settings.chunk_size == 10000

    def test_chunk_overlap_minimum_boundary(self) -> None:
        """Test chunk_overlap minimum constraint (ge=0)."""
        with pytest.raises(ValidationError) as exc_info:
            RAGSettings(chunk_overlap=-1)
        assert "greater than or equal to 0" in str(exc_info.value)

    def test_chunk_overlap_maximum_boundary(self) -> None:
        """Test chunk_overlap maximum constraint (le=5000)."""
        with pytest.raises(ValidationError) as exc_info:
            RAGSettings(chunk_overlap=5001)
        assert "less than or equal to 5000" in str(exc_info.value)

    def test_chunk_overlap_zero_valid(self) -> None:
        """Test chunk_overlap can be zero."""
        settings = RAGSettings(chunk_overlap=0)
        assert settings.chunk_overlap == 0

    def test_min_chunk_size_boundary(self) -> None:
        """Test min_chunk_size constraints."""
        with pytest.raises(ValidationError):
            RAGSettings(min_chunk_size=9)
        with pytest.raises(ValidationError):
            RAGSettings(min_chunk_size=1001)

    def test_max_chunk_size_boundary(self) -> None:
        """Test max_chunk_size constraints."""
        with pytest.raises(ValidationError):
            RAGSettings(max_chunk_size=499)
        with pytest.raises(ValidationError):
            RAGSettings(max_chunk_size=20001)

    def test_embedding_batch_size_minimum(self) -> None:
        """Test embedding_batch_size minimum constraint (ge=1)."""
        with pytest.raises(ValidationError):
            RAGSettings(embedding_batch_size=0)

    def test_embedding_batch_size_maximum(self) -> None:
        """Test embedding_batch_size maximum constraint (le=512)."""
        with pytest.raises(ValidationError):
            RAGSettings(embedding_batch_size=513)

    def test_top_k_minimum(self) -> None:
        """Test top_k minimum constraint (ge=1)."""
        with pytest.raises(ValidationError):
            RAGSettings(top_k=0)

    def test_top_k_maximum(self) -> None:
        """Test top_k maximum constraint (le=50)."""
        with pytest.raises(ValidationError):
            RAGSettings(top_k=51)

    def test_similarity_threshold_minimum(self) -> None:
        """Test similarity_threshold minimum constraint (ge=0.0)."""
        settings = RAGSettings(similarity_threshold=0.0)
        assert settings.similarity_threshold == 0.0

    def test_similarity_threshold_maximum(self) -> None:
        """Test similarity_threshold maximum constraint (le=1.0)."""
        settings = RAGSettings(similarity_threshold=1.0)
        assert settings.similarity_threshold == 1.0

    def test_similarity_threshold_below_minimum(self) -> None:
        """Test similarity_threshold below minimum raises."""
        with pytest.raises(ValidationError):
            RAGSettings(similarity_threshold=-0.1)

    def test_similarity_threshold_above_maximum(self) -> None:
        """Test similarity_threshold above maximum raises."""
        with pytest.raises(ValidationError):
            RAGSettings(similarity_threshold=1.1)


# =============================================================================
# Tests: Literal Type Validation
# =============================================================================


class TestLiteralTypeValidation:
    """Test validation of Literal type fields."""

    def test_valid_vector_store_types(self) -> None:
        """Test all valid vector_store_type values."""
        for store_type in ["chroma", "oci"]:
            settings = RAGSettings(vector_store_type=store_type)  # type: ignore
            assert settings.vector_store_type == store_type

    def test_invalid_vector_store_type(self) -> None:
        """Test invalid vector_store_type raises validation error."""
        with pytest.raises(ValidationError):
            RAGSettings(vector_store_type="postgresql")  # type: ignore

    def test_valid_llm_provider_types(self) -> None:
        """Test all valid llm_provider_type values."""
        for provider_type in ["ollama", "oci", "openai"]:
            settings = RAGSettings(llm_provider_type=provider_type)  # type: ignore
            assert settings.llm_provider_type == provider_type

    def test_invalid_llm_provider_type(self) -> None:
        """Test invalid llm_provider_type raises validation error."""
        with pytest.raises(ValidationError):
            RAGSettings(llm_provider_type="anthropic")  # type: ignore

    def test_valid_embedding_provider_types(self) -> None:
        """Test all valid embedding_provider_type values."""
        for provider_type in ["sentence-transformers", "openai", "oci"]:
            settings = RAGSettings(embedding_provider_type=provider_type)  # type: ignore
            assert settings.embedding_provider_type == provider_type

    def test_invalid_embedding_provider_type(self) -> None:
        """Test invalid embedding_provider_type raises validation error."""
        with pytest.raises(ValidationError):
            RAGSettings(embedding_provider_type="huggingface")  # type: ignore

    def test_valid_log_levels(self) -> None:
        """Test all valid log_level values."""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            settings = RAGSettings(log_level=level)  # type: ignore
            assert settings.log_level == level

    def test_invalid_log_level(self) -> None:
        """Test invalid log_level raises validation error."""
        with pytest.raises(ValidationError):
            RAGSettings(log_level="CRITICAL")  # type: ignore


# =============================================================================
# Tests: Config Conversion Methods
# =============================================================================


class TestConfigMethods:
    """Test configuration conversion helper methods."""

    def test_to_chunking_config(self) -> None:
        """Test to_chunking_config() creates proper ChunkingConfig."""
        settings = RAGSettings(
            chunk_size=1500, chunk_overlap=250, min_chunk_size=150, max_chunk_size=2500
        )
        chunking_config = settings.to_chunking_config()

        assert isinstance(chunking_config, ChunkingConfig)
        assert chunking_config.chunk_size == 1500
        assert chunking_config.chunk_overlap == 250
        assert chunking_config.min_chunk_size == 150
        assert chunking_config.max_chunk_size == 2500

    def test_to_chunking_config_with_defaults(self, clean_env: None) -> None:
        """Test to_chunking_config() with default values."""
        settings = RAGSettings()
        chunking_config = settings.to_chunking_config()

        assert chunking_config.chunk_size == 1000
        assert chunking_config.chunk_overlap == 200
        assert chunking_config.min_chunk_size == 100
        assert chunking_config.max_chunk_size == 2000

    def test_to_embedding_config(self) -> None:
        """Test to_embedding_config() creates proper dict."""
        settings = RAGSettings(
            embedding_model="test-model",
            embedding_device="cuda",
            embedding_normalize=False,
            embedding_batch_size=64,
        )
        config = settings.to_embedding_config()

        assert isinstance(config, dict)
        assert config["model_name"] == "test-model"
        assert config["device"] == "cuda"
        assert config["normalize_embeddings"] is False
        assert config["batch_size"] == 64

    def test_to_embedding_config_with_defaults(self, clean_env: None) -> None:
        """Test to_embedding_config() with default values."""
        settings = RAGSettings()
        config = settings.to_embedding_config()

        assert config["model_name"] == "all-MiniLM-L6-v2"
        assert config["device"] is None
        assert config["normalize_embeddings"] is True
        assert config["batch_size"] == 32

    def test_to_embedding_config_returns_dict(self) -> None:
        """Test to_embedding_config() returns dict, not other type."""
        settings = RAGSettings()
        config = settings.to_embedding_config()
        assert isinstance(config, dict)
        assert all(isinstance(k, str) for k in config.keys())


# =============================================================================
# Tests: Settings Caching
# =============================================================================


class TestSettingsCaching:
    """Test get_settings() caching behavior."""

    def test_get_settings_returns_rag_settings(self) -> None:
        """Test get_settings() returns RAGSettings instance."""
        settings = get_settings()
        assert isinstance(settings, RAGSettings)

    def test_get_settings_is_cached(self, clear_settings_cache: None) -> None:
        """Test get_settings() returns same instance on repeated calls."""
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2

    def test_get_settings_cache_can_be_cleared(self, clear_settings_cache: None) -> None:
        """Test cache can be cleared to get a fresh instance."""
        settings1 = get_settings()
        get_settings.cache_clear()
        settings2 = get_settings()
        assert settings1 is not settings2

    def test_get_settings_returns_default_config(
        self, clean_env: None, clear_settings_cache: None
    ) -> None:
        """Test get_settings() uses defaults when no env vars set."""
        settings = get_settings()
        assert settings.chunk_size == 1000
        assert settings.llm_provider_type == "ollama"

    def test_get_settings_respects_env_vars(
        self, clean_env: None, clear_settings_cache: None
    ) -> None:
        """Test get_settings() respects environment variables."""
        os.environ["RAG_CHUNK_SIZE"] = "2000"
        get_settings.cache_clear()
        settings = get_settings()
        assert settings.chunk_size == 2000


# =============================================================================
# Tests: Type Handling
# =============================================================================


class TestTypeHandling:
    """Test correct type handling for various field types."""

    def test_integer_fields_accept_ints(self) -> None:
        """Test integer fields accept integer values."""
        settings = RAGSettings(
            chunk_size=1000,
            chunk_overlap=200,
            embedding_batch_size=32,
            top_k=5,
        )
        assert isinstance(settings.chunk_size, int)
        assert isinstance(settings.chunk_overlap, int)
        assert isinstance(settings.embedding_batch_size, int)
        assert isinstance(settings.top_k, int)

    def test_integer_fields_coerce_from_strings(self, clean_env: None) -> None:
        """Test integer fields coerce from string environment variables."""
        os.environ["RAG_CHUNK_SIZE"] = "1500"
        os.environ["RAG_TOP_K"] = "10"
        settings = RAGSettings()
        assert isinstance(settings.chunk_size, int)
        assert isinstance(settings.top_k, int)
        assert settings.chunk_size == 1500
        assert settings.top_k == 10

    def test_float_field_accepts_floats(self) -> None:
        """Test float fields accept float values."""
        settings = RAGSettings(similarity_threshold=0.75)
        assert isinstance(settings.similarity_threshold, float)
        assert settings.similarity_threshold == 0.75

    def test_float_field_coerces_from_int(self) -> None:
        """Test float fields coerce from int values."""
        settings = RAGSettings(similarity_threshold=1)
        assert isinstance(settings.similarity_threshold, float)
        assert settings.similarity_threshold == 1.0

    def test_optional_string_field(self) -> None:
        """Test optional string field (embedding_device)."""
        settings1 = RAGSettings(embedding_device=None)
        assert settings1.embedding_device is None

        settings2 = RAGSettings(embedding_device="cpu")
        assert settings2.embedding_device == "cpu"

    def test_string_field_as_path(self) -> None:
        """Test string field used as path (chroma_persist_directory)."""
        settings = RAGSettings(chroma_persist_directory="/custom/path/chroma")
        assert settings.chroma_persist_directory == "/custom/path/chroma"


# =============================================================================
# Tests: Extra Configuration Handling
# =============================================================================


class TestExtraConfigHandling:
    """Test handling of extra/unknown configuration fields."""

    def test_extra_fields_ignored(self, clean_env: None) -> None:
        """Test that extra unknown fields are ignored (extra='ignore')."""
        # Should not raise an error
        settings = RAGSettings(unknown_field="value")  # type: ignore
        assert settings.chunk_size == 1000  # Should use defaults

    def test_extra_env_vars_ignored(self, clean_env: None) -> None:
        """Test that extra unknown env vars are ignored."""
        os.environ["RAG_UNKNOWN_SETTING"] = "some_value"
        # Should not raise an error
        settings = RAGSettings()
        assert settings.chunk_size == 1000  # Should use defaults


# =============================================================================
# Tests: Cross-Field Consistency
# =============================================================================


class TestCrossFieldConsistency:
    """Test relationships and consistency between fields."""

    def test_min_chunk_less_than_max_chunk(self) -> None:
        """Test creating config with min < max chunk size."""
        # While Pydantic doesn't enforce this, it makes semantic sense
        settings = RAGSettings(min_chunk_size=100, max_chunk_size=2000)
        assert settings.min_chunk_size < settings.max_chunk_size

    def test_chunk_size_between_min_and_max(self) -> None:
        """Test typical chunk_size is between min and max."""
        settings = RAGSettings()
        assert settings.min_chunk_size <= settings.chunk_size <= settings.max_chunk_size

    def test_chunk_overlap_less_than_chunk_size(self) -> None:
        """Test typical chunk_overlap is less than chunk_size."""
        settings = RAGSettings()
        assert settings.chunk_overlap < settings.chunk_size

    def test_custom_values_with_custom_embedding_config(self) -> None:
        """Test embedding config is generated correctly from custom settings."""
        settings = RAGSettings(
            embedding_model="custom-model",
            embedding_device="cuda",
            embedding_normalize=False,
            embedding_batch_size=128,
        )
        config = settings.to_embedding_config()

        assert config["model_name"] == "custom-model"
        assert config["device"] == "cuda"
        assert config["normalize_embeddings"] is False
        assert config["batch_size"] == 128


# =============================================================================
# Tests: Integration Scenarios
# =============================================================================


class TestIntegrationScenarios:
    """Test realistic configuration scenarios."""

    def test_development_environment_config(self, clean_env: None) -> None:
        """Test typical development environment configuration."""
        os.environ["RAG_LOG_LEVEL"] = "DEBUG"
        os.environ["RAG_OLLAMA_MODEL"] = "neural-chat"
        os.environ["RAG_VECTOR_STORE_TYPE"] = "chroma"
        settings = RAGSettings()

        assert settings.log_level == "DEBUG"
        assert settings.ollama_model == "neural-chat"
        assert settings.vector_store_type == "chroma"

    def test_production_environment_config(self, clean_env: None) -> None:
        """Test typical production environment configuration."""
        os.environ["RAG_LOG_LEVEL"] = "ERROR"
        os.environ["RAG_LLM_PROVIDER_TYPE"] = "oci"
        os.environ["RAG_VECTOR_STORE_TYPE"] = "oci"
        os.environ["RAG_EMBEDDING_PROVIDER_TYPE"] = "oci"
        settings = RAGSettings()

        assert settings.log_level == "ERROR"
        assert settings.llm_provider_type == "oci"
        assert settings.vector_store_type == "oci"
        assert settings.embedding_provider_type == "oci"

    def test_high_precision_retrieval_config(self) -> None:
        """Test configuration for high-precision retrieval."""
        settings = RAGSettings(
            top_k=3,
            similarity_threshold=0.8,
            chunk_overlap=400,  # More overlap for better context
        )

        assert settings.top_k == 3
        assert settings.similarity_threshold == 0.8
        assert settings.chunk_overlap == 400

    def test_large_scale_processing_config(self) -> None:
        """Test configuration for large-scale document processing."""
        settings = RAGSettings(
            chunk_size=2000,
            embedding_batch_size=256,
            top_k=20,
        )

        assert settings.chunk_size == 2000
        assert settings.embedding_batch_size == 256
        assert settings.top_k == 20

    def test_config_to_all_helper_methods(self) -> None:
        """Test creating all config objects from RAGSettings."""
        settings = RAGSettings()

        chunking = settings.to_chunking_config()
        embedding = settings.to_embedding_config()

        assert isinstance(chunking, ChunkingConfig)
        assert isinstance(embedding, dict)
        assert len(embedding) == 4  # 4 keys in embedding config
