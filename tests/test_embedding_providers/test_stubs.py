"""Unit tests for embedding provider stubs."""

from __future__ import annotations

import pytest

from rag_pipeline.embedding_providers.stubs import (
    OCIEmbeddingProvider,
    OpenAIEmbeddingProvider,
)
from rag_pipeline.embedding_providers.base import AbstractEmbeddingProvider


class TestOCIEmbeddingProvider:
    """Tests for OCI embedding provider stub."""

    def test_instantiation(self):
        """Test provider can be instantiated."""
        provider = OCIEmbeddingProvider()

        assert provider is not None
        assert provider.model_name == "cohere.embed-english-v3.0"

    def test_instantiation_with_model_name(self):
        """Test provider with custom model name."""
        provider = OCIEmbeddingProvider(model_name="custom-model")

        assert provider.model_name == "custom-model"

    def test_instantiation_with_compartment_id(self):
        """Test provider accepts compartment_id parameter."""
        compartment_id = "ocid1.compartment.oc1..xxx"
        provider = OCIEmbeddingProvider(compartment_id=compartment_id)

        assert provider.compartment_id == compartment_id

    def test_instantiation_with_dimension(self):
        """Test provider accepts dimension parameter."""
        provider = OCIEmbeddingProvider(dimension=1024)

        assert provider.get_embedding_dimension() == 1024

    def test_instantiation_with_all_params(self):
        """Test provider accepts all parameters."""
        compartment_id = "ocid1.compartment.oc1..xxx"
        provider = OCIEmbeddingProvider(
            model_name="custom-model",
            compartment_id=compartment_id,
            dimension=2048,
        )

        assert provider.model_name == "custom-model"
        assert provider.compartment_id == compartment_id
        assert provider.get_embedding_dimension() == 2048

    def test_get_provider_name(self):
        """Test provider name is correct."""
        provider = OCIEmbeddingProvider()
        assert provider._get_provider_name() == "oci-genai"

    def test_get_embedding_dimension(self):
        """Test embedding dimension is returned."""
        provider = OCIEmbeddingProvider(dimension=1024)
        assert provider.get_embedding_dimension() == 1024

    def test_default_dimension(self):
        """Test default embedding dimension."""
        provider = OCIEmbeddingProvider()
        assert provider.get_embedding_dimension() == 1024

    @pytest.mark.anyio
    async def test_embed_raises_not_implemented(self):
        """Test embed raises NotImplementedError."""
        provider = OCIEmbeddingProvider()

        with pytest.raises(NotImplementedError) as exc_info:
            await provider.embed("test text")

        assert "stub" in str(exc_info.value).lower()

    @pytest.mark.anyio
    async def test_embed_batch_raises_not_implemented(self):
        """Test embed_batch raises NotImplementedError."""
        provider = OCIEmbeddingProvider()

        with pytest.raises(NotImplementedError) as exc_info:
            await provider.embed_batch(["text1", "text2"])

        assert "stub" in str(exc_info.value).lower()

    @pytest.mark.anyio
    async def test_embed_raises_meaningful_error_message(self):
        """Test embed raises error with meaningful message."""
        provider = OCIEmbeddingProvider()

        with pytest.raises(NotImplementedError) as exc_info:
            await provider.embed("test")

        error_msg = str(exc_info.value).lower()
        assert "oci" in error_msg or "stub" in error_msg

    @pytest.mark.anyio
    async def test_embed_batch_raises_meaningful_error_message(self):
        """Test embed_batch raises error with meaningful message."""
        provider = OCIEmbeddingProvider()

        with pytest.raises(NotImplementedError) as exc_info:
            await provider.embed_batch(["test1", "test2"])

        error_msg = str(exc_info.value).lower()
        assert "oci" in error_msg or "stub" in error_msg


class TestOpenAIEmbeddingProvider:
    """Tests for OpenAI embedding provider stub."""

    def test_instantiation(self):
        """Test provider can be instantiated."""
        provider = OpenAIEmbeddingProvider()

        assert provider is not None
        assert provider.model_name == "text-embedding-ada-002"

    def test_instantiation_with_model_name(self):
        """Test provider with custom model name."""
        provider = OpenAIEmbeddingProvider(model_name="text-embedding-3-small")

        assert provider.model_name == "text-embedding-3-small"

    def test_instantiation_with_api_key(self):
        """Test provider accepts api_key parameter."""
        api_key = "sk-test123"
        provider = OpenAIEmbeddingProvider(api_key=api_key)

        assert provider.api_key == api_key

    def test_instantiation_with_dimension(self):
        """Test provider accepts dimension parameter."""
        provider = OpenAIEmbeddingProvider(dimension=3072)

        assert provider.get_embedding_dimension() == 3072

    def test_instantiation_with_all_params(self):
        """Test provider accepts all parameters."""
        api_key = "sk-test123"
        provider = OpenAIEmbeddingProvider(
            model_name="text-embedding-3-small",
            api_key=api_key,
            dimension=1536,
        )

        assert provider.model_name == "text-embedding-3-small"
        assert provider.api_key == api_key
        assert provider.get_embedding_dimension() == 1536

    def test_get_provider_name(self):
        """Test provider name is correct."""
        provider = OpenAIEmbeddingProvider()
        assert provider._get_provider_name() == "openai"

    def test_get_embedding_dimension(self):
        """Test embedding dimension is returned."""
        provider = OpenAIEmbeddingProvider(dimension=1536)
        assert provider.get_embedding_dimension() == 1536

    def test_default_dimension(self):
        """Test default embedding dimension."""
        provider = OpenAIEmbeddingProvider()
        assert provider.get_embedding_dimension() == 1536

    @pytest.mark.anyio
    async def test_embed_raises_not_implemented(self):
        """Test embed raises NotImplementedError."""
        provider = OpenAIEmbeddingProvider()

        with pytest.raises(NotImplementedError) as exc_info:
            await provider.embed("test text")

        assert "stub" in str(exc_info.value).lower()

    @pytest.mark.anyio
    async def test_embed_batch_raises_not_implemented(self):
        """Test embed_batch raises NotImplementedError."""
        provider = OpenAIEmbeddingProvider()

        with pytest.raises(NotImplementedError) as exc_info:
            await provider.embed_batch(["text1", "text2"])

        assert "stub" in str(exc_info.value).lower()

    @pytest.mark.anyio
    async def test_embed_raises_meaningful_error_message(self):
        """Test embed raises error with meaningful message."""
        provider = OpenAIEmbeddingProvider()

        with pytest.raises(NotImplementedError) as exc_info:
            await provider.embed("test")

        error_msg = str(exc_info.value).lower()
        assert "openai" in error_msg or "stub" in error_msg

    @pytest.mark.anyio
    async def test_embed_batch_raises_meaningful_error_message(self):
        """Test embed_batch raises error with meaningful message."""
        provider = OpenAIEmbeddingProvider()

        with pytest.raises(NotImplementedError) as exc_info:
            await provider.embed_batch(["test1", "test2"])

        error_msg = str(exc_info.value).lower()
        assert "openai" in error_msg or "stub" in error_msg


class TestStubsInheritance:
    """Tests for stub inheritance from AbstractEmbeddingProvider."""

    def test_oci_inherits_from_abstract(self):
        """Test OCI stub inherits from AbstractEmbeddingProvider."""
        provider = OCIEmbeddingProvider()
        assert isinstance(provider, AbstractEmbeddingProvider)

    def test_openai_inherits_from_abstract(self):
        """Test OpenAI stub inherits from AbstractEmbeddingProvider."""
        provider = OpenAIEmbeddingProvider()
        assert isinstance(provider, AbstractEmbeddingProvider)

    def test_oci_is_subclass(self):
        """Test OCIEmbeddingProvider is subclass of AbstractEmbeddingProvider."""
        assert issubclass(OCIEmbeddingProvider, AbstractEmbeddingProvider)

    def test_openai_is_subclass(self):
        """Test OpenAIEmbeddingProvider is subclass of AbstractEmbeddingProvider."""
        assert issubclass(OpenAIEmbeddingProvider, AbstractEmbeddingProvider)


class TestStubsAttributes:
    """Tests for stub attributes and properties."""

    def test_oci_has_model_name_attribute(self):
        """Test OCI provider has model_name attribute."""
        provider = OCIEmbeddingProvider(model_name="test-model")
        assert hasattr(provider, "model_name")
        assert provider.model_name == "test-model"

    def test_oci_has_compartment_id_attribute(self):
        """Test OCI provider has compartment_id attribute."""
        provider = OCIEmbeddingProvider(compartment_id="ocid1.compartment.oc1..xxx")
        assert hasattr(provider, "compartment_id")

    def test_openai_has_model_name_attribute(self):
        """Test OpenAI provider has model_name attribute."""
        provider = OpenAIEmbeddingProvider(model_name="test-model")
        assert hasattr(provider, "model_name")
        assert provider.model_name == "test-model"

    def test_openai_has_api_key_attribute(self):
        """Test OpenAI provider has api_key attribute."""
        provider = OpenAIEmbeddingProvider(api_key="sk-test")
        assert hasattr(provider, "api_key")


class TestStubsAbstractMethods:
    """Tests for implemented abstract methods."""

    def test_oci_implements_get_provider_name(self):
        """Test OCI implements _get_provider_name."""
        provider = OCIEmbeddingProvider()
        assert callable(provider._get_provider_name)
        assert provider._get_provider_name() == "oci-genai"

    def test_openai_implements_get_provider_name(self):
        """Test OpenAI implements _get_provider_name."""
        provider = OpenAIEmbeddingProvider()
        assert callable(provider._get_provider_name)
        assert provider._get_provider_name() == "openai"

    def test_oci_implements_embed(self):
        """Test OCI implements embed method."""
        provider = OCIEmbeddingProvider()
        assert callable(provider.embed)
        assert hasattr(provider.embed, "__await__") or hasattr(provider, "embed")

    def test_openai_implements_embed(self):
        """Test OpenAI implements embed method."""
        provider = OpenAIEmbeddingProvider()
        assert callable(provider.embed)
        assert hasattr(provider.embed, "__await__") or hasattr(provider, "embed")

    def test_oci_implements_embed_batch(self):
        """Test OCI implements embed_batch method."""
        provider = OCIEmbeddingProvider()
        assert callable(provider.embed_batch)

    def test_openai_implements_embed_batch(self):
        """Test OpenAI implements embed_batch method."""
        provider = OpenAIEmbeddingProvider()
        assert callable(provider.embed_batch)


class TestStubsErrorMessages:
    """Tests for error messages in stubs."""

    @pytest.mark.anyio
    async def test_oci_embed_error_mentions_stub(self):
        """Test OCI embed error message mentions stub."""
        provider = OCIEmbeddingProvider()
        try:
            await provider.embed("test")
            pytest.fail("Should raise NotImplementedError")
        except NotImplementedError as e:
            assert "stub" in str(e).lower()

    @pytest.mark.anyio
    async def test_openai_embed_error_mentions_stub(self):
        """Test OpenAI embed error message mentions stub."""
        provider = OpenAIEmbeddingProvider()
        try:
            await provider.embed("test")
            pytest.fail("Should raise NotImplementedError")
        except NotImplementedError as e:
            assert "stub" in str(e).lower()

    @pytest.mark.anyio
    async def test_oci_embed_batch_error_mentions_stub(self):
        """Test OCI embed_batch error message mentions stub."""
        provider = OCIEmbeddingProvider()
        try:
            await provider.embed_batch(["test"])
            pytest.fail("Should raise NotImplementedError")
        except NotImplementedError as e:
            assert "stub" in str(e).lower()

    @pytest.mark.anyio
    async def test_openai_embed_batch_error_mentions_stub(self):
        """Test OpenAI embed_batch error message mentions stub."""
        provider = OpenAIEmbeddingProvider()
        try:
            await provider.embed_batch(["test"])
            pytest.fail("Should raise NotImplementedError")
        except NotImplementedError as e:
            assert "stub" in str(e).lower()
