# OCI Context and Integration Guide

## Overview

This document provides context about OCI (Oracle Cloud Infrastructure) internal tooling and future integration points for the RAG pipeline.

## Current Understanding (Pre-Onboarding)

> **Note**: The information below is based on public documentation and may need updates after onboarding to confirm internal tooling.

### Code Repositories

| Tool | Likelihood | Notes |
|------|------------|-------|
| Visual Builder Studio (VBS) | High | Oracle's cloud-based Git service |
| Internal GitLab | Medium | Some teams use self-hosted GitLab |
| GitHub Enterprise | Low | Less common within Oracle |

**VBS Integration Considerations:**
- REST API available for repository access
- OAuth-based authentication
- Clone via HTTPS with personal access tokens

### Runbook Storage

| Tool | Likelihood | Notes |
|------|------------|-------|
| Fleet Application Management | High | Part of OCI Operations Insights |
| Confluence | High | Widely used at Oracle |
| OCI Object Storage | Medium | For version-controlled runbooks |
| Internal Wiki | Medium | Various internal tools |

**Runbook Format Expectations:**
- Markdown files (`.md`)
- Confluence pages (HTML export or API)
- Structured JSON/YAML runbook definitions

### Documentation

| Tool | Likelihood | Notes |
|------|------------|-------|
| Confluence | Very High | Standard Oracle documentation |
| README files in repos | High | In-repo documentation |
| Oracle Docs | Medium | Public-facing docs |

## OCI Services for Future Integration

### OCI GenAI Service

**Purpose**: LLM inference for text generation

**Configuration:**
```python
# Future: src/rag_pipeline/llm_providers/oci_genai.py
class OCIGenAIProvider:
    def __init__(
        self,
        compartment_id: str,
        model_id: str = "cohere.command",
        endpoint: str | None = None,  # Auto-detected by region
    ):
        ...
```

**Models Available:**
- Cohere Command (command, command-light)
- Cohere Command R/R+
- Meta Llama 2 (13B, 70B)
- Custom fine-tuned models

**Authentication:**
- OCI SDK config file (`~/.oci/config`)
- Instance principal (within OCI compute)
- Resource principal (OCI Functions)

**Endpoint Pattern:**
```
https://inference.generativeai.{region}.oci.oraclecloud.com
```

### Oracle Database 23ai Vector Search

**Purpose**: Production-grade vector storage with SQL access

**Configuration:**
```python
# Future: src/rag_pipeline/vector_stores/oci_vector.py
class OCIVectorStore:
    def __init__(
        self,
        connection_string: str,  # Oracle connection string
        table_name: str = "rag_embeddings",
        index_type: Literal["hnsw", "ivf"] = "hnsw",
    ):
        ...
```

**Features:**
- `VECTOR` data type for embeddings
- HNSW and IVF index types
- SQL-based querying with `VECTOR_DISTANCE()`
- Hybrid search (vector + metadata filtering)

**Schema Example:**
```sql
CREATE TABLE rag_embeddings (
    id VARCHAR2(64) PRIMARY KEY,
    document_id VARCHAR2(64) NOT NULL,
    content CLOB NOT NULL,
    embedding VECTOR(384) NOT NULL,  -- dimension matches model
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE VECTOR INDEX rag_embedding_idx 
ON rag_embeddings(embedding)
ORGANIZATION INMEMORY NEIGHBOR GRAPH;
```

### OCI Object Storage

**Purpose**: Scalable storage for documents and runbooks

**Configuration:**
```python
# Future: src/rag_pipeline/document_sources/stubs/oci_storage.py
class OCIStorageSource:
    def __init__(
        self,
        namespace: str,
        bucket_name: str,
        prefix: str = "",
    ):
        ...
```

**Usage Pattern:**
```python
# List objects
objects = await source.scan()

# Download and process
for obj in objects:
    content = await source.load(obj.name)
    chunks = chunker.chunk(content)
```

## Authentication Patterns

### Local Development

Use OCI SDK config file:

```ini
# ~/.oci/config
[DEFAULT]
user=ocid1.user.oc1..example
fingerprint=aa:bb:cc:dd:ee:ff:00:11:22:33:44:55:66:77:88:99
tenancy=ocid1.tenancy.oc1..example
region=us-ashburn-1
key_file=~/.oci/oci_api_key.pem
```

### Production (OCI Compute)

Use instance principal:

```python
from oci.auth import signers

signer = signers.get_instance_principals_signer()
client = GenerativeAiInferenceClient(config={}, signer=signer)
```

### OCI Functions

Use resource principal:

```python
from oci.auth import signers

signer = signers.get_resource_principals_signer()
client = GenerativeAiInferenceClient(config={}, signer=signer)
```

## Integration Roadmap

### Phase 1: Local Development (Current)
- ChromaDB for vector storage
- Ollama for LLM inference
- sentence-transformers for embeddings
- Local filesystem for documents

### Phase 2: OCI GenAI Integration
- Replace Ollama with OCI GenAI
- Use OCI GenAI embeddings
- Keep local vector store

### Phase 3: Full OCI Integration
- Oracle Database 23ai for vectors
- OCI Object Storage for documents
- VBS/Confluence integration
- OCI monitoring and logging

### Phase 4: Production Hardening
- OCI Container Instances deployment
- OCI API Gateway for external access
- OCI Streaming for async processing
- OCI Notifications for alerts

## Environment Configuration

### Required Environment Variables (Future)

```bash
# OCI Authentication
OCI_CONFIG_FILE=/path/to/.oci/config
OCI_CONFIG_PROFILE=DEFAULT

# GenAI Service
RAG_OCI_COMPARTMENT_ID=ocid1.compartment.oc1..example
RAG_OCI_GENAI_ENDPOINT=https://inference.generativeai.us-ashburn-1.oci.oraclecloud.com

# Vector Store (Oracle 23ai)
RAG_ORACLE_CONNECTION_STRING=user/pass@host:1521/service

# Object Storage
RAG_OCI_NAMESPACE=mytenancy
RAG_OCI_BUCKET=rag-documents
```

## Security Considerations

### Secrets Management
- Use OCI Vault for API keys and credentials
- Never commit secrets to repository
- Use environment variables for local development

### Network Security
- OCI services accessed via service gateway (no public internet)
- VCN security lists for database access
- Private endpoints where available

### Data Classification
- Be aware of data sensitivity in indexed documents
- PII detection before ingestion (future)
- Audit logging for all queries

## Useful OCI CLI Commands

```bash
# List GenAI models
oci generative-ai model list --compartment-id $COMPARTMENT_ID

# Test GenAI inference
oci generative-ai inference chat \
    --compartment-id $COMPARTMENT_ID \
    --model-id cohere.command \
    --messages '[{"role":"user","content":"Hello"}]'

# List Object Storage buckets
oci os bucket list --compartment-id $COMPARTMENT_ID

# Get object
oci os object get \
    --namespace $NAMESPACE \
    --bucket-name $BUCKET \
    --name path/to/file.md \
    --file output.md
```

## Resources

### Official Documentation
- [OCI GenAI Service](https://docs.oracle.com/en-us/iaas/Content/generative-ai/home.htm)
- [Oracle Database 23ai AI Vector Search](https://docs.oracle.com/en/database/oracle/oracle-database/23/vecse/)
- [OCI Object Storage](https://docs.oracle.com/en-us/iaas/Content/Object/home.htm)
- [Visual Builder Studio](https://docs.oracle.com/en/cloud/paas/visual-builder/)

### Internal Resources (Confirm After Onboarding)
- Object Storage team wiki
- Team runbook repository
- Architecture documentation
