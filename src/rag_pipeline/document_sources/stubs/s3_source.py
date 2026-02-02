"""S3/OCI Object Storage document source stub.

This module provides a stub implementation for loading documents
from S3-compatible object storage. Implementation is planned for future phases.

OCI Context:
    OCI Object Storage is S3-compatible and can be accessed via:
    - OCI SDK (recommended for OCI)
    - S3-compatible API with access key

Future Implementation Notes:
    - Support both OCI SDK and S3 API
    - Bucket and prefix filtering
    - Object versioning support
    - Pre-authenticated request URLs
    - Streaming for large objects
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from rag_pipeline.document_sources.base import AbstractDocumentSource, DocumentSourceConfig
from rag_pipeline.core.types import Document, SourceType
from rag_pipeline.core.exceptions import NotImplementedSourceError


@dataclass
class OCIObjectStorageConfig:
    """Configuration for OCI Object Storage specific settings.

    Attributes:
        namespace: OCI Object Storage namespace.
        config_profile: OCI config profile name.
        use_instance_principal: Use Instance Principal auth instead of API keys.
    """

    namespace: str = ""
    config_profile: str = "DEFAULT"
    use_instance_principal: bool = False


@dataclass
class S3SourceConfig:
    """Configuration for S3/OCI Object Storage source.

    NOTE: Uses COMPOSITION, not inheritance from DocumentSourceConfig.
    This avoids dataclass/non-dataclass inheritance issues.

    Attributes:
        base_config: Common document source configuration.
        bucket_name: Name of the bucket.
        prefix: Object key prefix to filter.
        endpoint_url: Custom endpoint (for OCI, MinIO, etc.).
        region: AWS region or OCI region.
        access_key_id: Access key (for S3 API).
        secret_access_key: Secret key (for S3 API).
        provider: Provider type ("aws", "oci", "minio").
        use_oci_sdk: Use OCI SDK instead of S3 API.
        oci_config: OCI-specific configuration.
    """

    bucket_name: str = ""
    prefix: str = ""
    endpoint_url: str | None = None
    region: str = "us-east-1"
    access_key_id: str = ""
    secret_access_key: str = ""
    provider: Literal["aws", "oci", "minio"] = "aws"
    use_oci_sdk: bool = False
    oci_config: OCIObjectStorageConfig = field(default_factory=OCIObjectStorageConfig)
    # COMPOSITION: Include base config as a field
    base_config: DocumentSourceConfig = field(default_factory=DocumentSourceConfig)


class S3Source(AbstractDocumentSource):
    """S3/OCI Object Storage document source (stub).

    Provides an interface for loading documents from S3-compatible storage.
    Not yet implemented - raises NotImplementedSourceError.

    Supports:
        - AWS S3
        - OCI Object Storage (via S3 compatibility or OCI SDK)
        - MinIO
        - Other S3-compatible storage

    Future Features:
        - Object listing with prefix filtering
        - Streaming download for large objects
        - Version support
        - ETag-based change detection
        - Multipart download for very large files

    Example (future usage):
        >>> source = S3Source(
        ...     config=S3SourceConfig(
        ...         bucket_name="team-runbooks",
        ...         prefix="operations/",
        ...         provider="oci",
        ...         use_oci_sdk=True,
        ...         oci_config=OCIObjectStorageConfig(namespace="mytenancy"),
        ...     )
        ... )
        >>> async for doc in source.load():
        ...     process(doc)
    """

    def __init__(
        self,
        config: S3SourceConfig | None = None,
    ) -> None:
        """Initialize S3 source.

        Args:
            config: S3 source configuration.
        """
        self.s3_config = config or S3SourceConfig()
        src_type = SourceType.OCI_OBJECT_STORAGE if self.s3_config.use_oci_sdk else SourceType.S3
        # COMPOSITION: Pass base_config to parent
        super().__init__(self.s3_config.base_config, src_type)

    async def load(self) -> AsyncIterator[Document]:
        """Load all documents from the bucket.

        Raises:
            NotImplementedSourceError: This source is not yet implemented.
        """
        raise NotImplementedSourceError(
            "S3Source is not yet implemented. "
            "Download objects locally and use FileSystemSource, "
            "or implement S3Source for your use case. "
            "Recommended libraries: aioboto3 (AWS), oci (OCI SDK)"
        )
        yield  # type: ignore

    async def load_changed(self, since: datetime) -> AsyncIterator[Document]:
        """Load objects modified since a given time.

        Raises:
            NotImplementedSourceError: This source is not yet implemented.
        """
        raise NotImplementedSourceError("S3Source.load_changed not yet implemented")
        yield  # type: ignore

    async def load_by_path(self, path: str) -> Document | None:
        """Load a single object by key.

        Args:
            path: The object key to load.

        Raises:
            NotImplementedSourceError: This source is not yet implemented.
        """
        raise NotImplementedSourceError("S3Source.load_by_path not yet implemented")

    async def list_paths(self) -> list[str]:
        """List all object keys in the bucket.

        Raises:
            NotImplementedSourceError: This source is not yet implemented.
        """
        raise NotImplementedSourceError("S3Source.list_paths not yet implemented")

    async def get_changed_paths(self, since: datetime) -> list[str]:
        """Get keys of objects modified since a given time.

        Args:
            since: Timestamp to filter objects from.

        Raises:
            NotImplementedSourceError: This source is not yet implemented.
        """
        raise NotImplementedSourceError("S3Source.get_changed_paths not yet implemented")

    async def get_deleted_paths(self, known_paths: list[str]) -> list[str]:
        """Find objects that have been deleted.

        Args:
            known_paths: Previously known object keys.

        Raises:
            NotImplementedSourceError: This source is not yet implemented.
        """
        raise NotImplementedSourceError("S3Source.get_deleted_paths not yet implemented")

    def get_source_info(self) -> dict[str, Any]:
        """Get source information.

        Returns:
            Dictionary with source type, status, and configuration info.
        """
        return {
            "type": "oci_object_storage" if self.s3_config.use_oci_sdk else self.s3_config.provider,
            "status": "stub",
            "bucket": self.s3_config.bucket_name,
            "prefix": self.s3_config.prefix,
            "endpoint": self.s3_config.endpoint_url,
            "region": self.s3_config.region,
            "message": "S3Source is a stub. Implementation planned for future phase.",
        }

    async def health_check(self) -> bool:
        """Check bucket accessibility.

        Returns:
            False (not yet implemented).
        """
        return False


# Implementation notes for future development:
#
# 1. For AWS S3, use aioboto3:
#    async with aioboto3.Session().client("s3") as s3:
#        paginator = s3.get_paginator("list_objects_v2")
#        async for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
#            for obj in page.get("Contents", []):
#                yield obj["Key"]
#
# 2. For OCI Object Storage, use oci SDK:
#    from oci.object_storage import ObjectStorageClient
#    client = ObjectStorageClient(config)
#    objects = client.list_objects(namespace, bucket, prefix=prefix)
#
# 3. OCI S3 Compatibility:
#    endpoint = f"https://{namespace}.compat.objectstorage.{region}.oraclecloud.com"
#    Use Customer Secret Keys from OCI IAM
#
# 4. For large files:
#    - Stream content instead of loading into memory
#    - Use multipart download for files > 100MB
#
# 5. Change detection:
#    - Use LastModified timestamp
#    - Or use ETag for content-based change detection
