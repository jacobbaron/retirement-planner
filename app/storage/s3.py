"""
Amazon S3 storage service implementation.

This module provides a storage service that uses Amazon S3 for storing
and retrieving files. It's suitable for production deployments and
multi-instance applications.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError

    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False

from .base import (
    StorageError,
    StorageNotFoundError,
    StoragePermissionError,
    StorageService,
)


class S3StorageService(StorageService):
    """
    Amazon S3 storage service.

    Stores files in an S3 bucket with optional metadata.
    """

    def __init__(
        self,
        bucket_name: str,
        region_name: str = "us-east-1",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        prefix: str = "",
    ):
        """
        Initialize the S3 storage service.

        Args:
            bucket_name: Name of the S3 bucket
            region_name: AWS region name
            aws_access_key_id: AWS access key ID (optional, can use IAM roles)
            aws_secret_access_key: AWS secret access key (optional, can use IAM roles)
            prefix: Optional prefix for all stored files

        Raises:
            StorageError: If S3 is not available or credentials are invalid
        """
        if not S3_AVAILABLE:
            raise StorageError(
                "boto3 is not installed. Install with: pip install boto3"
            )

        self.bucket_name = bucket_name
        self.region_name = region_name
        self.prefix = prefix.strip("/")

        # Initialize S3 client
        try:
            session_kwargs = {"region_name": region_name}
            if aws_access_key_id and aws_secret_access_key:
                session_kwargs.update(
                    {
                        "aws_access_key_id": aws_access_key_id,
                        "aws_secret_access_key": aws_secret_access_key,
                    }
                )

            self.s3_client = boto3.client("s3", **session_kwargs)

            # Test connection by checking if bucket exists
            self.s3_client.head_bucket(Bucket=bucket_name)

        except NoCredentialsError:
            raise StorageError("AWS credentials not found")
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "404":
                raise StorageError(f"S3 bucket '{bucket_name}' not found")
            elif error_code == "403":
                raise StoragePermissionError(
                    f"Access denied to S3 bucket '{bucket_name}'"
                )
            else:
                raise StorageError(f"Failed to connect to S3: {e}")

    def _get_s3_key(self, file_path: str) -> str:
        """
        Get the S3 key for a storage path.

        Args:
            file_path: The storage path/key

        Returns:
            str: S3 key
        """
        # Remove leading slash and add prefix if specified
        clean_path = file_path.lstrip("/")
        if self.prefix:
            return f"{self.prefix}/{clean_path}"
        return clean_path

    def _get_metadata_key(self, file_path: str) -> str:
        """
        Get the metadata S3 key for a storage path.

        Args:
            file_path: The storage path/key

        Returns:
            str: S3 key for metadata
        """
        return f"{self._get_s3_key(file_path)}.meta"

    def store_file(
        self, file_path: str, content: bytes, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store a file in S3.

        Args:
            file_path: The path where the file should be stored
            content: The file content as bytes
            metadata: Optional metadata to store with the file

        Returns:
            str: The storage key/path where the file was stored

        Raises:
            StorageError: If the file cannot be stored
        """
        try:
            s3_key = self._get_s3_key(file_path)

            # Upload the file
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=content,
                ContentType=(
                    metadata.get("content_type", "application/octet-stream")
                    if metadata
                    else "application/octet-stream"
                ),
            )

            # Store metadata if provided
            if metadata is not None:
                metadata_key = self._get_metadata_key(file_path)
                metadata_data = {
                    "created_at": datetime.utcnow().isoformat(),
                    "size": len(content),
                    "content_type": metadata.get(
                        "content_type", "application/octet-stream"
                    ),
                    **metadata,
                }

                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=metadata_key,
                    Body=json.dumps(metadata_data, indent=2).encode("utf-8"),
                    ContentType="application/json",
                )

            return file_path

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "403":
                raise StoragePermissionError(
                    f"Permission denied storing file {file_path}: {e}"
                )
            else:
                raise StorageError(f"Failed to store file {file_path}: {e}")

    def retrieve_file(self, file_path: str) -> bytes:
        """
        Retrieve a file from S3.

        Args:
            file_path: The path/key of the file to retrieve

        Returns:
            bytes: The file content

        Raises:
            StorageNotFoundError: If the file is not found
            StorageError: If the file cannot be retrieved
        """
        try:
            s3_key = self._get_s3_key(file_path)

            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)

            return response["Body"].read()

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "NoSuchKey":
                raise StorageNotFoundError(f"File not found: {file_path}")
            elif error_code == "403":
                raise StoragePermissionError(
                    f"Permission denied retrieving file {file_path}: {e}"
                )
            else:
                raise StorageError(f"Failed to retrieve file {file_path}: {e}")

    def delete_file(self, file_path: str) -> bool:
        """
        Delete a file from S3.

        Args:
            file_path: The path/key of the file to delete

        Returns:
            bool: True if the file was deleted, False if it didn't exist

        Raises:
            StorageError: If the file cannot be deleted
        """
        try:
            s3_key = self._get_s3_key(file_path)
            metadata_key = self._get_metadata_key(file_path)

            deleted = False

            # Delete the main file
            try:
                self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
                deleted = True
            except ClientError as e:
                if e.response["Error"]["Code"] != "NoSuchKey":
                    raise

            # Delete metadata file if it exists
            try:
                self.s3_client.delete_object(Bucket=self.bucket_name, Key=metadata_key)
            except ClientError as e:
                if e.response["Error"]["Code"] != "NoSuchKey":
                    raise

            return deleted

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "403":
                raise StoragePermissionError(
                    f"Permission denied deleting file {file_path}: {e}"
                )
            else:
                raise StorageError(f"Failed to delete file {file_path}: {e}")

    def file_exists(self, file_path: str) -> bool:
        """
        Check if a file exists in S3.

        Args:
            file_path: The path/key of the file to check

        Returns:
            bool: True if the file exists, False otherwise
        """
        try:
            s3_key = self._get_s3_key(file_path)
            self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            # Re-raise other errors
            raise StorageError(f"Failed to check if file exists {file_path}: {e}")

    def get_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Get metadata for a file in S3.

        Args:
            file_path: The path/key of the file

        Returns:
            Dict containing file metadata (size, created_at, etc.)

        Raises:
            StorageNotFoundError: If the file is not found
        """
        try:
            s3_key = self._get_s3_key(file_path)

            # Get object metadata from S3
            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)

            metadata = {
                "size": response["ContentLength"],
                "created_at": response["LastModified"].isoformat(),
                "modified_at": response["LastModified"].isoformat(),
                "content_type": response.get("ContentType", "application/octet-stream"),
                "etag": response.get("ETag", "").strip('"'),
            }

            # Try to load additional metadata if available
            try:
                metadata_key = self._get_metadata_key(file_path)
                metadata_response = self.s3_client.get_object(
                    Bucket=self.bucket_name, Key=metadata_key
                )
                stored_metadata = json.loads(
                    metadata_response["Body"].read().decode("utf-8")
                )
                metadata.update(stored_metadata)
            except ClientError:
                # Metadata file doesn't exist, use basic S3 metadata
                pass

            return metadata

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "404":
                raise StorageNotFoundError(f"File not found: {file_path}")
            elif error_code == "403":
                raise StoragePermissionError(
                    f"Permission denied getting metadata for file {file_path}: {e}"
                )
            else:
                raise StorageError(f"Failed to get metadata for file {file_path}: {e}")

    def list_files(self, prefix: str = "") -> List[str]:
        """
        List files in S3 with optional prefix filter.

        Args:
            prefix: Optional prefix to filter files

        Returns:
            List of file paths/keys
        """
        try:
            files = []
            s3_prefix = self._get_s3_key(prefix) if prefix else self.prefix

            paginator = self.s3_client.get_paginator("list_objects_v2")
            page_iterator = paginator.paginate(
                Bucket=self.bucket_name, Prefix=s3_prefix
            )

            for page in page_iterator:
                if "Contents" in page:
                    for obj in page["Contents"]:
                        key = obj["Key"]
                        # Skip metadata files
                        if key.endswith(".meta"):
                            continue

                        # Remove prefix to get relative path
                        if self.prefix and key.startswith(f"{self.prefix}/"):
                            relative_key = key[len(self.prefix) + 1 :]
                        else:
                            relative_key = key

                        files.append(relative_key)

            return sorted(files)

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "403":
                raise StoragePermissionError(
                    f"Permission denied listing files with prefix {prefix}: {e}"
                )
            else:
                raise StorageError(f"Failed to list files with prefix {prefix}: {e}")

    def get_storage_info(self) -> Dict[str, Any]:
        """
        Get information about the S3 storage backend.

        Returns:
            Dict containing storage information
        """
        try:
            # Get bucket info (call ensures access; response content not used)
            self.s3_client.head_bucket(Bucket=self.bucket_name)

            # Count files and calculate total size
            total_size = 0
            file_count = 0

            paginator = self.s3_client.get_paginator("list_objects_v2")
            page_iterator = paginator.paginate(
                Bucket=self.bucket_name, Prefix=self.prefix
            )

            for page in page_iterator:
                if "Contents" in page:
                    for obj in page["Contents"]:
                        if not obj["Key"].endswith(".meta"):
                            total_size += obj["Size"]
                            file_count += 1

            return {
                "type": "s3",
                "bucket_name": self.bucket_name,
                "region": self.region_name,
                "prefix": self.prefix,
                "total_files": file_count,
                "total_size_bytes": total_size,
                "accessible": True,
            }

        except ClientError as e:
            return {
                "type": "s3",
                "bucket_name": self.bucket_name,
                "region": self.region_name,
                "prefix": self.prefix,
                "error": str(e),
                "accessible": False,
            }
