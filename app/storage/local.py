"""
Local filesystem storage service implementation.

This module provides a storage service that uses the local filesystem
for storing and retrieving files. It's suitable for development and
single-instance deployments.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .base import (
    StorageError,
    StorageNotFoundError,
    StoragePermissionError,
    StorageService,
)


class LocalStorageService(StorageService):
    """
    Local filesystem storage service.

    Stores files in a local directory with optional metadata files.
    """

    def __init__(self, base_path: str = "storage", create_dirs: bool = True):
        """
        Initialize the local storage service.

        Args:
            base_path: Base directory for storing files
            create_dirs: Whether to create directories if they don't exist
        """
        self.base_path = Path(base_path)
        self.create_dirs = create_dirs

        if self.create_dirs:
            self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_file_path(self, file_path: str) -> Path:
        """
        Get the full local path for a storage path.

        Args:
            file_path: The storage path/key

        Returns:
            Path: Full local filesystem path
        """
        # Sanitize the path to prevent directory traversal
        safe_path = file_path.lstrip("/")
        safe_path = safe_path.replace("\\", "/")  # Normalize separators

        # Remove any .. attempts by resolving the path
        path_parts = safe_path.split("/")
        sanitized_parts = []
        for part in path_parts:
            if part == "..":
                # Go up one level if possible, but don't go above base_path
                if sanitized_parts:
                    sanitized_parts.pop()
            elif part and part != ".":
                sanitized_parts.append(part)

        safe_path = "/".join(sanitized_parts)
        return self.base_path / safe_path

    def _get_metadata_path(self, file_path: str) -> Path:
        """
        Get the metadata file path for a storage path.

        Args:
            file_path: The storage path/key

        Returns:
            Path: Path to the metadata file
        """
        file_path_obj = self._get_file_path(file_path)
        return file_path_obj.with_suffix(file_path_obj.suffix + ".meta")

    def store_file(
        self, file_path: str, content: bytes, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store a file in the local filesystem.

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
            local_path = self._get_file_path(file_path)

            # Create parent directories if needed
            if self.create_dirs:
                local_path.parent.mkdir(parents=True, exist_ok=True)

            # Write the file content
            with open(local_path, "wb") as f:
                f.write(content)

            # Store metadata if provided
            if metadata is not None:
                metadata_path = self._get_metadata_path(file_path)
                metadata_data = {
                    "created_at": datetime.utcnow().isoformat(),
                    "size": len(content),
                    "content_type": metadata.get(
                        "content_type", "application/octet-stream"
                    ),
                    **metadata,
                }

                with open(metadata_path, "w") as f:
                    json.dump(metadata_data, f, indent=2)

            return file_path

        except PermissionError as e:
            raise StoragePermissionError(
                f"Permission denied storing file {file_path}: {e}"
            )
        except OSError as e:
            raise StorageError(f"Failed to store file {file_path}: {e}")

    def retrieve_file(self, file_path: str) -> bytes:
        """
        Retrieve a file from the local filesystem.

        Args:
            file_path: The path/key of the file to retrieve

        Returns:
            bytes: The file content

        Raises:
            StorageNotFoundError: If the file is not found
            StorageError: If the file cannot be retrieved
        """
        try:
            local_path = self._get_file_path(file_path)

            if not local_path.exists():
                raise StorageNotFoundError(f"File not found: {file_path}")

            with open(local_path, "rb") as f:
                return f.read()

        except PermissionError as e:
            raise StoragePermissionError(
                f"Permission denied retrieving file {file_path}: {e}"
            )
        except OSError as e:
            raise StorageError(f"Failed to retrieve file {file_path}: {e}")

    def delete_file(self, file_path: str) -> bool:
        """
        Delete a file from the local filesystem.

        Args:
            file_path: The path/key of the file to delete

        Returns:
            bool: True if the file was deleted, False if it didn't exist

        Raises:
            StorageError: If the file cannot be deleted
        """
        try:
            local_path = self._get_file_path(file_path)
            metadata_path = self._get_metadata_path(file_path)

            deleted = False

            # Delete the main file
            if local_path.exists():
                local_path.unlink()
                deleted = True

            # Delete metadata file if it exists
            if metadata_path.exists():
                metadata_path.unlink()

            return deleted

        except PermissionError as e:
            raise StoragePermissionError(
                f"Permission denied deleting file {file_path}: {e}"
            )
        except OSError as e:
            raise StorageError(f"Failed to delete file {file_path}: {e}")

    def file_exists(self, file_path: str) -> bool:
        """
        Check if a file exists in the local filesystem.

        Args:
            file_path: The path/key of the file to check

        Returns:
            bool: True if the file exists, False otherwise
        """
        local_path = self._get_file_path(file_path)
        return local_path.exists()

    def get_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Get metadata for a file in the local filesystem.

        Args:
            file_path: The path/key of the file

        Returns:
            Dict containing file metadata (size, created_at, etc.)

        Raises:
            StorageNotFoundError: If the file is not found
        """
        local_path = self._get_file_path(file_path)
        metadata_path = self._get_metadata_path(file_path)

        if not local_path.exists():
            raise StorageNotFoundError(f"File not found: {file_path}")

        # Get basic file stats
        stat = local_path.stat()
        metadata = {
            "size": stat.st_size,
            "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "content_type": "application/octet-stream",
        }

        # Load additional metadata if available
        if metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    stored_metadata = json.load(f)
                    metadata.update(stored_metadata)
            except (json.JSONDecodeError, OSError):
                # If metadata file is corrupted, just use basic stats
                pass

        return metadata

    def list_files(self, prefix: str = "") -> list[str]:
        """
        List files in the local filesystem with optional prefix filter.

        Args:
            prefix: Optional prefix to filter files

        Returns:
            List of file paths/keys
        """
        try:
            files = []
            prefix_path = self._get_file_path(prefix) if prefix else self.base_path

            if not prefix_path.exists():
                return files

            # Walk through the directory tree
            for root, dirs, filenames in os.walk(prefix_path):
                for filename in filenames:
                    # Skip metadata files
                    if filename.endswith(".meta"):
                        continue

                    full_path = Path(root) / filename
                    # Get relative path from base
                    relative_path = full_path.relative_to(self.base_path)
                    # Convert to forward slashes for consistency
                    storage_path = str(relative_path).replace("\\", "/")
                    files.append(storage_path)

            return sorted(files)

        except OSError as e:
            raise StorageError(f"Failed to list files with prefix {prefix}: {e}")

    def get_storage_info(self) -> Dict[str, Any]:
        """
        Get information about the storage backend.

        Returns:
            Dict containing storage information
        """
        try:
            total_size = 0
            file_count = 0

            for root, dirs, filenames in os.walk(self.base_path):
                for filename in filenames:
                    if not filename.endswith(".meta"):
                        file_path = Path(root) / filename
                        total_size += file_path.stat().st_size
                        file_count += 1

            return {
                "type": "local",
                "base_path": str(self.base_path),
                "total_files": file_count,
                "total_size_bytes": total_size,
                "exists": self.base_path.exists(),
            }

        except OSError as e:
            return {
                "type": "local",
                "base_path": str(self.base_path),
                "error": str(e),
                "exists": False,
            }
