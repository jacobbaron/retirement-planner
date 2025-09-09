"""
Base storage service interface and exceptions.

This module defines the abstract interface that all storage implementations
must follow, along with common exceptions.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional


class StorageError(Exception):
    """Base exception for storage-related errors."""


class StorageNotFoundError(StorageError):
    """Raised when a requested file is not found in storage."""


class StoragePermissionError(StorageError):
    """Raised when there are permission issues with storage operations."""


class StorageService(ABC):
    """
    Abstract base class for storage services.

    This interface defines the contract that all storage implementations
    must follow for storing and retrieving files.
    """

    @abstractmethod
    def store_file(
        self, file_path: str, content: bytes, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store a file in the storage backend.

        Args:
            file_path: The path where the file should be stored
            content: The file content as bytes
            metadata: Optional metadata to store with the file

        Returns:
            str: The storage key/path where the file was stored

        Raises:
            StorageError: If the file cannot be stored
        """

    @abstractmethod
    def retrieve_file(self, file_path: str) -> bytes:
        """
        Retrieve a file from the storage backend.

        Args:
            file_path: The path/key of the file to retrieve

        Returns:
            bytes: The file content

        Raises:
            StorageNotFoundError: If the file is not found
            StorageError: If the file cannot be retrieved
        """

    @abstractmethod
    def delete_file(self, file_path: str) -> bool:
        """
        Delete a file from the storage backend.

        Args:
            file_path: The path/key of the file to delete

        Returns:
            bool: True if the file was deleted, False if it didn't exist

        Raises:
            StorageError: If the file cannot be deleted
        """

    @abstractmethod
    def file_exists(self, file_path: str) -> bool:
        """
        Check if a file exists in the storage backend.

        Args:
            file_path: The path/key of the file to check

        Returns:
            bool: True if the file exists, False otherwise
        """

    @abstractmethod
    def get_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Get metadata for a file in the storage backend.

        Args:
            file_path: The path/key of the file

        Returns:
            Dict containing file metadata (size, created_at, etc.)

        Raises:
            StorageNotFoundError: If the file is not found
        """

    @abstractmethod
    def list_files(self, prefix: str = "") -> list[str]:
        """
        List files in the storage backend with optional prefix filter.

        Args:
            prefix: Optional prefix to filter files

        Returns:
            List of file paths/keys
        """

    def store_file_from_path(
        self,
        local_path: Path,
        storage_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store a local file to the storage backend.

        Args:
            local_path: Path to the local file
            storage_path: The path where the file should be stored
            metadata: Optional metadata to store with the file

        Returns:
            str: The storage key/path where the file was stored

        Raises:
            StorageError: If the file cannot be stored
        """
        if not local_path.exists():
            raise StorageError(f"Local file does not exist: {local_path}")

        with open(local_path, "rb") as f:
            content = f.read()

        return self.store_file(storage_path, content, metadata)

    def retrieve_file_to_path(self, storage_path: str, local_path: Path) -> None:
        """
        Retrieve a file from storage and save it to a local path.

        Args:
            storage_path: The path/key of the file to retrieve
            local_path: Local path where the file should be saved

        Raises:
            StorageNotFoundError: If the file is not found
            StorageError: If the file cannot be retrieved or saved
        """
        content = self.retrieve_file(storage_path)

        # Ensure parent directory exists
        local_path.parent.mkdir(parents=True, exist_ok=True)

        with open(local_path, "wb") as f:
            f.write(content)
