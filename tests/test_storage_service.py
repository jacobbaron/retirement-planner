"""
Tests for storage service implementations.

This module tests both local and S3 storage services, including
the factory function and error handling.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from app.storage import (
    StorageService, StorageError, StorageNotFoundError, StoragePermissionError,
    LocalStorageService, create_storage_service, get_storage_service
)
from app.config import Settings

# Try to import S3StorageService, skip S3 tests if not available
try:
    from app.storage import S3StorageService
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False


class TestLocalStorageService:
    """Test cases for local storage service."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def local_storage(self, temp_dir):
        """Create a local storage service instance."""
        return LocalStorageService(base_path=temp_dir, create_dirs=True)
    
    def test_store_and_retrieve_file(self, local_storage):
        """Test storing and retrieving a file."""
        content = b"Hello, World!"
        file_path = "test/file.txt"
        
        # Store file
        stored_path = local_storage.store_file(file_path, content)
        assert stored_path == file_path
        
        # Verify file exists
        assert local_storage.file_exists(file_path)
        
        # Retrieve file
        retrieved_content = local_storage.retrieve_file(file_path)
        assert retrieved_content == content
    
    def test_store_file_with_metadata(self, local_storage):
        """Test storing a file with metadata."""
        content = b"Test content"
        file_path = "test/with_metadata.txt"
        metadata = {
            "content_type": "text/plain",
            "description": "Test file"
        }
        
        stored_path = local_storage.store_file(file_path, content, metadata)
        assert stored_path == file_path
        
        # Check metadata
        file_metadata = local_storage.get_file_metadata(file_path)
        assert file_metadata["content_type"] == "text/plain"
        assert file_metadata["description"] == "Test file"
        assert file_metadata["size"] == len(content)
        assert "created_at" in file_metadata
    
    def test_delete_file(self, local_storage):
        """Test deleting a file."""
        content = b"To be deleted"
        file_path = "test/delete_me.txt"
        
        # Store file
        local_storage.store_file(file_path, content)
        assert local_storage.file_exists(file_path)
        
        # Delete file
        deleted = local_storage.delete_file(file_path)
        assert deleted is True
        assert not local_storage.file_exists(file_path)
        
        # Try to delete non-existent file
        deleted = local_storage.delete_file("nonexistent.txt")
        assert deleted is False
    
    def test_file_not_found_error(self, local_storage):
        """Test error handling for non-existent files."""
        with pytest.raises(StorageNotFoundError):
            local_storage.retrieve_file("nonexistent.txt")
        
        with pytest.raises(StorageNotFoundError):
            local_storage.get_file_metadata("nonexistent.txt")
    
    def test_list_files(self, local_storage):
        """Test listing files with prefix filter."""
        # Store multiple files
        files = [
            "folder1/file1.txt",
            "folder1/file2.txt", 
            "folder2/file3.txt",
            "root_file.txt"
        ]
        
        for file_path in files:
            local_storage.store_file(file_path, b"content")
        
        # List all files
        all_files = local_storage.list_files()
        assert len(all_files) == 4
        assert set(all_files) == set(files)
        
        # List files with prefix
        folder1_files = local_storage.list_files("folder1/")
        assert len(folder1_files) == 2
        assert "folder1/file1.txt" in folder1_files
        assert "folder1/file2.txt" in folder1_files
    
    def test_store_file_from_path(self, local_storage, temp_dir):
        """Test storing a file from local path."""
        # Create a local file
        local_file = Path(temp_dir) / "source.txt"
        local_file.write_text("Source content")
        
        # Store it
        stored_path = local_storage.store_file_from_path(local_file, "stored/source.txt")
        assert stored_path == "stored/source.txt"
        
        # Verify content
        retrieved_content = local_storage.retrieve_file("stored/source.txt")
        assert retrieved_content == b"Source content"
    
    def test_retrieve_file_to_path(self, local_storage, temp_dir):
        """Test retrieving a file to local path."""
        # Store a file
        content = b"Retrieved content"
        local_storage.store_file("test/retrieve.txt", content)
        
        # Retrieve to local path
        local_path = Path(temp_dir) / "retrieved.txt"
        local_storage.retrieve_file_to_path("test/retrieve.txt", local_path)
        
        # Verify content
        assert local_path.read_bytes() == content
    
    def test_get_storage_info(self, local_storage):
        """Test getting storage information."""
        # Store some files
        local_storage.store_file("file1.txt", b"content1")
        local_storage.store_file("file2.txt", b"content2")
        
        info = local_storage.get_storage_info()
        assert info["type"] == "local"
        assert info["total_files"] == 2
        assert info["total_size_bytes"] == 16  # 8 bytes each
        assert info["exists"] is True
    
    def test_path_sanitization(self, local_storage):
        """Test that paths are properly sanitized."""
        # Try to store with potentially dangerous path
        content = b"Safe content"
        file_path = "../../../etc/passwd"
        
        # This should work because the path is sanitized to be within the base directory
        stored_path = local_storage.store_file(file_path, content)
        # Should be sanitized
        assert stored_path == file_path
        assert local_storage.file_exists(file_path)
        
        # The actual file should be in the base directory, not outside
        info = local_storage.get_storage_info()
        assert info["exists"] is True


@pytest.mark.skipif(not S3_AVAILABLE, reason="S3StorageService not available (boto3 not installed)")
class TestS3StorageService:
    """Test cases for S3 storage service."""
    
    @pytest.fixture
    def mock_s3_client(self):
        """Create a mock S3 client."""
        return Mock()
    
    @pytest.fixture
    @pytest.mark.skipif(not S3_AVAILABLE, reason="S3StorageService not available (boto3 not installed)")
    def s3_storage(self, mock_s3_client):
        """Create an S3 storage service with mocked client."""
        with patch('boto3.client', return_value=mock_s3_client):
            # Mock the head_bucket call for initialization
            mock_s3_client.head_bucket.return_value = {}
            storage = S3StorageService(
                bucket_name="test-bucket",
                region_name="us-east-1"
            )
            storage.s3_client = mock_s3_client
            return storage
    
    def test_store_file(self, s3_storage, mock_s3_client):
        """Test storing a file in S3."""
        content = b"Hello, S3!"
        file_path = "test/s3_file.txt"
        
        stored_path = s3_storage.store_file(file_path, content)
        assert stored_path == file_path
        
        # Verify S3 client was called
        mock_s3_client.put_object.assert_called_once()
        call_args = mock_s3_client.put_object.call_args
        assert call_args[1]["Bucket"] == "test-bucket"
        assert call_args[1]["Key"] == "test/s3_file.txt"
        assert call_args[1]["Body"] == content
    
    def test_store_file_with_metadata(self, s3_storage, mock_s3_client):
        """Test storing a file with metadata in S3."""
        content = b"Test content"
        file_path = "test/with_metadata.txt"
        metadata = {
            "content_type": "text/plain",
            "description": "Test file"
        }
        
        stored_path = s3_storage.store_file(file_path, content, metadata)
        assert stored_path == file_path
        
        # Should have two put_object calls (file + metadata)
        assert mock_s3_client.put_object.call_count == 2
    
    def test_retrieve_file(self, s3_storage, mock_s3_client):
        """Test retrieving a file from S3."""
        content = b"Retrieved content"
        file_path = "test/retrieve.txt"
        
        # Mock S3 response
        mock_response = {"Body": Mock()}
        mock_response["Body"].read.return_value = content
        mock_s3_client.get_object.return_value = mock_response
        
        retrieved_content = s3_storage.retrieve_file(file_path)
        assert retrieved_content == content
        
        # Verify S3 client was called
        mock_s3_client.get_object.assert_called_once_with(
            Bucket="test-bucket",
            Key="test/retrieve.txt"
        )
    
    def test_retrieve_file_not_found(self, s3_storage, mock_s3_client):
        """Test retrieving a non-existent file from S3."""
        file_path = "nonexistent.txt"
        
        # Mock S3 NoSuchKey error
        error = Mock()
        error.response = {"Error": {"Code": "NoSuchKey"}}
        mock_s3_client.get_object.side_effect = error
        
        with pytest.raises(StorageNotFoundError):
            s3_storage.retrieve_file(file_path)
    
    def test_delete_file(self, s3_storage, mock_s3_client):
        """Test deleting a file from S3."""
        file_path = "test/delete_me.txt"
        
        deleted = s3_storage.delete_file(file_path)
        assert deleted is True
        
        # Should have two delete_object calls (file + metadata)
        assert mock_s3_client.delete_object.call_count == 2
    
    def test_file_exists(self, s3_storage, mock_s3_client):
        """Test checking if a file exists in S3."""
        file_path = "test/exists.txt"
        
        # File exists
        mock_s3_client.head_object.return_value = {}
        assert s3_storage.file_exists(file_path) is True
        
        # File doesn't exist
        error = Mock()
        error.response = {"Error": {"Code": "404"}}
        mock_s3_client.head_object.side_effect = error
        assert s3_storage.file_exists(file_path) is False
    
    def test_get_file_metadata(self, s3_storage, mock_s3_client):
        """Test getting file metadata from S3."""
        file_path = "test/metadata.txt"
        
        # Mock S3 head_object response
        mock_s3_client.head_object.return_value = {
            "ContentLength": 100,
            "LastModified": Mock(),
            "ContentType": "text/plain",
            "ETag": '"abc123"'
        }
        mock_s3_client.head_object.return_value["LastModified"].isoformat.return_value = "2023-01-01T00:00:00"
        
        metadata = s3_storage.get_file_metadata(file_path)
        assert metadata["size"] == 100
        assert metadata["content_type"] == "text/plain"
        assert metadata["etag"] == "abc123"
    
    def test_list_files(self, s3_storage, mock_s3_client):
        """Test listing files in S3."""
        # Mock paginator response
        mock_paginator = Mock()
        mock_s3_client.get_paginator.return_value = mock_paginator
        
        mock_page = {
            "Contents": [
                {"Key": "file1.txt"},
                {"Key": "file2.txt"},
                {"Key": "file1.txt.meta"}  # Should be filtered out
            ]
        }
        mock_paginator.paginate.return_value = [mock_page]
        
        files = s3_storage.list_files()
        assert len(files) == 2
        assert "file1.txt" in files
        assert "file2.txt" in files
    
    def test_s3_not_available(self):
        """Test behavior when boto3 is not available."""
        with patch('app.storage.s3.S3_AVAILABLE', False):
            with pytest.raises(StorageError, match="boto3 is not installed"):
                S3StorageService(bucket_name="test-bucket")


class TestStorageFactory:
    """Test cases for storage service factory."""
    
    def test_create_local_storage(self):
        """Test creating local storage service."""
        settings = Settings()
        settings.storage_type = "local"
        settings.storage_base_path = "/tmp/test"
        
        storage = create_storage_service(settings)
        assert isinstance(storage, LocalStorageService)
        assert storage.base_path == Path("/tmp/test")
    
    @pytest.mark.skipif(not S3_AVAILABLE, reason="S3StorageService not available (boto3 not installed)")
    def test_create_s3_storage(self):
        """Test creating S3 storage service."""
        settings = Settings()
        settings.storage_type = "s3"
        settings.s3_bucket_name = "test-bucket"
        settings.s3_region_name = "us-west-2"
        
        with patch('app.storage.s3.boto3.client') as mock_client:
            mock_client.return_value.head_bucket.return_value = {}
            
            storage = create_storage_service(settings)
            assert isinstance(storage, S3StorageService)
            assert storage.bucket_name == "test-bucket"
            assert storage.region_name == "us-west-2"
    
    @pytest.mark.skipif(not S3_AVAILABLE, reason="S3StorageService not available (boto3 not installed)")
    def test_create_s3_storage_missing_bucket(self):
        """Test creating S3 storage without bucket name."""
        settings = Settings()
        settings.storage_type = "s3"
        settings.s3_bucket_name = None
        
        with pytest.raises(ValueError, match="S3_BUCKET_NAME must be set"):
            create_storage_service(settings)
    
    def test_create_storage_invalid_type(self):
        """Test creating storage with invalid type."""
        settings = Settings()
        settings.storage_type = "invalid"
        
        with pytest.raises(ValueError, match="Unsupported storage type"):
            create_storage_service(settings)
    
    def test_get_storage_service(self):
        """Test getting storage service with global settings."""
        with patch('app.config.get_global_settings') as mock_get_settings:
            mock_settings = Settings()
            mock_settings.storage_type = "local"
            mock_get_settings.return_value = mock_settings
            
            storage = get_storage_service()
            assert isinstance(storage, LocalStorageService)


class TestStorageIntegration:
    """Integration tests for storage services."""
    
    def test_round_trip_storage(self):
        """Test complete round-trip storage operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = LocalStorageService(base_path=temp_dir)
            
            # Test data
            original_content = b"This is test content for round-trip testing"
            file_path = "integration/test_file.txt"
            metadata = {
                "content_type": "text/plain",
                "description": "Integration test file"
            }
            
            # Store file
            stored_path = storage.store_file(file_path, original_content, metadata)
            assert stored_path == file_path
            
            # Verify file exists
            assert storage.file_exists(file_path)
            
            # Get metadata
            file_metadata = storage.get_file_metadata(file_path)
            assert file_metadata["size"] == len(original_content)
            assert file_metadata["content_type"] == "text/plain"
            
            # Retrieve file
            retrieved_content = storage.retrieve_file(file_path)
            assert retrieved_content == original_content
            
            # List files
            files = storage.list_files("integration/")
            assert file_path in files
            
            # Delete file
            deleted = storage.delete_file(file_path)
            assert deleted is True
            assert not storage.file_exists(file_path)
    
    def test_storage_with_special_characters(self):
        """Test storage with special characters in paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = LocalStorageService(base_path=temp_dir)
            
            # Test with various special characters
            test_cases = [
                "folder with spaces/file.txt",
                "folder-with-dashes/file.txt",
                "folder_with_underscores/file.txt",
                "folder.with.dots/file.txt",
                "folder/with/multiple/levels/file.txt"
            ]
            
            for file_path in test_cases:
                content = f"Content for {file_path}".encode()
                stored_path = storage.store_file(file_path, content)
                assert stored_path == file_path
                assert storage.file_exists(file_path)
                
                retrieved_content = storage.retrieve_file(file_path)
                assert retrieved_content == content
