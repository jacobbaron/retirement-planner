"""
Storage service factory for creating storage instances based on configuration.

This module provides a factory function to create the appropriate storage
service based on the application configuration.
"""

from typing import Union
from app.config import Settings
from .base import StorageService
from .local import LocalStorageService
from .s3 import S3StorageService


def create_storage_service(settings: Settings) -> StorageService:
    """
    Create a storage service instance based on configuration.
    
    Args:
        settings: Application settings containing storage configuration
        
    Returns:
        StorageService: Configured storage service instance
        
    Raises:
        ValueError: If storage configuration is invalid
    """
    if settings.storage_type == "local":
        return LocalStorageService(
            base_path=settings.storage_base_path,
            create_dirs=True
        )
    
    elif settings.storage_type == "s3":
        if not settings.s3_bucket_name:
            raise ValueError("S3_BUCKET_NAME must be set when using S3 storage")
        
        return S3StorageService(
            bucket_name=settings.s3_bucket_name,
            region_name=settings.s3_region_name,
            aws_access_key_id=settings.s3_access_key_id,
            aws_secret_access_key=settings.s3_secret_access_key,
            prefix=settings.s3_prefix
        )
    
    else:
        raise ValueError(f"Unsupported storage type: {settings.storage_type}")


def get_storage_service() -> StorageService:
    """
    Get a storage service instance using global settings.
    
    Returns:
        StorageService: Configured storage service instance
    """
    from app.config import get_global_settings
    settings = get_global_settings()
    return create_storage_service(settings)
