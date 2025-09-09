"""Tests for Flask application startup with configuration."""

import os
from unittest.mock import patch

import pytest

from app import create_app
from app.config import reset_global_settings, Settings


class TestAppStartup:
    """Test cases for Flask application startup."""

    def test_app_creation_with_valid_config(self):
        """Test that app creates successfully with valid configuration."""
        # Reset global settings to allow environment patching
        reset_global_settings()
        
        with patch.dict(os.environ, {
            "SECRET_KEY": "valid-secret-key-123"
        }, clear=True):
            app = create_app()
            
            assert app is not None
            assert app.config["SECRET_KEY"] == "valid-secret-key-123"
            assert "postgresql://retirement_user:retirement_password@" in app.config["DATABASE_URL"]
            assert app.config["REDIS_URL"] == "redis://redis:6379/0"

    def test_app_creation_fails_without_secret_key(self):
        """Test that app creation fails when SECRET_KEY is missing."""
        # Reset global settings to allow environment patching
        reset_global_settings()
        
        with patch.dict(os.environ, {}, clear=True):
            # Create a new Settings instance that will fail validation
            from pydantic import ValidationError
            with pytest.raises(ValidationError) as exc_info:
                # This should fail because SECRET_KEY is required
                from app.config import Settings
                Settings(_env_file=None)
            
            # Should raise a ValidationError from Pydantic
            assert "SECRET_KEY" in str(exc_info.value)

    def test_app_creation_fails_with_placeholder_secret_key(self):
        """Test that app creation fails with placeholder SECRET_KEY."""
        # Reset global settings to allow environment patching
        reset_global_settings()
        
        with patch.dict(os.environ, {
            "SECRET_KEY": "your-secret-key-here-change-in-production"
        }, clear=True):
            with pytest.raises(Exception) as exc_info:
                create_app()
            
            # Should raise a ValidationError from Pydantic
            assert "SECRET_KEY must be set to a secure value" in str(exc_info.value)

    def test_app_uses_custom_environment_variables(self):
        """Test that app uses custom environment variables."""
        # Reset global settings to allow environment patching
        reset_global_settings()
        
        with patch.dict(os.environ, {
            "SECRET_KEY": "custom-secret-key",
            "APP_ENV": "production",
            "DB_URL": "postgresql://custom:custom@custom:5432/custom",
            "REDIS_URL": "redis://custom:6379/2",
            "LOG_LEVEL": "ERROR"
        }, clear=True):
            app = create_app()
            
            assert app.config["SECRET_KEY"] == "custom-secret-key"
            assert app.config["DATABASE_URL"] == "postgresql://custom:custom@custom:5432/custom"
            assert app.config["REDIS_URL"] == "redis://custom:6379/2"
            assert app.config["ENV"] == "development"  # flask_env default
            assert app.config["DEBUG"] is False  # production env

    def test_app_debug_mode_based_on_environment(self):
        """Test that debug mode is set based on APP_ENV."""
        # Test development environment
        reset_global_settings()
        
        with patch.dict(os.environ, {
            "SECRET_KEY": "valid-secret-key-123",
            "APP_ENV": "development"
        }, clear=True):
            app = create_app()
            assert app.config["DEBUG"] is True

        # Test production environment
        reset_global_settings()
        with patch.dict(os.environ, {
            "SECRET_KEY": "valid-secret-key-123",
            "APP_ENV": "production"
        }, clear=True):
            app = create_app()
            assert app.config["DEBUG"] is False

        # Test testing environment
        _settings = None
        with patch.dict(os.environ, {
            "SECRET_KEY": "valid-secret-key-123",
            "APP_ENV": "testing"
        }, clear=True):
            app = create_app()
            assert app.config["DEBUG"] is False
