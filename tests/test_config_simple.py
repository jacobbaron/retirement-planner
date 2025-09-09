"""Simple tests for configuration management."""

import tempfile
import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from app.config import Settings, get_settings


class TestConfigSimple:
    """Simple test cases for configuration."""

    def test_secret_key_validation_works(self):
        """Test that SECRET_KEY validation works correctly."""
        # Test with valid secret key
        with patch.dict(os.environ, {"SECRET_KEY": "valid-secret-123"}, clear=True):
            settings = Settings()
            assert settings.secret_key == "valid-secret-123"

        # Test with placeholder secret key (should fail)
        with patch.dict(os.environ, {"SECRET_KEY": "your-secret-key-here-change-in-production"}, clear=True):
            with pytest.raises(ValidationError) as exc_info:
                Settings()
            assert "SECRET_KEY must be set to a secure value" in str(exc_info.value)

    def test_app_env_validation_works(self):
        """Test that APP_ENV validation works correctly."""
        # Test with valid environment
        with patch.dict(os.environ, {"SECRET_KEY": "valid-secret-123", "APP_ENV": "production"}, clear=True):
            settings = Settings()
            assert settings.app_env == "production"

        # Test with invalid environment (should fail)
        with patch.dict(os.environ, {"SECRET_KEY": "valid-secret-123", "APP_ENV": "invalid"}, clear=True):
            with pytest.raises(ValidationError) as exc_info:
                Settings()
            assert "APP_ENV must be one of" in str(exc_info.value)

    def test_log_level_validation_works(self):
        """Test that LOG_LEVEL validation works correctly."""
        # Test with valid log level
        with patch.dict(os.environ, {"SECRET_KEY": "valid-secret-123", "LOG_LEVEL": "DEBUG"}, clear=True):
            settings = Settings()
            assert settings.log_level == "DEBUG"

        # Test with invalid log level (should fail)
        with patch.dict(os.environ, {"SECRET_KEY": "valid-secret-123", "LOG_LEVEL": "INVALID"}, clear=True):
            with pytest.raises(ValidationError) as exc_info:
                Settings()
            assert "LOG_LEVEL must be one of" in str(exc_info.value)

    def test_env_file_loading_works(self):
        """Test that .env file loading works correctly."""
        # Create a temporary .env file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("SECRET_KEY=test-secret-from-file\n")
            f.write("APP_ENV=testing\n")
            f.write("LOG_LEVEL=ERROR\n")
            temp_env_file = f.name

        try:
            # Test loading from the temporary file (disable default .env loading)
            with patch.dict(os.environ, {}, clear=True):
                settings = Settings(_env_file=temp_env_file)
                assert settings.secret_key == "test-secret-from-file"
                assert settings.app_env == "testing"
                assert settings.log_level == "ERROR"
        finally:
            os.unlink(temp_env_file)

    def test_get_settings_function_works(self):
        """Test that get_settings function works correctly."""
        with patch.dict(os.environ, {"SECRET_KEY": "valid-secret-123"}, clear=True):
            settings = get_settings()
            assert isinstance(settings, Settings)
            assert settings.secret_key == "valid-secret-123"

    def test_missing_secret_key_fails(self):
        """Test that missing SECRET_KEY causes failure."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValidationError) as exc_info:
                Settings(_env_file=None)
            assert "SECRET_KEY" in str(exc_info.value)
