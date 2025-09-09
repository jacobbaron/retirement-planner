"""Tests for application configuration management."""

import os
import tempfile
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from app.config import Settings, get_settings


class TestSettings:
    """Test cases for Settings class."""

    def test_settings_loads_from_env_example(self):
        """Test that settings can load from .env.example file."""
        # Create a temporary .env file based on .env.example
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("APP_ENV=development\n")
            f.write("SECRET_KEY=test-secret-key-123\n")
            f.write("DB_URL=postgresql://test:test@localhost:5432/test\n")
            f.write("REDIS_URL=redis://localhost:6379/0\n")
            f.write("LOG_LEVEL=DEBUG\n")
            temp_env_file = f.name

        try:
            with patch.dict(os.environ, {}, clear=True):
                settings = Settings(_env_file=temp_env_file)

                assert settings.app_env == "development"
                assert settings.secret_key == "test-secret-key-123"
                assert settings.db_url == "postgresql://test:test@localhost:5432/test"
                assert settings.redis_url == "redis://localhost:6379/0"
                assert settings.log_level == "DEBUG"
        finally:
            os.unlink(temp_env_file)

    def test_missing_secret_key_raises_exception(self):
        """Test that missing SECRET_KEY raises ValidationError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValidationError) as exc_info:
                Settings(_env_file=None)

            # Check that the error is about SECRET_KEY
            assert "SECRET_KEY" in str(exc_info.value)

    def test_placeholder_secret_key_raises_exception(self):
        """Test that placeholder SECRET_KEY raises ValidationError."""
        with patch.dict(
            os.environ,
            {"SECRET_KEY": "your-secret-key-here-change-in-production"},
            clear=True,
        ):
            with pytest.raises(ValidationError) as exc_info:
                Settings()

            # Check that the error is about SECRET_KEY
            assert "SECRET_KEY must be set to a secure value" in str(exc_info.value)

    def test_valid_secret_key_passes(self):
        """Test that valid SECRET_KEY passes validation."""
        with patch.dict(os.environ, {"SECRET_KEY": "valid-secret-key-123"}, clear=True):
            settings = Settings()
            assert settings.secret_key == "valid-secret-key-123"

    def test_app_env_validation(self):
        """Test APP_ENV validation."""
        with patch.dict(
            os.environ,
            {"SECRET_KEY": "valid-secret-key-123", "APP_ENV": "invalid-env"},
            clear=True,
        ):
            with pytest.raises(ValidationError) as exc_info:
                Settings()

            assert "APP_ENV must be one of" in str(exc_info.value)

    def test_log_level_validation(self):
        """Test LOG_LEVEL validation."""
        with patch.dict(
            os.environ,
            {"SECRET_KEY": "valid-secret-key-123", "LOG_LEVEL": "INVALID_LEVEL"},
            clear=True,
        ):
            with pytest.raises(ValidationError) as exc_info:
                Settings()

            assert "LOG_LEVEL must be one of" in str(exc_info.value)

    def test_log_level_case_insensitive(self):
        """Test that LOG_LEVEL is case insensitive."""
        with patch.dict(
            os.environ,
            {"SECRET_KEY": "valid-secret-key-123", "LOG_LEVEL": "debug"},
            clear=True,
        ):
            settings = Settings()
            assert settings.log_level == "DEBUG"

    def test_default_values(self):
        """Test default values when environment variables are not set."""
        with patch.dict(os.environ, {"SECRET_KEY": "valid-secret-key-123"}, clear=True):
            settings = Settings()

            assert settings.app_env == "development"
            assert settings.flask_app == "wsgi.py"
            assert settings.flask_env == "development"
            # Note: In Docker environment, DB_URL defaults to use 'db' hostname
            assert (
                "postgresql://retirement_user:retirement_password@" in settings.db_url
            )
            # Redis URL may be different in Docker environment
            assert "redis://" in settings.redis_url
            assert settings.log_level == "INFO"

    def test_get_settings_function(self):
        """Test the get_settings function."""
        with patch.dict(os.environ, {"SECRET_KEY": "valid-secret-key-123"}, clear=True):
            settings = get_settings()
            assert isinstance(settings, Settings)
            assert settings.secret_key == "valid-secret-key-123"

    def test_environment_variable_aliases(self):
        """Test that environment variable aliases work correctly."""
        with patch.dict(
            os.environ,
            {
                "SECRET_KEY": "valid-secret-key-123",
                "APP_ENV": "production",
                "DB_URL": "postgresql://prod:prod@prod:5432/prod",
                "REDIS_URL": "redis://prod:6379/1",
                "LOG_LEVEL": "ERROR",
            },
            clear=True,
        ):
            settings = Settings()

            assert settings.app_env == "production"
            assert settings.db_url == "postgresql://prod:prod@prod:5432/prod"
            assert settings.redis_url == "redis://prod:6379/1"
            assert settings.log_level == "ERROR"
