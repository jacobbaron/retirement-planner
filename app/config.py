"""Application configuration management using Pydantic Settings."""

from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Application Environment
    app_env: str = Field(default="development", alias="APP_ENV")
    
    # Flask Configuration
    secret_key: str = Field(..., alias="SECRET_KEY")
    flask_app: str = Field(default="wsgi.py", alias="FLASK_APP")
    flask_env: str = Field(default="development", alias="FLASK_ENV")
    
    # Database Configuration
    db_url: str = Field(
        default="postgresql://retirement_user:retirement_password@localhost:5432/retirement_planner",
        alias="DB_URL"
    )
    
    # Redis Configuration
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v):
        """Ensure SECRET_KEY is provided and not a placeholder."""
        if not v or v == "your-secret-key-here-change-in-production":
            raise ValueError("SECRET_KEY must be set to a secure value")
        return v

    @field_validator("app_env")
    @classmethod
    def validate_app_env(cls, v):
        """Validate application environment."""
        allowed_envs = {"development", "testing", "production"}
        if v not in allowed_envs:
            raise ValueError(f"APP_ENV must be one of {allowed_envs}")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level."""
        allowed_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed_levels:
            raise ValueError(f"LOG_LEVEL must be one of {allowed_levels}")
        return v.upper()


def get_settings(env_file: Optional[str] = None) -> Settings:
    """Get application settings instance."""
    if env_file:
        return Settings(_env_file=env_file)
    return Settings()


# Global settings instance - will be created when first imported
_settings: Optional[Settings] = None


def get_global_settings() -> Settings:
    """Get or create global settings instance."""
    global _settings
    if _settings is None:
        _settings = get_settings()
    return _settings


# For backward compatibility
settings = get_global_settings()
