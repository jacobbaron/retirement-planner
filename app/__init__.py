"""Retirement Planner Flask Application Factory."""

from typing import Optional

from flask import Flask

from app.config import get_global_settings


def create_app(config_name: Optional[str] = None) -> Flask:
    """Create and configure the Flask application.

    Args:
        config_name: Configuration name (development, testing, production)

    Returns:
        Flask: Configured Flask application instance
    """
    app = Flask(__name__)

    # Configuration from Pydantic Settings
    settings = get_global_settings()
    app.config["SECRET_KEY"] = settings.secret_key
    app.config["DATABASE_URL"] = settings.db_url
    app.config["REDIS_URL"] = settings.redis_url
    app.config["ENV"] = settings.flask_env
    app.config["DEBUG"] = settings.app_env == "development"

    # Register blueprints
    from app.blueprints.health import health_bp

    app.register_blueprint(health_bp)

    return app
