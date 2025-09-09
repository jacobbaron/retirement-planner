"""Retirement Planner Flask Application Factory."""

from flask import Flask


def create_app(config_name=None):
    """Create and configure the Flask application.
    
    Args:
        config_name: Configuration name (development, testing, production)
        
    Returns:
        Flask: Configured Flask application instance
    """
    app = Flask(__name__)
    
    # Basic configuration
    app.config['SECRET_KEY'] = 'dev-secret-key'  # Will be overridden by env config in EP-1-T3
    
    # Register blueprints
    from app.blueprints.health import health_bp
    app.register_blueprint(health_bp)
    
    return app
