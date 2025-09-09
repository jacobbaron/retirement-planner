"""Health check blueprint."""

from flask import Blueprint, jsonify

health_bp = Blueprint('health', __name__)


@health_bp.route('/healthz')
def health_check():
    """Health check endpoint.
    
    Returns:
        JSON response with status information
    """
    return jsonify({"status": "ok"})
