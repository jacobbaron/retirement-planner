"""Health check blueprint."""

from flask import Blueprint, Response, jsonify

health_bp = Blueprint("health", __name__)


@health_bp.route("/healthz")
def health_check() -> Response:
    """Health check endpoint.

    Returns:
        JSON response with status information
    """
    return jsonify({"status": "ok"})
