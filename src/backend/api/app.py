"""
Flask application factory for the holographic fixed points API.

This module provides:
- Application factory pattern for flexible configuration
- Custom JSON encoder for NumPy types
- CORS configuration for frontend access
- Blueprint registration for modular routing
"""

import json
import os
from typing import Any

import numpy as np
from flask import Flask, jsonify


class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, complex):
            return {"real": obj.real, "imag": obj.imag}
        return super().default(obj)


def create_app(config_name: str = "development") -> Flask:
    """
    Application factory for the Flask API.

    Args:
        config_name: Configuration profile ('development', 'production', 'testing').

    Returns:
        Configured Flask application.
    """
    app = Flask(__name__)

    # Configuration
    app.config.update(
        SECRET_KEY=os.environ.get("SECRET_KEY", "dev-key-change-in-production"),
        JSON_SORT_KEYS=False,
        # CORS headers
        ACCESS_CONTROL_ALLOW_ORIGIN="*",
    )

    # Custom JSON encoder for NumPy
    app.json_encoder = NumpyJSONEncoder

    # Register blueprints
    from .routes import simulations_bp, precomputed_bp

    app.register_blueprint(simulations_bp, url_prefix="/api/v1/simulations")
    app.register_blueprint(precomputed_bp, url_prefix="/api/v1/data")

    # Health check endpoint
    @app.route("/health")
    def health_check():
        return jsonify({"status": "healthy", "version": "0.1.0"})

    # API info
    @app.route("/api/v1")
    def api_info():
        return jsonify({
            "name": "APL Holographic Fixed Points API",
            "version": "0.1.0",
            "endpoints": {
                "simulations": "/api/v1/simulations",
                "precomputed": "/api/v1/data",
            },
            "solvers": ["kuramoto", "bekenstein", "turing", "criticality"],
        })

    # CORS headers
    @app.after_request
    def add_cors_headers(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return response

    return app


def run_dev_server():
    """Run the development server."""
    app = create_app("development")
    app.run(host="0.0.0.0", port=5000, debug=True)


if __name__ == "__main__":
    run_dev_server()
