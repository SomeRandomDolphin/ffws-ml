import os
import logging
from flask import Flask, jsonify

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress tensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def create_app():
    """Application factory."""
    app = Flask('FFWS')

    # Register blueprints
    from api.routes import api_bp
    app.register_blueprint(api_bp)

    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({"error": "Endpoint not found"}), 404

    @app.errorhandler(500)
    def internal_error(error):
        logger.exception("Internal server error")
        return jsonify({"error": "Internal server error"}), 500

    @app.errorhandler(Exception)
    def handle_exception(error):
        logger.exception(f"Unhandled exception: {error}")
        return jsonify({"error": "An unexpected error occurred"}), 500

    logger.info("FFWS Application initialized")
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
