import os
from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO

from config import get_config

socketio = SocketIO(cors_allowed_origins="*", async_mode='threading')


def create_app(config_class=None):
    app = Flask(__name__)

    if config_class is None:
        config_class = get_config()
    app.config.from_object(config_class)

    CORS(app, origins=app.config.get('CORS_ORIGINS', '*'), supports_credentials=True)
    socketio.init_app(app)

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

    from app.routes.api import api_bp
    from app.routes.health import health_bp

    app.register_blueprint(api_bp, url_prefix='/api/v1')
    app.register_blueprint(health_bp, url_prefix='/health')

    from app.websocket import handlers  # noqa: F401

    @app.route('/')
    def root():
        return {
            'name': 'RealityCheck AI API',
            'version': '2.0.0',
            'endpoints': {
                'health': '/health/',
                'models': '/api/v1/models',
                'analyze_image': '/api/v1/analyze/image',
                'analyze_video': '/api/v1/analyze/video',
            },
            'docs': 'https://github.com/Saqib-Patel/RealityCheck-AI'
        }

    @app.errorhandler(404)
    def not_found(error):
        return {'error': 'Not found', 'status': 404}, 404

    @app.errorhandler(500)
    def internal_error(error):
        return {'error': 'Internal server error', 'status': 500}, 500

    @app.after_request
    def after_request(response):
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        return response

    return app
