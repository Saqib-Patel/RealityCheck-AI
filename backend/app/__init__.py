import os
import gc
import logging
from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO

from config import get_config

logger = logging.getLogger(__name__)

# ── Memory optimizations (set BEFORE any heavy imports) ──────────────────
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['OMP_NUM_THREADS'] = '1'          # single-threaded BLAS – saves ~50 MB
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# Force non-interactive matplotlib backend (no Tk/Qt)
try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass

# ── SocketIO – use gevent async mode (replaces eventlet) ────────────────
socketio = SocketIO(
    cors_allowed_origins="*",
    async_mode='gevent',
    ping_timeout=60,
    ping_interval=25,
    logger=False,
    engineio_logger=False,
)

# ── Global detector (loaded ONCE, reused across all requests) ───────────
DETECTOR = None


def get_detector():
    """Return the global detector, initialising lazily on first call."""
    global DETECTOR
    if DETECTOR is None:
        logger.info("Initialising global OptimizedDetector …")
        from app.services.optimized_detector import OptimizedDetector
        DETECTOR = OptimizedDetector()
        gc.collect()
        logger.info("OptimizedDetector ready.")
    return DETECTOR


def create_app(config_class=None):
    app = Flask(__name__)

    if config_class is None:
        config_class = get_config()
    app.config.from_object(config_class)

    # Hard-cap upload size to 10 MB (overrides any config value)
    app.config['MAX_CONTENT_LENGTH'] = min(
        app.config.get('MAX_CONTENT_LENGTH', 10 * 1024 * 1024),
        10 * 1024 * 1024,
    )

    CORS(app, resources={r"/*": {"origins": "*"}})
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
            'version': '3.0.0',
            'memory_mode': 'optimised-512mb',
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
