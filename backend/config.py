import os
from pathlib import Path


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-prod')
    DEBUG = False
    TESTING = False

    BASE_DIR = Path(__file__).parent.parent
    UPLOAD_FOLDER = BASE_DIR / 'uploads'
    RESULTS_FOLDER = BASE_DIR / 'results'
    MODELS_FOLDER = BASE_DIR / 'model_weights'

    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB hard cap
    ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp'}
    ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'webm'}

    DEFAULT_MODEL = 'EfficientNetAutoAttB4'
    DEFAULT_DATASET = 'DFDC'
    DEFAULT_THRESHOLD = 0.5
    DEFAULT_VIDEO_FRAMES = 30          # reduced from 50 for memory

    # CORS: Allow Vercel domains and localhost
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*').split(',')
    SOCKETIO_ASYNC_MODE = 'gevent'     # gevent replaces deprecated eventlet

    # ── Free-tier memory flags (override via env vars) ──────────────
    ENABLE_ARTIFACT_ANALYSIS = os.environ.get('ENABLE_ARTIFACT_ANALYSIS', 'true').lower() == 'true'
    USE_ML_MODEL = os.environ.get('USE_ML_MODEL', 'true').lower() == 'true'
    MAX_IMAGE_DIM = int(os.environ.get('MAX_IMAGE_DIM', '512'))


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False
    SECRET_KEY = os.environ.get('SECRET_KEY')

    def __init__(self):
        if not self.SECRET_KEY:
            raise ValueError('SECRET_KEY env var is required in production')


class TestingConfig(Config):
    TESTING = True
    DEBUG = True


config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


def get_config():
    env = os.environ.get('FLASK_ENV', 'development')
    return config.get(env, config['default'])
