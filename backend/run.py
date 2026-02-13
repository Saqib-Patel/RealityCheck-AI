import os
import logging

# ── Memory-saving env vars (must be set BEFORE any heavy import) ─────────
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:128')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from app import create_app, socketio  # noqa: E402

app = create_app()

if __name__ == '__main__':
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV', 'development') == 'development'

    logger.info("RealityCheck AI API starting …")
    logger.info(f"Server: http://{host}:{port}")
    logger.info(f"Mode: {'Development' if debug else 'Production'}")
    logger.info(f"ML model:  {os.environ.get('USE_ML_MODEL', 'true')}")
    logger.info(f"Artifacts: {os.environ.get('ENABLE_ARTIFACT_ANALYSIS', 'true')}")

    socketio.run(app, host=host, port=port, debug=debug)
