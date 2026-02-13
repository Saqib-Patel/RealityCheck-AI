import os
import logging
from app import create_app, socketio

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = create_app()

if __name__ == '__main__':
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV', 'development') == 'development'

    logger.info(f"DeepFake Detection Hub API starting...")
    logger.info(f"Server: http://{host}:{port}")
    logger.info(f"Mode: {'Development' if debug else 'Production'}")

    socketio.run(app, host=host, port=port, debug=debug)
