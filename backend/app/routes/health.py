import os
import gc
from flask import Blueprint, jsonify

health_bp = Blueprint('health', __name__)


@health_bp.route('/', methods=['GET'])
def health_check():
    """Lightweight health check – no heavy imports, no torch."""
    return jsonify({'status': 'healthy', 'service': 'deepfake-detection-api', 'version': '3.0.0'})


@health_bp.route('/ready', methods=['GET'])
def readiness_check():
    """Report whether the ML model is loaded (lazy – only checks, never loads)."""
    from app import DETECTOR
    model_loaded = DETECTOR is not None
    return jsonify({
        'status': 'ready' if model_loaded else 'warming',
        'model_loaded': model_loaded,
        'mode': 'cpu',
    })


@health_bp.route('/live', methods=['GET'])
def liveness_check():
    return jsonify({'status': 'alive'})


@health_bp.route('/memory', methods=['GET'])
def memory_check():
    """Diagnostic endpoint – reports RSS in MB."""
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB → MB (Linux)
    except Exception:
        rss = -1
    gc.collect()
    return jsonify({'rss_mb': round(rss, 1)})
