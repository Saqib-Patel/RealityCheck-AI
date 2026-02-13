from flask import Blueprint, jsonify
import torch

health_bp = Blueprint('health', __name__)


@health_bp.route('/', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'service': 'deepfake-detection-api', 'version': '2.0.0'})


@health_bp.route('/ready', methods=['GET'])
def readiness_check():
    gpu = torch.cuda.is_available()
    return jsonify({
        'status': 'ready',
        'gpu_available': gpu,
        'gpu_name': torch.cuda.get_device_name(0) if gpu else None,
        'torch_version': torch.__version__
    })


@health_bp.route('/live', methods=['GET'])
def liveness_check():
    return jsonify({'status': 'alive'})
