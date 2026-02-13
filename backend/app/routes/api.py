import os
import uuid
import logging
import threading
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename

from app import get_detector
from app.services.result_manager import ResultManager
from app.utils.validators import validate_analysis_params
from app.services.optimized_detector import downscale_image_file

logger = logging.getLogger(__name__)

api_bp = Blueprint('api', __name__)

# Use the GLOBAL singleton detector – never instantiate per-request
result_manager = ResultManager()

# Track in-progress analyses
_analysis_status = {}


def _save_and_downscale(file, analysis_id, upload_folder):
    """Save upload, then immediately downscale to ≤512 px / 85 % JPEG."""
    filename = secure_filename(f"{analysis_id}_{file.filename}")
    filepath = os.path.join(upload_folder, filename)
    file.save(filepath)

    # Reject files > 10 MB even if Flask didn't catch it
    if os.path.getsize(filepath) > 10 * 1024 * 1024:
        os.remove(filepath)
        return None, 'File too large (max 10 MB)'

    # Downscale immediately to save memory during inference
    downscale_image_file(filepath, max_dim=512, quality=85)
    return filepath, None


def _run_image_analysis_async(filepath, model, dataset, threshold, analysis_id, app):
    """Run image analysis in background thread."""
    with app.app_context():
        try:
            _analysis_status[analysis_id] = {'status': 'processing', 'progress': 0}
            detector = get_detector()
            result = detector.analyze_image(
                image_path=filepath, model=model, dataset=dataset,
                threshold=threshold, analysis_id=analysis_id
            )
            result_manager.save_result(analysis_id, result)
            _analysis_status[analysis_id] = {'status': 'completed', 'progress': 100}
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            _analysis_status[analysis_id] = {'status': 'failed', 'error': str(e)}
            result_manager.save_result(analysis_id, {
                'analysis_id': analysis_id, 'status': 'failed', 'error': str(e)
            })
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)


def _run_video_analysis_async(filepath, model, dataset, threshold, frames, analysis_id, app):
    """Run video analysis in background thread."""
    with app.app_context():
        try:
            _analysis_status[analysis_id] = {'status': 'processing', 'progress': 0}
            detector = get_detector()
            result = detector.analyze_video(
                video_path=filepath, model=model, dataset=dataset,
                threshold=threshold, frames=frames, analysis_id=analysis_id
            )
            result_manager.save_result(analysis_id, result)
            _analysis_status[analysis_id] = {'status': 'completed', 'progress': 100}
        except Exception as e:
            logger.error(f"Video analysis failed: {e}")
            _analysis_status[analysis_id] = {'status': 'failed', 'error': str(e)}
            result_manager.save_result(analysis_id, {
                'analysis_id': analysis_id, 'status': 'failed', 'error': str(e)
            })
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)


def allowed_file(filename, file_type='image'):
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    if file_type == 'image':
        return ext in current_app.config['ALLOWED_IMAGE_EXTENSIONS']
    elif file_type == 'video':
        return ext in current_app.config['ALLOWED_VIDEO_EXTENSIONS']
    return False


@api_bp.route('/models', methods=['GET'])
def get_models():
    models = [
        {
            'id': 'EfficientNetB4',
            'name': 'EfficientNet-B4',
            'description': 'Standard EfficientNet for face manipulation detection',
            'accuracy': {'DFDC': 0.90, 'FFPP': 0.92},
            'speed': 'fast',
            'recommended': False
        },
        {
            'id': 'EfficientNetB4ST',
            'name': 'EfficientNet-B4 ST',
            'description': 'EfficientNet with Siamese Tuning for better generalization',
            'accuracy': {'DFDC': 0.92, 'FFPP': 0.94},
            'speed': 'fast',
            'recommended': False
        },
        {
            'id': 'EfficientNetAutoAttB4',
            'name': 'EfficientNet Auto-Attention B4',
            'description': 'Auto-attention mechanism for precise fake region detection',
            'accuracy': {'DFDC': 0.93, 'FFPP': 0.95},
            'speed': 'medium',
            'recommended': True
        },
        {
            'id': 'EfficientNetAutoAttB4ST',
            'name': 'EfficientNet Auto-Attention B4 ST',
            'description': 'Best model — auto-attention + siamese tuning',
            'accuracy': {'DFDC': 0.94, 'FFPP': 0.96},
            'speed': 'medium',
            'recommended': False
        }
    ]

    datasets = [
        {
            'id': 'DFDC',
            'name': 'DeepFake Detection Challenge',
            'description': "Facebook's large-scale deepfake dataset",
            'size': '100,000+ videos'
        },
        {
            'id': 'FFPP',
            'name': 'FaceForensics++',
            'description': 'Academic benchmark with Face2Face, FaceSwap, etc.',
            'size': '1,000+ source videos'
        }
    ]

    return jsonify({
        'models': models,
        'datasets': datasets,
        'defaults': {
            'model': current_app.config['DEFAULT_MODEL'],
            'dataset': current_app.config['DEFAULT_DATASET'],
            'threshold': current_app.config['DEFAULT_THRESHOLD']
        }
    })


@api_bp.route('/analyze/image', methods=['POST'])
def analyze_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename, 'image'):
        return jsonify({'error': 'Invalid file type. Allowed: jpg, jpeg, png, webp'}), 400

    model = request.form.get('model', current_app.config['DEFAULT_MODEL'])
    dataset = request.form.get('dataset', current_app.config['DEFAULT_DATASET'])
    threshold = float(request.form.get('threshold', current_app.config['DEFAULT_THRESHOLD']))

    # validate before doing anything expensive
    is_valid, err, _ = validate_analysis_params(model=model, dataset=dataset, threshold=threshold)
    if not is_valid:
        return jsonify({'error': err}), 400

    analysis_id = str(uuid.uuid4())
    filepath, err = _save_and_downscale(file, analysis_id, current_app.config['UPLOAD_FOLDER'])
    if err:
        return jsonify({'error': err}), 400

    # Run analysis in background thread - return immediately
    _analysis_status[analysis_id] = {'status': 'processing', 'progress': 0}
    thread = threading.Thread(
        target=_run_image_analysis_async,
        args=(filepath, model, dataset, threshold, analysis_id, current_app._get_current_object())
    )
    thread.daemon = True
    thread.start()

    return jsonify({
        'analysis_id': analysis_id,
        'status': 'processing',
        'message': 'Analysis started. Poll /api/v1/results/{id} or listen to WebSocket for progress.'
    }), 202


@api_bp.route('/analyze/video', methods=['POST'])
def analyze_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename, 'video'):
        return jsonify({'error': 'Invalid file type. Allowed: mp4, avi, mov, webm'}), 400

    model = request.form.get('model', current_app.config['DEFAULT_MODEL'])
    dataset = request.form.get('dataset', current_app.config['DEFAULT_DATASET'])
    threshold = float(request.form.get('threshold', current_app.config['DEFAULT_THRESHOLD']))
    frames = int(request.form.get('frames', current_app.config['DEFAULT_VIDEO_FRAMES']))

    is_valid, err, _ = validate_analysis_params(model=model, dataset=dataset, threshold=threshold, frames=frames)
    if not is_valid:
        return jsonify({'error': err}), 400

    analysis_id = str(uuid.uuid4())
    filename = secure_filename(f"{analysis_id}_{file.filename}")
    filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Reject oversized files
    if os.path.getsize(filepath) > 10 * 1024 * 1024:
        os.remove(filepath)
        return jsonify({'error': 'File too large (max 10 MB)'}), 400

    # Run analysis in background thread - return immediately
    _analysis_status[analysis_id] = {'status': 'processing', 'progress': 0}
    thread = threading.Thread(
        target=_run_video_analysis_async,
        args=(filepath, model, dataset, threshold, frames, analysis_id, current_app._get_current_object())
    )
    thread.daemon = True
    thread.start()

    return jsonify({
        'analysis_id': analysis_id,
        'status': 'processing',
        'message': 'Analysis started. Poll /api/v1/results/{id} or listen to WebSocket for progress.'
    }), 202


@api_bp.route('/analyze/batch', methods=['POST'])
def analyze_batch():
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400

    files = request.files.getlist('files')
    if len(files) == 0:
        return jsonify({'error': 'No files selected'}), 400
    if len(files) > 20:
        return jsonify({'error': 'Max 20 files per batch'}), 400

    model = request.form.get('model', current_app.config['DEFAULT_MODEL'])
    dataset = request.form.get('dataset', current_app.config['DEFAULT_DATASET'])
    threshold = float(request.form.get('threshold', current_app.config['DEFAULT_THRESHOLD']))

    batch_id = str(uuid.uuid4())
    file_list = []
    for file in files:
        if file.filename == '':
            continue
        file_id = str(uuid.uuid4())
        filename = secure_filename(f"{file_id}_{file.filename}")
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        file_type = 'video' if allowed_file(file.filename, 'video') else 'image'
        file_list.append({'id': file_id, 'path': filepath, 'original_name': file.filename, 'type': file_type})

    result = get_detector().analyze_batch(
        files=file_list, model=model, dataset=dataset,
        threshold=threshold, batch_id=batch_id
    )

    return jsonify({'batch_id': batch_id, 'file_count': len(file_list), 'status': 'processing', 'results': result})


@api_bp.route('/compare', methods=['POST'])
def compare_models():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    models = request.form.getlist('models') or ['EfficientNetB4', 'EfficientNetAutoAttB4']
    dataset = request.form.get('dataset', current_app.config['DEFAULT_DATASET'])
    threshold = float(request.form.get('threshold', current_app.config['DEFAULT_THRESHOLD']))

    comparison_id = str(uuid.uuid4())
    filename = secure_filename(f"{comparison_id}_{file.filename}")
    filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    file_type = 'video' if allowed_file(file.filename, 'video') else 'image'

    try:
        results = get_detector().compare_models(
            file_path=filepath, file_type=file_type,
            models=models, dataset=dataset, threshold=threshold
        )
        return jsonify({'comparison_id': comparison_id, 'file_type': file_type, 'results': results})
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


@api_bp.route('/results', methods=['GET'])
def get_results():
    limit = request.args.get('limit', 20, type=int)
    offset = request.args.get('offset', 0, type=int)
    results = result_manager.get_all_results(limit=limit, offset=offset)
    return jsonify({'results': results, 'limit': limit, 'offset': offset, 'total': result_manager.get_total_count()})


@api_bp.route('/results/<result_id>', methods=['GET'])
def get_result(result_id):
    # First check if analysis is still in progress
    if result_id in _analysis_status:
        status_info = _analysis_status[result_id]
        if status_info['status'] == 'processing':
            return jsonify({
                'analysis_id': result_id,
                'status': 'processing',
                'progress': status_info.get('progress', 0),
                'message': 'Analysis in progress...'
            }), 202
        elif status_info['status'] == 'failed':
            return jsonify({
                'analysis_id': result_id,
                'status': 'failed',
                'error': status_info.get('error', 'Unknown error')
            }), 500

    # Check for completed result
    result = result_manager.get_result(result_id)
    if result is None:
        return jsonify({'error': 'Result not found'}), 404
    return jsonify(result)


@api_bp.route('/results/<result_id>', methods=['DELETE'])
def delete_result(result_id):
    if not result_manager.delete_result(result_id):
        return jsonify({'error': 'Result not found'}), 404
    return jsonify({'message': 'Deleted', 'id': result_id})
