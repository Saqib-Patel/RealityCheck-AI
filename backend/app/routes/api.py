import os
import uuid
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename

from app.services.detector import DeepFakeDetector
from app.services.result_manager import ResultManager
from app.utils.validators import validate_analysis_params

api_bp = Blueprint('api', __name__)

detector = DeepFakeDetector()
result_manager = ResultManager()


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
    filename = secure_filename(f"{analysis_id}_{file.filename}")
    filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        result = detector.analyze_image(
            image_path=filepath, model=model, dataset=dataset,
            threshold=threshold, analysis_id=analysis_id
        )
        result_manager.save_result(analysis_id, result)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e), 'analysis_id': analysis_id, 'status': 'failed'}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


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

    try:
        result = detector.analyze_video(
            video_path=filepath, model=model, dataset=dataset,
            threshold=threshold, frames=frames, analysis_id=analysis_id
        )
        result_manager.save_result(analysis_id, result)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e), 'analysis_id': analysis_id, 'status': 'failed'}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


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

    result = detector.analyze_batch(
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
        results = detector.compare_models(
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
    result = result_manager.get_result(result_id)
    if result is None:
        return jsonify({'error': 'Result not found'}), 404
    return jsonify(result)


@api_bp.route('/results/<result_id>', methods=['DELETE'])
def delete_result(result_id):
    if not result_manager.delete_result(result_id):
        return jsonify({'error': 'Result not found'}), 404
    return jsonify({'message': 'Deleted', 'id': result_id})
