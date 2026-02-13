from typing import Dict, List, Optional, Tuple, Any

VALID_MODELS = ['EfficientNetB4', 'EfficientNetB4ST', 'EfficientNetAutoAttB4', 'EfficientNetAutoAttB4ST']
VALID_DATASETS = ['DFDC', 'FFPP']


def validate_file(filename: str, allowed_extensions: set) -> Tuple[bool, Optional[str]]:
    if '.' not in filename:
        return False, "File has no extension"
    ext = filename.rsplit('.', 1)[1].lower()
    if ext not in allowed_extensions:
        return False, f"Invalid extension. Allowed: {', '.join(allowed_extensions)}"
    return True, None


def validate_analysis_params(
    model: Optional[str] = None,
    dataset: Optional[str] = None,
    threshold: Optional[float] = None,
    frames: Optional[int] = None
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    validated = {}

    if model is not None and model not in VALID_MODELS:
        return False, f"Invalid model. Options: {', '.join(VALID_MODELS)}", {}

    if dataset is not None and dataset not in VALID_DATASETS:
        return False, f"Invalid dataset. Options: {', '.join(VALID_DATASETS)}", {}

    if threshold is not None:
        try:
            threshold = float(threshold)
            if not 0 <= threshold <= 1:
                return False, "Threshold must be between 0 and 1", {}
            validated['threshold'] = threshold
        except ValueError:
            return False, "Threshold must be a number", {}

    if frames is not None:
        try:
            frames = int(frames)
            if not 1 <= frames <= 200:
                return False, "Frames must be between 1 and 200", {}
            validated['frames'] = frames
        except ValueError:
            return False, "Frames must be an integer", {}

    if model: validated['model'] = model
    if dataset: validated['dataset'] = dataset

    return True, None, validated


def validate_batch_params(files: List[Dict]) -> Tuple[bool, Optional[str]]:
    if not files:
        return False, "No files provided"
    if len(files) > 20:
        return False, "Max 20 files per batch"
    return True, None
