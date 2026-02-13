"""
OptimizedDetector – single-model, memory-safe deepfake detector for Render 512 MB.

Key design decisions
────────────────────
• ONE EfficientNetAutoAttB4 model (DFDC), loaded once at startup  (~180 MB)
• BlazeFace loaded lazily on first request                         (~  5 MB)
• Artifact analysis is configurable via config.ENABLE_ARTIFACT_ANALYSIS
• Every image is down-scaled to ≤512 px before *any* processing
• Face crops resized to 224×224 before inference
• Aggressive gc.collect() after every inference call
• No ensemble, no cross-validation – single model only
"""

import os
import sys
import gc
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

# ── Lazy-loaded heavy deps ──────────────────────────────────────────────
_torch = None
_BlazeFace = None
_FaceExtractor = None
_VideoReader = None
_fornet = None
_weights = None
_utils = None


def _lazy_import_torch():
    """Import PyTorch + model helpers once, on demand."""
    global _torch, _BlazeFace, _FaceExtractor, _VideoReader, _fornet, _weights, _utils
    if _torch is not None:
        return
    logger.info("[OptimizedDetector] Lazy-loading PyTorch …")
    import torch as t
    _torch = t
    from blazeface import BlazeFace as BF, FaceExtractor as FE, VideoReader as VR
    from architectures import fornet as fn, weights as w
    from isplutils import utils as u
    _BlazeFace = BF
    _FaceExtractor = FE
    _VideoReader = VR
    _fornet = fn
    _weights = w
    _utils = u
    logger.info("[OptimizedDetector] PyTorch loaded OK.")


# ── Image pre-processing helpers ────────────────────────────────────────

def downscale_image_file(image_path: str, max_dim: int = 512, quality: int = 85) -> str:
    """
    Resize the saved image to *max_dim* px on its longest side (in-place).
    Uses cv2.INTER_AREA for high-quality downscaling.
    Returns the (same) path for convenience.
    """
    img = cv2.imread(image_path)
    if img is None:
        return image_path
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return image_path                       # already small enough
    scale = max_dim / max(h, w)
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imwrite(image_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    logger.debug(f"Downscaled {w}×{h} → {img.shape[1]}×{img.shape[0]}")
    return image_path


def downscale_face_crop(face_img: np.ndarray, target: int = 224) -> np.ndarray:
    """Resize a face crop to *target*×*target* for EfficientNet-B4."""
    h, w = face_img.shape[:2]
    if max(h, w) <= target:
        return face_img
    return cv2.resize(face_img, (target, target), interpolation=cv2.INTER_AREA)


# ── Main detector class ─────────────────────────────────────────────────

class OptimizedDetector:
    """
    Memory-optimised deepfake detector – fits within Render free tier (512 MB).

    Memory budget (approximate):
        Python interpreter + Flask  ~  60 MB
        OpenCV / numpy / scipy      ~  40 MB
        PyTorch CPU runtime         ~ 100 MB
        EfficientNetAutoAttB4 (1)   ~ 180 MB
        BlazeFace                   ~   5 MB
        Headroom                    ~ 127 MB
        ─────────────────────────────────────
        Total                       ~ 400 MB  (100 MB buffer to 512 MB)
    """

    # The single model + dataset we load  (most accurate on real-world content)
    MODEL_NAME = 'EfficientNetAutoAttB4'
    DATASET = 'DFDC'
    FACE_POLICY = 'scale'
    FACE_SIZE = 224

    def __init__(self):
        self._model = None
        self._transformer = None
        self._facedet = None
        self._video_reader = None
        self._device = None
        self._torch_ready = False

        # Read config flags from environment
        self.enable_artifact = os.environ.get(
            'ENABLE_ARTIFACT_ANALYSIS', 'true'
        ).lower() == 'true'
        self.use_ml_model = os.environ.get(
            'USE_ML_MODEL', 'true'
        ).lower() == 'true'

        logger.info(
            f"[OptimizedDetector] created  "
            f"ml_model={self.use_ml_model}  artifacts={self.enable_artifact}"
        )

    # ── Lazy initialisers ───────────────────────────────────────────────

    def _ensure_torch(self):
        if self._torch_ready:
            return
        _lazy_import_torch()
        self._device = _torch.device('cpu')     # CPU-only on free tier
        self._torch_ready = True

    def _ensure_model(self):
        """Load the single EfficientNet model ONCE."""
        if self._model is not None:
            return
        self._ensure_torch()
        from torch.utils.model_zoo import load_url
        key = f"{self.MODEL_NAME}_{self.DATASET}"
        logger.info(f"[OptimizedDetector] Loading model {key} …")
        url = _weights.weight_url[key]
        net = getattr(_fornet, self.MODEL_NAME)().eval().to(self._device)
        net.load_state_dict(load_url(url, map_location=self._device, check_hash=True))
        self._model = net
        self._transformer = _utils.get_transformer(
            self.FACE_POLICY, self.FACE_SIZE,
            net.get_normalizer(), train=False,
        )
        gc.collect()
        logger.info(f"[OptimizedDetector] Model {key} loaded.")

    def _ensure_face_detector(self):
        """Load BlazeFace lazily."""
        if self._facedet is not None:
            return
        self._ensure_torch()
        blazeface_dir = BASE_DIR / 'blazeface'
        self._facedet = _BlazeFace().to(self._device)
        self._facedet.load_weights(str(blazeface_dir / "blazeface.pth"))
        self._facedet.load_anchors(str(blazeface_dir / "anchors.npy"))
        self._video_reader = _VideoReader(verbose=False)
        gc.collect()
        logger.info("[OptimizedDetector] BlazeFace loaded.")

    # ── Progress helper ─────────────────────────────────────────────────

    @staticmethod
    def _emit_progress(analysis_id: str, stage: str, progress: int, message: str):
        from app import socketio
        socketio.emit('analysis_progress', {
            'analysis_id': analysis_id,
            'stage': stage,
            'progress': progress,
            'message': message,
            'timestamp': datetime.now().isoformat(),
        }, namespace='/ws')

    # ── Memory cleanup ──────────────────────────────────────────────────

    @staticmethod
    def _cleanup():
        gc.collect()
        if _torch is not None and _torch.cuda.is_available():
            _torch.cuda.empty_cache()

    # ── Artifact analysis (pure numpy/scipy – no PyTorch) ───────────────

    def _analyze_image_artifacts(self, image_path: str) -> Dict[str, float]:
        """
        7-channel forensic artifact analysis.  Uses only numpy/scipy/cv2.
        Channels: noise, jpeg_ghost, edge_inconsistency, color_distribution,
        frequency, ela, texture.
        """
        from scipy.fft import dctn
        from scipy.ndimage import uniform_filter

        try:
            img = cv2.imread(image_path)
            if img is None:
                return self._empty_artifact_scores()

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
            h, w = gray.shape

            # 1. Frequency spectrum (DCT)
            dct = dctn(gray)
            dct_log = np.log1p(np.abs(dct))
            yy, xx = np.mgrid[0:h, 0:w]
            dist = np.sqrt((yy / h) ** 2 + (xx / w) ** 2)
            low_mask = dist < 0.15
            mid_mask = (dist >= 0.15) & (dist < 0.50)
            high_mask = dist >= 0.50
            very_high_mask = dist >= 0.75

            le = np.mean(dct_log[low_mask]) if np.any(low_mask) else 1.0
            me = np.mean(dct_log[mid_mask]) if np.any(mid_mask) else 0.0
            he = np.mean(dct_log[high_mask]) if np.any(high_mask) else 0.0
            vhe = np.mean(dct_log[very_high_mask]) if np.any(very_high_mask) else 0.0

            freq_ratio_s = float(np.clip(1.0 - ((he / (le + 1e-10)) / 0.25), 0, 1))
            vh_deficit_s = float(np.clip(1.0 - ((vhe / (le + 1e-10)) / 0.15), 0, 1))
            dropoff_s = float(np.clip(((me / (he + 1e-10)) - 1.0) / 1.2, 0, 1))
            hfv = dct_log[high_mask]
            spec_smooth_s = float(np.clip(1.0 - ((np.std(hfv) / (np.mean(hfv) + 1e-10)) / 0.6), 0, 1)) if len(hfv) > 10 else 0.0
            total_e = np.sum(dct_log)
            low_pct = np.sum(dct_log[low_mask]) / (total_e + 1e-10)
            energy_conc_s = float(np.clip((low_pct - 0.3) / 0.3, 0, 1))
            frequency_score = 0.25 * freq_ratio_s + 0.20 * vh_deficit_s + 0.20 * dropoff_s + 0.20 * spec_smooth_s + 0.15 * energy_conc_s

            # 2. Noise uniformity
            smoothed = uniform_filter(gray, size=3)
            noise = gray - smoothed
            bsz = max(12, min(h, w) // 12)
            bstds, bmeans = [], []
            for i in range(0, h - bsz, bsz):
                for j in range(0, w - bsz, bsz):
                    bstds.append(np.std(noise[i:i + bsz, j:j + bsz]))
                    bmeans.append(np.mean(gray[i:i + bsz, j:j + bsz]))
            if len(bstds) > 4:
                cv_n = np.std(bstds) / (np.mean(bstds) + 1e-10)
                corr = abs(np.corrcoef(bmeans, bstds)[0, 1]) if np.std(bmeans) > 1e-10 and np.std(bstds) > 1e-10 else 0.0
                if np.isnan(corr):
                    corr = 0.0
                uni_s = float(np.clip(1.0 - (cv_n / 0.40), 0, 1))
                dec_s = float(np.clip(1.0 - (corr / 0.25), 0, 1))
                low_n_s = float(np.clip(1.0 - (np.mean(bstds) / 2.5), 0, 1))
                flat = noise.flatten()
                kurt = float(np.mean(flat ** 4) / (np.mean(flat ** 2) ** 2 + 1e-10))
                kurt_s = float(np.clip(1.0 - ((kurt - 3.0) / 3.0), 0, 1))
                noise_score = 0.30 * uni_s + 0.25 * dec_s + 0.25 * low_n_s + 0.20 * kurt_s
            else:
                noise_score = 0.0

            # 3. Cross-channel coherence
            color_coherence_score = 0.0
            try:
                chs = [c.astype(np.float64) for c in cv2.split(img)]
                cn = [(c - uniform_filter(c, size=3)).flatten() for c in chs]
                ccs = []
                for ci in range(3):
                    for cj in range(ci + 1, 3):
                        r = abs(np.corrcoef(cn[ci], cn[cj])[0, 1])
                        if not np.isnan(r):
                            ccs.append(r)
                if ccs:
                    color_coherence_score = float(np.clip((np.mean(ccs) - 0.35) / 0.35, 0, 1))
            except Exception:
                pass

            # 4. Texture repetition
            texture_score = 0.0
            try:
                psz = min(64, min(h, w) // 4)
                if psz >= 32:
                    cy, cx = h // 2, w // 2
                    patch = gray[cy - psz:cy + psz, cx - psz:cx + psz]
                    fp = np.fft.fft2(patch - np.mean(patch))
                    ac = np.real(np.fft.ifft2(np.abs(fp) ** 2))
                    ac = ac / (ac[0, 0] + 1e-10)
                    cr = max(3, ac.shape[0] // 16)
                    am = ac.copy()
                    am[:cr, :cr] = am[:cr, -cr:] = am[-cr:, :cr] = am[-cr:, -cr:] = 0
                    texture_score = float(np.clip((np.max(np.abs(am)) - 0.10) / 0.20, 0, 1))
            except Exception:
                pass

            # 5. Edge coherence
            edge_score = 0.0
            try:
                et = cv2.Canny(img, 100, 200)
                el = cv2.Canny(img, 50, 100)
                tc = np.sum(et > 0)
                lc = np.sum(el > 0)
                ec_s = float(np.clip(((tc / (lc + 1e-10)) - 0.35) / 0.35, 0, 1)) if lc > 0 else 0.0
                sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                gm = np.sqrt(sx ** 2 + sy ** 2)
                emask = el > 0
                if np.sum(emask) > 100:
                    eg = gm[emask]
                    eu_s = float(np.clip(1.0 - ((np.std(eg) / (np.mean(eg) + 1e-10)) / 0.7), 0, 1))
                else:
                    eu_s = 0.0
                edge_score = 0.50 * ec_s + 0.50 * eu_s
            except Exception:
                pass

            # 6. Histogram smoothness
            histogram_score = 0.0
            try:
                hs = []
                for ci in range(3):
                    hist = cv2.calcHist([img], [ci], None, [256], [0, 256]).flatten()
                    hist = hist / (np.sum(hist) + 1e-10)
                    g2 = np.diff(np.diff(hist))
                    hs.append(float(np.clip(1.0 - (np.mean(np.abs(g2)) / 0.002), 0, 1)))
                histogram_score = float(np.mean(hs))
            except Exception:
                pass

            # 7. JPEG artefacts
            jpeg_score = 0.0
            try:
                if h >= 16 and w >= 16:
                    hb, ho, vb, vo = [], [], [], []
                    for c in range(1, w):
                        d = np.mean(np.abs(gray[:, c] - gray[:, c - 1]))
                        (hb if c % 8 == 0 else ho).append(d)
                    for r in range(1, h):
                        d = np.mean(np.abs(gray[r, :] - gray[r - 1, :]))
                        (vb if r % 8 == 0 else vo).append(d)
                    jh = (np.mean(hb) / (np.mean(ho) + 1e-10) > 1.02) if hb and ho else False
                    jv = (np.mean(vb) / (np.mean(vo) + 1e-10) > 1.02) if vb and vo else False
                    if not jh and not jv:
                        jpeg_score = 0.45
                    elif not jh or not jv:
                        jpeg_score = 0.25
                ext = os.path.splitext(image_path)[1].lower()
                if ext in ('.png', '.webp', '.bmp', '.tiff'):
                    jpeg_score = max(jpeg_score, 0.35)
            except Exception:
                pass

            combined = (
                0.22 * frequency_score + 0.20 * noise_score +
                0.15 * color_coherence_score + 0.12 * texture_score +
                0.12 * edge_score + 0.10 * histogram_score + 0.09 * jpeg_score
            )
            high = sum(1 for s in [frequency_score, noise_score, color_coherence_score,
                                    texture_score, edge_score, histogram_score] if s > 0.45)
            if high >= 4:
                combined = min(1.0, combined * 1.25)
            elif high >= 3:
                combined = min(1.0, combined * 1.15)

            return {
                'frequency_score': round(frequency_score, 4),
                'noise_score': round(noise_score, 4),
                'color_coherence_score': round(color_coherence_score, 4),
                'texture_score': round(texture_score, 4),
                'edge_score': round(edge_score, 4),
                'histogram_score': round(histogram_score, 4),
                'jpeg_score': round(jpeg_score, 4),
                'combined_artifact_score': round(combined, 4),
            }
        except Exception as e:
            logger.error(f"Artifact analysis error: {e}")
            return self._empty_artifact_scores()

    @staticmethod
    def _empty_artifact_scores() -> Dict[str, float]:
        return {k: 0.0 for k in (
            'frequency_score', 'noise_score', 'color_coherence_score',
            'texture_score', 'edge_score', 'histogram_score',
            'jpeg_score', 'combined_artifact_score',
        )}

    # ── Public API ──────────────────────────────────────────────────────

    def analyze_image(
        self,
        image_path: str,
        model: str = 'EfficientNetAutoAttB4',
        dataset: str = 'DFDC',
        threshold: float = 0.5,
        analysis_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        start = time.time()

        # ① Down-scale the uploaded image immediately
        downscale_image_file(image_path, max_dim=512)

        # ② Artifact analysis (runs on every request, lightweight)
        artifact_scores = self._empty_artifact_scores()
        if self.enable_artifact:
            if analysis_id:
                self._emit_progress(analysis_id, 'analyzing', 20, 'Running artifact analysis …')
            artifact_scores = self._analyze_image_artifacts(image_path)

        # ③ ML model inference (if enabled)
        model_confidence = 0.0
        faces_detected = 0
        per_face = []
        method = 'artifact_only'

        if self.use_ml_model:
            try:
                if analysis_id:
                    self._emit_progress(analysis_id, 'loading', 30, 'Loading model …')
                self._ensure_model()
                self._ensure_face_detector()

                if analysis_id:
                    self._emit_progress(analysis_id, 'detecting', 45, 'Detecting faces …')

                im = Image.open(image_path).convert('RGB')
                face_extractor = _FaceExtractor(facedet=self._facedet)
                im_faces = face_extractor.process_image(img=im)
                faces_detected = len(im_faces.get('faces', []))

                if faces_detected > 0:
                    if analysis_id:
                        self._emit_progress(
                            analysis_id, 'analyzing', 60,
                            f'Inference on {faces_detected} face(s) …',
                        )
                    for face in im_faces['faces']:
                        face_np = np.array(face) if not isinstance(face, np.ndarray) else face
                        face_np = downscale_face_crop(face_np, target=self.FACE_SIZE)
                        face_pil = Image.fromarray(face_np) if isinstance(face_np, np.ndarray) else face_np
                        t = _torch.stack([self._transformer(image=face_pil)['image']])
                        with _torch.no_grad():
                            pred = _torch.sigmoid(self._model(t.to(self._device))).cpu().numpy().flatten()
                        per_face.append(float(pred[0]))

                    model_confidence = float(max(per_face))
                    method = 'single_model'
            except Exception as e:
                logger.error(f"ML inference error: {e}")
                method = 'artifact_only_fallback'
        else:
            method = 'artifact_only'

        # ④ Combine scores
        artifact_conf = artifact_scores['combined_artifact_score']
        if method.startswith('artifact_only'):
            combined = artifact_conf
        else:
            strong = sum(1 for s in [
                artifact_scores.get('frequency_score', 0),
                artifact_scores.get('noise_score', 0),
                artifact_scores.get('color_coherence_score', 0),
                artifact_scores.get('texture_score', 0),
                artifact_scores.get('edge_score', 0),
                artifact_scores.get('histogram_score', 0),
            ] if s > 0.40)

            if model_confidence > 0.7:
                combined = model_confidence * 0.70 + artifact_conf * 0.30
            elif model_confidence < 0.25:
                if strong >= 3:
                    combined = max(artifact_conf, 0.65)
                elif artifact_conf > 0.40:
                    combined = model_confidence * 0.10 + artifact_conf * 0.90
                else:
                    combined = model_confidence * 0.50 + artifact_conf * 0.50
            else:
                if strong >= 2:
                    combined = model_confidence * 0.20 + artifact_conf * 0.80
                else:
                    combined = model_confidence * 0.35 + artifact_conf * 0.65

        is_fake = combined > threshold

        if analysis_id:
            self._emit_progress(analysis_id, 'complete', 100, 'Analysis complete!')

        self._cleanup()

        return {
            'analysis_id': analysis_id,
            'status': 'completed',
            'type': 'image',
            'verdict': 'fake' if is_fake else 'real',
            'confidence': round(combined, 4),
            'threshold': threshold,
            'is_fake': is_fake,
            'faces_detected': faces_detected,
            'analysis_details': {
                'method': method,
                'model_confidence': round(model_confidence, 4),
                'per_face_scores': [round(p, 4) for p in per_face],
                'artifact_scores': artifact_scores,
            },
            'model': self.MODEL_NAME if method != 'artifact_only' else 'artifact_analysis',
            'dataset': self.DATASET if method != 'artifact_only' else 'N/A',
            'processing_time': round(time.time() - start, 3),
            'timestamp': datetime.now().isoformat(),
        }

    def analyze_video(
        self,
        video_path: str,
        model: str = 'EfficientNetAutoAttB4',
        dataset: str = 'DFDC',
        threshold: float = 0.5,
        frames: int = 30,
        analysis_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        start = time.time()

        if analysis_id:
            self._emit_progress(analysis_id, 'extracting', 10, 'Opening video …')

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        duration = total_frames / fps if fps > 0 else 0

        sample_count = min(frames, total_frames, 15)   # cap at 15 for memory
        indices = [int(i * total_frames / sample_count) for i in range(sample_count)]

        scores: List[float] = []
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            for idx, fidx in enumerate(indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
                ret, frame = cap.read()
                if not ret:
                    continue

                fpath = os.path.join(tmpdir, f'f_{idx}.jpg')
                # Down-scale frame before writing
                fh, fw = frame.shape[:2]
                if max(fh, fw) > 512:
                    sc = 512 / max(fh, fw)
                    frame = cv2.resize(frame, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
                cv2.imwrite(fpath, frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

                try:
                    res = self.analyze_image(
                        image_path=fpath, model=model, dataset=dataset,
                        threshold=threshold, analysis_id=None,
                    )
                    scores.append(res['confidence'])
                except Exception:
                    pass

                if analysis_id:
                    pct = 10 + int((idx / len(indices)) * 80)
                    self._emit_progress(
                        analysis_id, 'analyzing', pct,
                        f'Frame {idx + 1}/{len(indices)} …',
                    )

        cap.release()

        if not scores:
            return {
                'analysis_id': analysis_id, 'status': 'error',
                'error': 'Could not analyse any frames',
                'type': 'video',
                'processing_time': round(time.time() - start, 3),
            }

        avg = sum(scores) / len(scores)
        mx = max(scores)
        combined = 0.6 * avg + 0.4 * mx
        is_fake = combined > threshold

        if analysis_id:
            self._emit_progress(analysis_id, 'complete', 100, 'Analysis complete!')
        self._cleanup()

        return {
            'analysis_id': analysis_id, 'status': 'completed', 'type': 'video',
            'verdict': 'fake' if is_fake else 'real',
            'confidence': round(combined, 4),
            'threshold': threshold, 'is_fake': is_fake,
            'frames_analyzed': len(scores),
            'video_metadata': {
                'duration_seconds': round(duration, 2),
                'total_frames': total_frames,
                'fps': round(fps, 2),
            },
            'analysis_details': {
                'method': 'single_model_video',
                'avg_score': round(avg, 4),
                'max_score': round(mx, 4),
                'min_score': round(min(scores), 4),
                'frames_sampled': len(indices),
            },
            'model': self.MODEL_NAME,
            'dataset': self.DATASET,
            'processing_time': round(time.time() - start, 3),
            'timestamp': datetime.now().isoformat(),
        }

    # ── Batch / compare (thin wrappers) ─────────────────────────────────

    def analyze_batch(
        self, files: List[Dict], model: str = 'EfficientNetAutoAttB4',
        dataset: str = 'DFDC', threshold: float = 0.5,
        batch_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        results = []
        for i, f in enumerate(files):
            if batch_id:
                self._emit_progress(batch_id, 'batch', int(i / len(files) * 100), f'File {i + 1}/{len(files)} …')
            try:
                if f['type'] == 'video':
                    r = self.analyze_video(video_path=f['path'], model=model, dataset=dataset, threshold=threshold, analysis_id=f['id'])
                else:
                    r = self.analyze_image(image_path=f['path'], model=model, dataset=dataset, threshold=threshold, analysis_id=f['id'])
                r['original_filename'] = f['original_name']
                results.append(r)
            except Exception as e:
                results.append({'analysis_id': f['id'], 'original_filename': f['original_name'], 'status': 'error', 'error': str(e)})
            finally:
                if os.path.exists(f['path']):
                    os.remove(f['path'])
        if batch_id:
            self._emit_progress(batch_id, 'complete', 100, 'Batch complete!')
        return results

    def compare_models(self, file_path: str, file_type: str, models: List[str],
                       dataset: str = 'DFDC', threshold: float = 0.5) -> List[Dict[str, Any]]:
        """On free tier we only have one model – run it once."""
        try:
            if file_type == 'video':
                return [self.analyze_video(video_path=file_path, threshold=threshold)]
            return [self.analyze_image(image_path=file_path, threshold=threshold)]
        except Exception as e:
            return [{'model': self.MODEL_NAME, 'status': 'error', 'error': str(e)}]
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
