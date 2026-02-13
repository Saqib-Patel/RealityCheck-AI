import os
import sys
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.model_zoo import load_url
from scipy.special import expit
from scipy.fft import dctn
from scipy.ndimage import uniform_filter

BASE_DIR = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from blazeface import FaceExtractor, BlazeFace, VideoReader
from architectures import fornet, weights
from isplutils import utils

from app import socketio


class DeepFakeDetector:

    def __init__(self):
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.face_policy = 'scale'
        self.face_size = 224
        self._model_cache: Dict[str, torch.nn.Module] = {}
        self._facedet = None
        self._face_extractor = None
        self._init_face_detector()

    def _init_face_detector(self):
        blazeface_dir = BASE_DIR / 'blazeface'
        self._facedet = BlazeFace().to(self.device)
        self._facedet.load_weights(str(blazeface_dir / "blazeface.pth"))
        self._facedet.load_anchors(str(blazeface_dir / "anchors.npy"))
        self._video_reader = VideoReader(verbose=False)

    def _get_model(self, model_name: str, dataset: str) -> torch.nn.Module:
        # cache_key = "EfficientNetAutoAttB4_DFDC" etc — picks the right pretrained weights
        cache_key = f"{model_name}_{dataset}"
        if cache_key not in self._model_cache:
            model_url = weights.weight_url[cache_key]
            net = getattr(fornet, model_name)().eval().to(self.device)
            net.load_state_dict(load_url(model_url, map_location=self.device, check_hash=True))
            self._model_cache[cache_key] = net
        return self._model_cache[cache_key]

    def _get_transformer(self, model: torch.nn.Module):
        return utils.get_transformer(self.face_policy, self.face_size, model.get_normalizer(), train=False)

    def _emit_progress(self, analysis_id: str, stage: str, progress: int, message: str):
        socketio.emit('analysis_progress', {
            'analysis_id': analysis_id,
            'stage': stage,
            'progress': progress,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }, namespace='/ws')

    def _analyze_image_artifacts(self, image_path: str) -> Dict[str, float]:
        """
        Comprehensive artifact analysis for AI-generated image detection.
        Uses multiple forensic signals to catch AI-generated content that
        face-swap detection models miss (Stable Diffusion, Midjourney, DALL-E, etc.).

        Detection channels:
          1. Frequency spectrum analysis (DCT) — AI lacks natural sensor noise patterns
          2. Noise uniformity analysis — AI noise is too uniform across blocks
          3. Cross-channel noise coherence — AI channels are unnaturally correlated
          4. Texture repetition analysis — AI shows micro-pattern repetition
          5. Edge coherence analysis — AI edges lack natural sub-pixel variation
          6. Color histogram smoothness — AI histograms are too smooth/perfect
          7. JPEG quantization artifacts — real photos usually have JPEG artifacts

        Returns individual scores (0-1, higher = more likely AI-generated).
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return self._empty_artifact_scores()

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
            h, w = gray.shape

            # --- 1. Frequency spectrum analysis (significantly tightened) ---
            # AI-generated images lack natural high-frequency sensor noise
            # and show unnaturally smooth spectral energy distribution.
            # Modern diffusion models (SD, MJ) have even smoother spectra than GANs.
            dct = dctn(gray)
            dct_log = np.log1p(np.abs(dct))

            yy, xx = np.mgrid[0:h, 0:w]
            dist = np.sqrt((yy / h) ** 2 + (xx / w) ** 2)

            low_mask = dist < 0.15       # tighter low-freq band
            mid_mask = (dist >= 0.15) & (dist < 0.50)
            high_mask = dist >= 0.50
            very_high_mask = dist >= 0.75  # new: very-high-freq band

            low_energy = np.mean(dct_log[low_mask]) if np.any(low_mask) else 1.0
            mid_energy = np.mean(dct_log[mid_mask]) if np.any(mid_mask) else 0.0
            high_energy = np.mean(dct_log[high_mask]) if np.any(high_mask) else 0.0
            very_high_energy = np.mean(dct_log[very_high_mask]) if np.any(very_high_mask) else 0.0

            # Ratio analysis — tighter threshold (was 0.30, now 0.25)
            freq_ratio = high_energy / (low_energy + 1e-10)
            freq_ratio_score = float(np.clip(1.0 - (freq_ratio / 0.25), 0.0, 1.0))

            # Very-high-freq deficit — diffusion models almost completely lack this band
            vh_ratio = very_high_energy / (low_energy + 1e-10)
            vh_deficit_score = float(np.clip(1.0 - (vh_ratio / 0.15), 0.0, 1.0))

            # Spectral dropoff steepness
            mid_high_ratio = mid_energy / (high_energy + 1e-10)
            dropoff_score = float(np.clip((mid_high_ratio - 1.0) / 1.2, 0.0, 1.0))

            # Spectral variance — AI images have unnaturally smooth DCT spectra
            high_freq_values = dct_log[high_mask]
            if len(high_freq_values) > 10:
                spectral_cv = float(np.std(high_freq_values) / (np.mean(high_freq_values) + 1e-10))
                spectral_smoothness_score = float(np.clip(1.0 - (spectral_cv / 0.6), 0.0, 1.0))
            else:
                spectral_smoothness_score = 0.0

            # Spectral energy concentration — AI images concentrate energy in low freqs
            total_energy = np.sum(dct_log)
            low_energy_pct = np.sum(dct_log[low_mask]) / (total_energy + 1e-10)
            energy_concentration_score = float(np.clip((low_energy_pct - 0.3) / 0.3, 0.0, 1.0))

            frequency_score = (
                0.25 * freq_ratio_score +
                0.20 * vh_deficit_score +
                0.20 * dropoff_score +
                0.20 * spectral_smoothness_score +
                0.15 * energy_concentration_score
            )

            # --- 2. Noise uniformity analysis (tightened thresholds) ---
            smoothed = uniform_filter(gray, size=3)
            noise = gray - smoothed

            # Use smaller blocks for finer-grained analysis
            block_size = max(12, min(h, w) // 12)
            block_stds = []
            block_means = []
            for i in range(0, h - block_size, block_size):
                for j in range(0, w - block_size, block_size):
                    block = noise[i:i + block_size, j:j + block_size]
                    block_stds.append(np.std(block))
                    block_means.append(np.mean(gray[i:i + block_size, j:j + block_size]))

            if len(block_stds) > 4:
                cv_noise = np.std(block_stds) / (np.mean(block_stds) + 1e-10)

                if np.std(block_means) > 1e-10 and np.std(block_stds) > 1e-10:
                    correlation = abs(np.corrcoef(block_means, block_stds)[0, 1])
                    if np.isnan(correlation):
                        correlation = 0.0
                else:
                    correlation = 0.0

                # Tighter thresholds (was 0.5/0.3, now 0.40/0.25)
                uniformity_score = float(np.clip(1.0 - (cv_noise / 0.40), 0.0, 1.0))
                decorrelation_score = float(np.clip(1.0 - (correlation / 0.25), 0.0, 1.0))

                # Noise level — AI images have less total noise (tighter: was 3.0, now 2.5)
                mean_noise_level = np.mean(block_stds)
                low_noise_score = float(np.clip(1.0 - (mean_noise_level / 2.5), 0.0, 1.0))

                # Noise kurtosis — AI noise has more Gaussian distribution (kurtosis ~ 3)
                # Real sensor noise has heavier tails (kurtosis > 4)
                all_noise = noise.flatten()
                noise_kurtosis = float(np.mean(all_noise ** 4) / (np.mean(all_noise ** 2) ** 2 + 1e-10))
                kurtosis_score = float(np.clip(1.0 - ((noise_kurtosis - 3.0) / 3.0), 0.0, 1.0))

                noise_score = (
                    0.30 * uniformity_score +
                    0.25 * decorrelation_score +
                    0.25 * low_noise_score +
                    0.20 * kurtosis_score
                )
            else:
                noise_score = 0.0

            # --- 3. Cross-channel noise coherence (tightened) ---
            color_coherence_score = 0.0
            try:
                b, g, r = cv2.split(img)
                channels = [c.astype(np.float64) for c in [r, g, b]]
                channel_noises = []
                for ch in channels:
                    ch_smooth = uniform_filter(ch, size=3)
                    channel_noises.append((ch - ch_smooth).flatten())

                cross_corrs = []
                for ci in range(3):
                    for cj in range(ci + 1, 3):
                        corr = abs(np.corrcoef(channel_noises[ci], channel_noises[cj])[0, 1])
                        if not np.isnan(corr):
                            cross_corrs.append(corr)

                if cross_corrs:
                    mean_cross_corr = np.mean(cross_corrs)
                    # Tighter threshold (was 0.4/0.4, now 0.35/0.35)
                    # Real images: ~0.25-0.45, AI images: ~0.6-0.95
                    color_coherence_score = float(np.clip((mean_cross_corr - 0.35) / 0.35, 0.0, 1.0))
            except Exception:
                color_coherence_score = 0.0

            # --- 4. Texture repetition analysis (NEW) ---
            # AI generators often produce subtle micro-pattern repetitions
            # that don't exist in real photographs
            texture_score = 0.0
            try:
                # Analyze autocorrelation in small patches
                patch_size = min(64, min(h, w) // 4)
                if patch_size >= 32:
                    # Take center crop for efficiency
                    cy, cx = h // 2, w // 2
                    patch = gray[cy - patch_size:cy + patch_size, cx - patch_size:cx + patch_size]

                    # 2D autocorrelation via FFT
                    f_patch = np.fft.fft2(patch - np.mean(patch))
                    power_spectrum = np.abs(f_patch) ** 2
                    autocorr = np.real(np.fft.ifft2(power_spectrum))
                    autocorr = autocorr / (autocorr[0, 0] + 1e-10)  # normalize

                    # Check for unexpected peaks (exclude DC and immediate neighbors)
                    ac_h, ac_w = autocorr.shape
                    # Zero out center region (DC + neighbors)
                    autocorr_masked = autocorr.copy()
                    center_r = max(3, ac_h // 16)
                    autocorr_masked[:center_r, :center_r] = 0
                    autocorr_masked[:center_r, -center_r:] = 0
                    autocorr_masked[-center_r:, :center_r] = 0
                    autocorr_masked[-center_r:, -center_r:] = 0

                    # High autocorrelation peaks outside center = repetition
                    max_ac = np.max(np.abs(autocorr_masked))
                    # Real images: max_ac < 0.15, AI with repetition: max_ac > 0.2
                    texture_score = float(np.clip((max_ac - 0.10) / 0.20, 0.0, 1.0))
            except Exception:
                texture_score = 0.0

            # --- 5. Edge coherence analysis (NEW) ---
            # Real photos have natural edge variation from optics + sensor
            # AI images have edges that are too consistent/smooth
            edge_score = 0.0
            try:
                # Compute edges with Canny at two different thresholds
                edges_tight = cv2.Canny(img, 100, 200)
                edges_loose = cv2.Canny(img, 50, 100)

                # Ratio of tight to loose edges
                # Real images: more difference between thresholds (ratio 0.3-0.5)
                # AI images: edges are more binary/consistent (ratio 0.5-0.8)
                tight_count = np.sum(edges_tight > 0)
                loose_count = np.sum(edges_loose > 0)
                if loose_count > 0:
                    edge_ratio = tight_count / (loose_count + 1e-10)
                    edge_consistency_score = float(np.clip((edge_ratio - 0.35) / 0.35, 0.0, 1.0))
                else:
                    edge_consistency_score = 0.0

                # Analyze edge gradient magnitudes — AI edges have less variation
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                gradient_mag = np.sqrt(sobelx ** 2 + sobely ** 2)

                # Look at gradient magnitude distribution at edge locations
                edge_mask = edges_loose > 0
                if np.sum(edge_mask) > 100:
                    edge_gradients = gradient_mag[edge_mask]
                    grad_cv = np.std(edge_gradients) / (np.mean(edge_gradients) + 1e-10)
                    # Real images: grad_cv > 0.8, AI: grad_cv < 0.6
                    edge_uniformity_score = float(np.clip(1.0 - (grad_cv / 0.7), 0.0, 1.0))
                else:
                    edge_uniformity_score = 0.0

                edge_score = 0.50 * edge_consistency_score + 0.50 * edge_uniformity_score
            except Exception:
                edge_score = 0.0

            # --- 6. Color histogram smoothness (NEW) ---
            # Real camera images have slightly irregular histograms due to sensor noise
            # AI-generated images tend to have very smooth, well-distributed histograms
            histogram_score = 0.0
            try:
                hist_smoothness_scores = []
                for ch_idx in range(3):
                    hist = cv2.calcHist([img], [ch_idx], None, [256], [0, 256]).flatten()
                    hist = hist / (np.sum(hist) + 1e-10)

                    # Compute histogram gradient (first derivative)
                    hist_grad = np.diff(hist)
                    hist_grad_smoothness = np.std(hist_grad) / (np.mean(np.abs(hist_grad)) + 1e-10)

                    # Second derivative — measures "roughness"
                    hist_grad2 = np.diff(hist_grad)
                    hist_roughness = np.mean(np.abs(hist_grad2))

                    # Real images: rougher histograms (higher roughness)
                    # AI images: smoother histograms (lower roughness)
                    ch_smoothness = float(np.clip(1.0 - (hist_roughness / 0.002), 0.0, 1.0))
                    hist_smoothness_scores.append(ch_smoothness)

                histogram_score = float(np.mean(hist_smoothness_scores))
            except Exception:
                histogram_score = 0.0

            # --- 7. JPEG quantization artifacts (enhanced) ---
            jpeg_score = 0.0
            try:
                if h >= 16 and w >= 16:
                    # Check both horizontal AND vertical block boundaries
                    h_diffs_at_boundary = []
                    h_diffs_elsewhere = []
                    v_diffs_at_boundary = []
                    v_diffs_elsewhere = []

                    for col in range(1, w):
                        diff = np.mean(np.abs(gray[:, col] - gray[:, col - 1]))
                        if col % 8 == 0:
                            h_diffs_at_boundary.append(diff)
                        else:
                            h_diffs_elsewhere.append(diff)

                    for row in range(1, h):
                        diff = np.mean(np.abs(gray[row, :] - gray[row - 1, :]))
                        if row % 8 == 0:
                            v_diffs_at_boundary.append(diff)
                        else:
                            v_diffs_elsewhere.append(diff)

                    has_jpeg_h = False
                    has_jpeg_v = False

                    if h_diffs_at_boundary and h_diffs_elsewhere:
                        h_boundary_ratio = np.mean(h_diffs_at_boundary) / (np.mean(h_diffs_elsewhere) + 1e-10)
                        has_jpeg_h = h_boundary_ratio > 1.02

                    if v_diffs_at_boundary and v_diffs_elsewhere:
                        v_boundary_ratio = np.mean(v_diffs_at_boundary) / (np.mean(v_diffs_elsewhere) + 1e-10)
                        has_jpeg_v = v_boundary_ratio > 1.02

                    if not has_jpeg_h and not has_jpeg_v:
                        # No JPEG artifacts at all — suspicious for a "real photo"
                        jpeg_score = 0.45
                    elif not has_jpeg_h or not has_jpeg_v:
                        jpeg_score = 0.25

                # File extension check
                ext = os.path.splitext(image_path)[1].lower()
                if ext in ('.png', '.webp', '.bmp', '.tiff'):
                    jpeg_score = max(jpeg_score, 0.35)
            except Exception:
                jpeg_score = 0.0

            # --- Combined artifact score (re-weighted for 7 channels) ---
            combined_artifact_score = (
                0.22 * frequency_score +
                0.20 * noise_score +
                0.15 * color_coherence_score +
                0.12 * texture_score +
                0.12 * edge_score +
                0.10 * histogram_score +
                0.09 * jpeg_score
            )

            # Apply confidence boost: if multiple independent signals agree,
            # this is stronger evidence than any single signal
            high_scores = sum(1 for s in [frequency_score, noise_score, color_coherence_score,
                                           texture_score, edge_score, histogram_score]
                              if s > 0.45)
            if high_scores >= 4:
                # 4+ independent signals all pointing to AI → boost confidence
                combined_artifact_score = min(1.0, combined_artifact_score * 1.25)
            elif high_scores >= 3:
                combined_artifact_score = min(1.0, combined_artifact_score * 1.15)

            return {
                'frequency_score': round(frequency_score, 4),
                'noise_score': round(noise_score, 4),
                'color_coherence_score': round(color_coherence_score, 4),
                'texture_score': round(texture_score, 4),
                'edge_score': round(edge_score, 4),
                'histogram_score': round(histogram_score, 4),
                'jpeg_score': round(jpeg_score, 4),
                'combined_artifact_score': round(combined_artifact_score, 4)
            }

        except Exception as e:
            logger.error(f"Artifact analysis error: {e}")
            return self._empty_artifact_scores()

    def _empty_artifact_scores(self) -> Dict[str, float]:
        """Return zeroed artifact scores for error/fallback cases."""
        return {
            'frequency_score': 0.0, 'noise_score': 0.0,
            'color_coherence_score': 0.0, 'texture_score': 0.0,
            'edge_score': 0.0, 'histogram_score': 0.0,
            'jpeg_score': 0.0, 'combined_artifact_score': 0.0
        }

    def analyze_image(self, image_path: str, model: str = 'EfficientNetAutoAttB4',
                      dataset: str = 'DFDC', threshold: float = 0.5,
                      analysis_id: Optional[str] = None) -> Dict[str, Any]:
        start_time = time.time()

        if analysis_id:
            self._emit_progress(analysis_id, 'loading', 10, 'Loading model...')

        net = self._get_model(model, dataset)
        transf = self._get_transformer(net)

        if analysis_id:
            self._emit_progress(analysis_id, 'detecting', 30, 'Detecting faces...')

        im = Image.open(image_path).convert('RGB')
        face_extractor = FaceExtractor(facedet=self._facedet)
        im_faces = face_extractor.process_image(img=im)

        # --- Run supplementary artifact analysis ---
        if analysis_id:
            self._emit_progress(analysis_id, 'analyzing', 45, 'Analyzing image artifacts...')

        artifact_scores = self._analyze_image_artifacts(image_path)

        faces_detected = len(im_faces.get('faces', []))

        if faces_detected == 0:
            # No faces found — rely entirely on artifact analysis
            artifact_conf = artifact_scores['combined_artifact_score']
            is_fake = artifact_conf > threshold

            if analysis_id:
                self._emit_progress(analysis_id, 'complete', 100, 'Analysis complete!')

            return {
                'analysis_id': analysis_id, 'status': 'completed', 'type': 'image',
                'verdict': 'fake' if is_fake else 'real',
                'confidence': round(artifact_conf, 4),
                'threshold': threshold, 'is_fake': is_fake,
                'faces_detected': 0,
                'analysis_details': {
                    'method': 'artifact_only',
                    'note': 'No faces detected — verdict based on image artifact analysis',
                    'artifact_scores': artifact_scores
                },
                'model': model, 'dataset': dataset,
                'processing_time': round(time.time() - start_time, 3),
                'timestamp': datetime.now().isoformat()
            }

        if analysis_id:
            self._emit_progress(analysis_id, 'analyzing', 60, f'Running inference on {faces_detected} face(s)...')

        # --- Analyze ALL detected faces with MULTI-MODEL ENSEMBLE ---
        # Using multiple models dramatically improves detection on face-swap deepfakes.
        # Different model/dataset combos catch different manipulation types.
        all_face_predictions = []
        for face in im_faces['faces']:
            faces_t = torch.stack([transf(image=face)['image']])
            with torch.no_grad():
                pred = torch.sigmoid(net(faces_t.to(self.device))).cpu().numpy().flatten()
            all_face_predictions.append(float(pred[0]))

        # Cross-validate with additional models for better accuracy
        ensemble_predictions = list(all_face_predictions)  # start with primary model
        ensemble_models_used = [f"{model}_{dataset}"]

        # Try alternative dataset (if primary is DFDC, also try FFPP and vice versa)
        alt_dataset = 'FFPP' if dataset == 'DFDC' else 'DFDC'
        try:
            alt_net = self._get_model(model, alt_dataset)
            alt_transf = self._get_transformer(alt_net)
            for face in im_faces['faces']:
                faces_t = torch.stack([alt_transf(image=face)['image']])
                with torch.no_grad():
                    pred = torch.sigmoid(alt_net(faces_t.to(self.device))).cpu().numpy().flatten()
                ensemble_predictions.append(float(pred[0]))
            ensemble_models_used.append(f"{model}_{alt_dataset}")
        except Exception:
            pass  # alternative model not available, skip

        # Try a different architecture (EfficientNetB4ST) for cross-validation
        cross_model = 'EfficientNetB4ST'
        if cross_model != model:
            try:
                cross_net = self._get_model(cross_model, dataset)
                cross_transf = self._get_transformer(cross_net)
                for face in im_faces['faces']:
                    faces_t = torch.stack([cross_transf(image=face)['image']])
                    with torch.no_grad():
                        pred = torch.sigmoid(cross_net(faces_t.to(self.device))).cpu().numpy().flatten()
                    ensemble_predictions.append(float(pred[0]))
                ensemble_models_used.append(f"{cross_model}_{dataset}")
            except Exception:
                pass

        # Use the MAX fake probability across all faces AND all models
        # (if ANY model on ANY face detects fakeness, flag the image)
        model_confidence = float(max(ensemble_predictions))
        avg_model_confidence = float(np.mean(all_face_predictions))  # avg of primary model only
        artifact_confidence = artifact_scores['combined_artifact_score']

        # --- Intelligent score combination ---
        # The face-swap model detects face-swap deepfakes but NOT AI-generated images.
        # Artifact analysis catches AI-generated content (SD, MJ, DALL-E, etc.).
        # CRITICAL: We must allow artifact scores to FULLY OVERRIDE the model
        # when multiple independent artifact channels agree on AI-generation.

        # Count how many independent artifact channels are showing AI signals
        artifact_channel_scores = [
            artifact_scores.get('frequency_score', 0),
            artifact_scores.get('noise_score', 0),
            artifact_scores.get('color_coherence_score', 0),
            artifact_scores.get('texture_score', 0),
            artifact_scores.get('edge_score', 0),
            artifact_scores.get('histogram_score', 0),
        ]
        strong_ai_signals = sum(1 for s in artifact_channel_scores if s > 0.40)
        moderate_ai_signals = sum(1 for s in artifact_channel_scores if s > 0.25)

        if model_confidence > 0.7:
            # Model is confident it's a face-swap fake — trust the model
            combined_confidence = model_confidence * 0.70 + artifact_confidence * 0.30
        elif model_confidence < 0.25:
            # Model thinks faces are real (not swapped).
            # But this does NOT mean the image is real — it could be AI-generated.
            if strong_ai_signals >= 3:
                # 3+ independent channels see AI-generation → almost certainly AI
                # Let artifact analysis completely override the face model
                combined_confidence = max(artifact_confidence, 0.65)
            elif strong_ai_signals >= 2 and artifact_confidence > 0.35:
                # 2+ strong signals with moderate combined score
                combined_confidence = model_confidence * 0.10 + artifact_confidence * 0.90
            elif artifact_confidence > 0.40:
                # Artifact analysis sees AI-generation signals — let it dominate
                combined_confidence = model_confidence * 0.10 + artifact_confidence * 0.90
            elif artifact_confidence > 0.25:
                # Moderate artifact signal — still give it strong weight
                combined_confidence = model_confidence * 0.20 + artifact_confidence * 0.80
            elif moderate_ai_signals >= 4:
                # Many moderate signals — suspicious even if individually weak
                combined_confidence = model_confidence * 0.25 + artifact_confidence * 0.75
            else:
                # Low artifact signal too — likely genuinely real
                combined_confidence = model_confidence * 0.50 + artifact_confidence * 0.50
        else:
            # Model is uncertain (0.25 - 0.7) — give artifact analysis more weight
            if strong_ai_signals >= 2:
                combined_confidence = model_confidence * 0.20 + artifact_confidence * 0.80
            else:
                combined_confidence = model_confidence * 0.35 + artifact_confidence * 0.65

        is_fake = combined_confidence > threshold

        if analysis_id:
            self._emit_progress(analysis_id, 'complete', 100, 'Analysis complete!')

        return {
            'analysis_id': analysis_id, 'status': 'completed', 'type': 'image',
            'verdict': 'fake' if is_fake else 'real',
            'confidence': round(combined_confidence, 4),
            'threshold': threshold, 'is_fake': is_fake,
            'faces_detected': faces_detected,
            'analysis_details': {
                'method': 'ensemble_combined',
                'model_confidence': round(model_confidence, 4),
                'model_confidence_avg': round(avg_model_confidence, 4),
                'per_face_scores': [round(p, 4) for p in all_face_predictions],
                'ensemble_predictions': [round(p, 4) for p in ensemble_predictions],
                'ensemble_models_used': ensemble_models_used,
                'artifact_scores': artifact_scores,
                'strong_ai_signals': strong_ai_signals,
                'moderate_ai_signals': moderate_ai_signals
            },
            'model': model, 'dataset': dataset,
            'processing_time': round(time.time() - start_time, 3),
            'timestamp': datetime.now().isoformat()
        }

    def analyze_video(self, video_path: str, model: str = 'EfficientNetAutoAttB4',
                      dataset: str = 'DFDC', threshold: float = 0.5,
                      frames: int = 50, analysis_id: Optional[str] = None) -> Dict[str, Any]:
        start_time = time.time()

        if analysis_id:
            self._emit_progress(analysis_id, 'loading', 5, 'Loading model...')

        net = self._get_model(model, dataset)
        transf = self._get_transformer(net)

        if analysis_id:
            self._emit_progress(analysis_id, 'extracting', 15, 'Extracting video frames...')

        video_read_fn = lambda x: self._video_reader.read_frames(x, num_frames=frames)
        face_extractor = FaceExtractor(video_read_fn=video_read_fn, facedet=self._facedet)
        vid_faces = face_extractor.process_video(video_path)

        if not vid_faces or len(vid_faces) == 0:
            return {
                'analysis_id': analysis_id, 'status': 'error',
                'error': 'No faces detected in the video',
                'type': 'video', 'processing_time': time.time() - start_time
            }

        if analysis_id:
            self._emit_progress(analysis_id, 'analyzing', 40, 'Analyzing frames...')

        valid_frames = [f for f in vid_faces if len(f.get('faces', []))]

        if len(valid_frames) == 0:
            return {
                'analysis_id': analysis_id, 'status': 'error',
                'error': 'No valid faces found in video frames',
                'type': 'video', 'processing_time': time.time() - start_time
            }

        faces_t = torch.stack([transf(image=f['faces'][0])['image'] for f in valid_frames])

        # batch inference — 16 frames at a time to avoid OOM
        frame_predictions = []
        batch_size = 16

        with torch.no_grad():
            for i in range(0, len(faces_t), batch_size):
                batch = faces_t[i:i+batch_size].to(self.device)
                batch_pred = net(batch).cpu().numpy().flatten()
                frame_predictions.extend(batch_pred.tolist())

                if analysis_id:
                    progress = 40 + int((i / len(faces_t)) * 50)
                    self._emit_progress(
                        analysis_id, 'analyzing', progress,
                        f'Analyzing frame {min(i+batch_size, len(faces_t))}/{len(faces_t)}...'
                    )

        predictions_array = np.array(frame_predictions)
        confidence = float(expit(predictions_array.mean()))
        is_fake = confidence > threshold

        frame_details = [
            {'frame_index': i, 'confidence': float(expit(p)), 'is_fake': float(expit(p)) > threshold}
            for i, p in enumerate(frame_predictions)
        ]

        if analysis_id:
            self._emit_progress(analysis_id, 'complete', 100, 'Analysis complete!')

        return {
            'analysis_id': analysis_id, 'status': 'completed', 'type': 'video',
            'verdict': 'fake' if is_fake else 'real',
            'confidence': confidence, 'threshold': threshold, 'is_fake': is_fake,
            'frames_requested': frames,
            'frames_analyzed': len(valid_frames),
            'frames_with_faces': len(valid_frames),
            'frame_details': frame_details,
            'confidence_stats': {
                'mean': confidence,
                'min': float(min(expit(p) for p in frame_predictions)),
                'max': float(max(expit(p) for p in frame_predictions)),
                'std': float(np.std([expit(p) for p in frame_predictions]))
            },
            'model': model, 'dataset': dataset,
            'processing_time': round(time.time() - start_time, 3),
            'timestamp': datetime.now().isoformat()
        }

    def analyze_batch(self, files: List[Dict[str, Any]], model: str = 'EfficientNetAutoAttB4',
                      dataset: str = 'DFDC', threshold: float = 0.5,
                      batch_id: Optional[str] = None) -> List[Dict[str, Any]]:
        results = []
        total = len(files)

        for i, file_info in enumerate(files):
            if batch_id:
                self._emit_progress(batch_id, 'batch', int((i / total) * 100), f'Processing file {i+1}/{total}...')

            try:
                if file_info['type'] == 'video':
                    result = self.analyze_video(
                        video_path=file_info['path'], model=model, dataset=dataset,
                        threshold=threshold, analysis_id=file_info['id']
                    )
                else:
                    result = self.analyze_image(
                        image_path=file_info['path'], model=model, dataset=dataset,
                        threshold=threshold, analysis_id=file_info['id']
                    )
                result['original_filename'] = file_info['original_name']
                results.append(result)
            except Exception as e:
                results.append({
                    'analysis_id': file_info['id'],
                    'original_filename': file_info['original_name'],
                    'status': 'error', 'error': str(e)
                })
            finally:
                if os.path.exists(file_info['path']):
                    os.remove(file_info['path'])

        if batch_id:
            self._emit_progress(batch_id, 'complete', 100, 'Batch processing complete!')

        return results

    def compare_models(self, file_path: str, file_type: str, models: List[str],
                       dataset: str = 'DFDC', threshold: float = 0.5) -> List[Dict[str, Any]]:
        results = []
        for model_name in models:
            try:
                if file_type == 'video':
                    result = self.analyze_video(video_path=file_path, model=model_name, dataset=dataset, threshold=threshold)
                else:
                    result = self.analyze_image(image_path=file_path, model=model_name, dataset=dataset, threshold=threshold)
                results.append(result)
            except Exception as e:
                results.append({'model': model_name, 'status': 'error', 'error': str(e)})
        return results
