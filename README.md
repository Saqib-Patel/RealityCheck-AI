<div align="center">

# 🎭 DeepFake Detection Hub

### AI-Powered Deepfake & AI-Generated Image Detection

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Visit_App-00f0ff?style=for-the-badge)](https://deepfake-detection-hub.vercel.app)
[![Portfolio](https://img.shields.io/badge/👤_Portfolio-Mohammed_Saqib_Patel-6366f1?style=for-the-badge)](https://www.linkedin.com/in/mohammedsaqibpatel/)

![Next.js](https://img.shields.io/badge/Next.js_15-000000?style=flat-square&logo=next.js&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?style=flat-square&logo=typescript&logoColor=white)
![Python](https://img.shields.io/badge/Python_3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch_2.0-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)
![WebSocket](https://img.shields.io/badge/WebSocket-010101?style=flat-square&logo=socketdotio&logoColor=white)

</div>

---

## 📖 Overview

A **production-grade, full-stack deepfake detection platform** that combines deep learning face-swap detection with a custom-built AI-generation forensics engine.

Users upload images or videos through a modern Next.js frontend, and the Flask backend processes them through a **multi-model ensemble pipeline** — running face detection (BlazeFace), face-swap analysis (EfficientNet CNNs), and a **7-channel forensic artifact analysis** engine — all with real-time WebSocket progress updates.

> **Why this project stands out:** Unlike typical deepfake detection demos that rely on a single model, this system uses an ensemble approach with 3 models cross-validated across 2 datasets, plus a novel signal-processing-based artifact detection engine that can identify AI-generated content (Stable Diffusion, DALL-E, Midjourney) — a class of fakes that standard face-swap detectors completely miss.

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        CLIENT (Next.js 15)                          │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌────────────────────┐   │
│  │  Upload  │→ │ WebSocket│→ │ Progress │→ │  Results Dashboard │   │
│  │  Zone    │  │ Client   │  │ Tracker  │  │  (Charts + Export) │   │
│  └─────────┘  └──────────┘  └──────────┘  └────────────────────┘   │
└────────────────────────────┬─────────────────────────────────────────┘
                             │ REST API + WebSocket (Socket.IO)
┌────────────────────────────▼─────────────────────────────────────────┐
│                        SERVER (Flask + PyTorch)                      │
│                                                                      │
│  ┌─────────────────┐   ┌──────────────────────────────────────────┐  │
│  │   BlazeFace      │   │   Multi-Model Ensemble                  │  │
│  │   Face Detector   │→ │   ┌──────────────────────────────────┐  │  │
│  │   (GPU-accel.)    │   │   │ EfficientNetAutoAttB4 (DFDC)    │  │  │
│  └─────────────────┘   │   │ EfficientNetAutoAttB4 (FFPP)    │  │  │
│                          │   │ EfficientNetB4ST (cross-val)    │  │  │
│  ┌─────────────────┐   │   └──────────────┬───────────────────┘  │  │
│  │  7-Channel       │   └──────────────────┼──────────────────────┘  │
│  │  Artifact Engine  │                      │                        │
│  │  ┌─────────────┐ │    ┌─────────────────▼──────────────────────┐  │
│  │  │ Frequency   │ │    │  Intelligent Score Combiner            │  │
│  │  │ Noise       │ │───▶│  • Multi-signal agreement detection    │  │
│  │  │ Color       │ │    │  • Adaptive weighting (model vs.       │  │
│  │  │ Texture     │ │    │    artifact based on signal strength)   │  │
│  │  │ Edge        │ │    │  • Confidence boost for consensus      │  │
│  │  │ Histogram   │ │    └────────────────────────────────────────┘  │
│  │  │ JPEG        │ │                                                │
│  │  └─────────────┘ │                                                │
│  └─────────────────┘                                                 │
└──────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🧠 **Multi-Model Ensemble** | 3 models cross-validated across 2 datasets (DFDC + FFPP) using MAX prediction — catches manipulations any single model misses |
| 🔬 **7-Channel Forensic Engine** | Custom artifact analysis: DCT frequency spectrum, noise uniformity, cross-channel coherence, texture autocorrelation, edge coherence, color histogram smoothness, JPEG quantization |
| ⚡ **Real-Time WebSocket Updates** | Live progress tracking during analysis via Socket.IO — not polling, true push-based updates |
| 🎬 **Image + Video Support** | Frame-by-frame video analysis with batch inference (16 frames/batch) to prevent GPU OOM |
| 📊 **Detailed Analytics** | Per-face scores, per-model breakdown, individual artifact channel scores, and exportable reports |
| 🔒 **Privacy-First** | Zero data persistence — files processed in-memory and deleted immediately after analysis |
| 🐳 **Docker-Ready** | Full `docker-compose.yml` with production configs for one-command deployment |
| 🎨 **Premium UI** | Cyberpunk Aurora theme with Framer Motion animations, glassmorphism, and responsive design |

---

## 🔬 Detection Pipeline (Technical Deep-Dive)

### Stage 1: Face Detection
[BlazeFace](https://arxiv.org/abs/1907.05047) (Google) extracts all faces from the input, running on GPU for real-time performance.

### Stage 2: Face-Swap Detection (Neural Network Ensemble)
Three pre-trained EfficientNet-B4 models run in parallel:

| Model | Dataset | What It Catches |
|-------|---------|----------------|
| EfficientNetAutoAttB4 | DFDC | Face swaps from the Facebook DeepFake Detection Challenge |
| EfficientNetAutoAttB4 | FFPP | Face manipulations from FaceForensics++ (Face2Face, FaceSwap, NeuralTextures) |
| EfficientNetB4ST | DFDC | Different architecture (Siamese Tuning) for cross-validation |

Uses **MAX prediction** across all models — if *any* model flags *any* face, the image is flagged.

### Stage 3: AI-Generation Forensics (7-Channel Artifact Engine)
A custom signal-processing pipeline that catches AI-generated content (Stable Diffusion, DALL-E 3, Midjourney) that face-swap detectors miss:

| Channel | Signal | AI vs Real |
|---------|--------|-----------|
| **DCT Frequency** | High-freq energy ratio + very-high-freq deficit | AI: smooth spectrum, missing sensor noise |
| **Noise Uniformity** | Block-wise noise CV + brightness-noise correlation + kurtosis | AI: uniform noise, Gaussian distribution |
| **Cross-Channel** | R/G/B noise pairwise correlation | AI: ~0.7-0.95 correlated vs real ~0.3-0.5 |
| **Texture** | 2D autocorrelation for micro-pattern repetition | AI: subtle repeating patterns |
| **Edge** | Canny threshold ratio + gradient CV at edges | AI: too-consistent edge profiles |
| **Histogram** | 2nd-derivative roughness of color histograms | AI: unnaturally smooth histograms |
| **JPEG** | 8×8 block boundary discontinuities (H+V) | Real photos: JPEG artifacts present |

### Stage 4: Intelligent Score Combination
Adaptive fusion that counts **how many independent channels agree**:
- **3+ strong signals** → artifact analysis overrides model (combined ≥ 0.65)
- **Model confident (>0.7)** → trust the model (70% weight)
- **Model uncertain** → artifact analysis gets 65-80% weight

---

## 🚀 Quick Start

### Prerequisites
- **Node.js** 18+ • **Python** 3.10+ • **CUDA** (optional, for GPU acceleration)

### Local Development
```bash
# Clone
git clone https://github.com/Saqib-Patel/DeepFake-Detection-Hub.git
cd DeepFake-Detection-Hub

# Backend
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -r backend/requirements.txt
python backend/run.py

# Frontend (new terminal)
cd frontend-nextjs
npm install
npm run dev
```

**Frontend** → http://localhost:3000 &nbsp;|&nbsp; **API** → http://localhost:5000 &nbsp;|&nbsp; **Health** → http://localhost:5000/health

### Docker
```bash
cp .env.example .env
docker-compose up --build
```

---

## 📡 REST API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/models` | List available models and datasets |
| `POST` | `/api/v1/analyze/image` | Analyze single image (multipart form) |
| `POST` | `/api/v1/analyze/video` | Analyze video with frame extraction |
| `POST` | `/api/v1/analyze/batch` | Batch analysis (up to 20 files) |
| `POST` | `/api/v1/compare` | Compare all models on same file |
| `GET` | `/api/v1/results` | List cached results |
| `GET` | `/api/v1/results/:id` | Get specific result |
| `DELETE` | `/api/v1/results/:id` | Delete result |
| `GET` | `/health` | Health check |
| `GET` | `/health/ready` | Readiness probe + GPU info |

<details>
<summary><strong>Example Request & Response</strong></summary>

```bash
curl -X POST http://localhost:5000/api/v1/analyze/image \
  -F "file=@photo.jpg" \
  -F "model=EfficientNetAutoAttB4" \
  -F "dataset=DFDC"
```

```json
{
  "analysis_id": "a1b2c3d4",
  "status": "completed",
  "verdict": "fake",
  "confidence": 0.8721,
  "is_fake": true,
  "faces_detected": 1,
  "analysis_details": {
    "method": "ensemble_combined",
    "model_confidence": 0.1234,
    "ensemble_models_used": [
      "EfficientNetAutoAttB4_DFDC",
      "EfficientNetAutoAttB4_FFPP",
      "EfficientNetB4ST_DFDC"
    ],
    "artifact_scores": {
      "frequency_score": 0.7123,
      "noise_score": 0.6541,
      "color_coherence_score": 0.4892,
      "texture_score": 0.3201,
      "edge_score": 0.5100,
      "histogram_score": 0.6200,
      "jpeg_score": 0.4500,
      "combined_artifact_score": 0.5543
    },
    "strong_ai_signals": 4,
    "moderate_ai_signals": 6
  },
  "model": "EfficientNetAutoAttB4",
  "dataset": "DFDC",
  "processing_time": 2.341
}
```

</details>

---

## 📁 Project Structure

```
DeepFake-Detection-Hub/
├── frontend-nextjs/            # Next.js 15 App Router + TypeScript
│   ├── app/                    # Pages: /, /analyze, /history, /how-it-works
│   ├── components/             # Reusable UI + feature components
│   │   ├── ui/                 # Design system (Button, Card, Badge, etc.)
│   │   ├── features/           # Analysis, Upload, Results, Model Selector
│   │   └── layout/             # Header, Footer with responsive nav
│   ├── hooks/                  # Custom hooks (useWebSocket, useLocalStorage)
│   ├── lib/                    # API client, WebSocket, utils, constants
│   └── types/                  # TypeScript type definitions
│
├── backend/                    # Flask REST API + WebSocket server
│   └── app/
│       ├── routes/             # API endpoints (api.py, health.py)
│       ├── services/           # Core logic (detector.py, result_manager.py)
│       ├── utils/              # Input validation
│       └── websocket/          # Socket.IO event handlers
│
├── architectures/              # PyTorch model definitions + pretrained weight URLs
├── blazeface/                  # BlazeFace face detection (anchors + weights)
├── isplutils/                  # Image processing utilities
├── docker-compose.yml          # Multi-service Docker orchestration
├── Dockerfile.backend          # Backend container config
└── render.yaml                 # Render.com deployment blueprint
```

---

## 🛠️ Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Frontend** | Next.js 15, React 18, TypeScript, Tailwind CSS, Framer Motion, Socket.IO Client, Radix UI, Lucide Icons |
| **Backend** | Flask, Flask-SocketIO, Gunicorn, Eventlet, Python 3.10+ |
| **ML/AI** | PyTorch 2.0, EfficientNet (with Auto-Attention), BlazeFace, DCT/FFT Spectral Analysis |
| **Image Processing** | OpenCV, Pillow, NumPy, SciPy (DCT, uniform_filter, correlation) |
| **DevOps** | Docker, Docker Compose, Render, Vercel |

---

## 🔬 Research Attribution

Built upon research from **ISPL — Politecnico di Milano**:

> *Video Face Manipulation Detection Through Ensemble of CNNs*
> Bonettini, Cannas, Mandelli, Bondi, Bestagini (ICPR 2020)

The artifact analysis engine is an original contribution extending the face-swap detection approach to cover AI-generated content.

---

## 👤 Author

**Mohammed Saqib Patel**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/mohammedsaqibpatel/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat-square&logo=github)](https://github.com/Saqib-Patel)
[![Twitter](https://img.shields.io/badge/Twitter-Follow-1DA1F2?style=flat-square&logo=twitter)](https://x.com/patel_saqib26)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
