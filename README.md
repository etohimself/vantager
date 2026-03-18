# Tahmin Platformu

**Internal ML prediction platform for tabular, time series, NLP, and audio tasks.**
Built with AutoGluon, faster-whisper, sentence-transformers, and a bundled LLM — all served from a single Python process.

![Python 3.11](https://img.shields.io/badge/python-3.11-blue?logo=python&logoColor=white)
![CUDA 12.4](https://img.shields.io/badge/CUDA-12.4-76B900?logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/license-proprietary-gray)

---

## What It Does

| Capability | Engine | GPU? |
|---|---|---|
| **Tabular prediction** — classification & regression on CSV data | AutoGluon Tabular | Optional |
| **Time series forecasting** — multi-step ahead predictions | AutoGluon TimeSeries | Optional |
| **NLP / text embeddings** — text classification via embeddings | sentence-transformers | Optional |
| **Speech-to-text** — audio transcription & call analysis | faster-whisper (CTranslate2) | Recommended |
| **LLM assistant** — natural language model explanations | Bundled llama.cpp (Qwen 3.5 4B) | Recommended |

All features are accessible through a single web UI at `http://localhost:8080`.

---

## Quick Start

### Option A: Docker (recommended)

```bash
# Build
docker build -t tahmin-platformu .

# Run with GPU
docker run --gpus all -p 8080:8080 \
  -e ADMIN_PASSWORD=YourSecurePassword \
  -v tahmin-data:/app/data \
  tahmin-platformu
```

Open **http://localhost:8080** and log in with `admin` / `YourSecurePassword`.

### Option B: Run directly (development)

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python server.py
```

> First launch creates a default admin account (`admin` / `Admin123!`).
> Set `ADMIN_PASSWORD` env var in production.

---

## Deploy on RunPod

### 1. Push your Docker image

Build and push to a container registry (GitHub Container Registry shown here):

```bash
# One-time: log in to GHCR
echo $GHCR_TOKEN | docker login ghcr.io -u YOUR_GITHUB_USER --password-stdin

# Build & push
docker build -t ghcr.io/YOUR_GITHUB_USER/tahmin-platformu:latest .
docker push ghcr.io/YOUR_GITHUB_USER/tahmin-platformu:latest
```

### 2. Create a RunPod template

Go to [runpod.io/console/user/templates](https://runpod.io/console/user/templates) and create a new template:

| Field | Value |
|---|---|
| **Container Image** | `ghcr.io/YOUR_GITHUB_USER/tahmin-platformu:latest` |
| **Container Disk** | 20 GB (for OS + Python packages) |
| **Volume Disk** | 50 GB+ (for ML models, data) |
| **Volume Mount Path** | `/workspace` |
| **Expose HTTP Ports** | `8080` |
| **Environment Variables** | See table below |

Set these environment variables in the template:

| Variable | Value | Why |
|---|---|---|
| `ADMIN_PASSWORD` | (your password) | Secure the admin account |
| `DATA_DIR` | `/workspace/data` | Persist data on the network volume |

### 3. Launch a pod

Select your template, pick a GPU (RTX 3090/4090 or A100 recommended), and deploy.
Your app will be available at the RunPod proxy URL shown in the pod dashboard.

> **Tip:** Using a network volume (`/workspace`) means your trained models, user accounts, and LLM cache survive pod restarts and even pod deletion.

### CI/CD with GitHub Actions

A workflow is included at `.github/workflows/docker.yml`. On every push to `main`, it:

1. Builds the Docker image with BuildKit layer caching
2. Pushes to GitHub Container Registry (`ghcr.io`)
3. Tags with both `latest` and the commit SHA

**Setup:** No secrets needed — `GITHUB_TOKEN` is automatic. Just push to `main` and the image builds.

**Deploy:** After the image is pushed, restart your RunPod pod to pull the latest version. Or add a webhook to auto-restart (see RunPod API docs).

### RunPod with Persistent Storage (Budget-Friendly)

For maximum uptime on cheap spot/community GPUs:

1. **Create a Network Volume** on RunPod (50GB+) in your preferred region
2. **Set `DATA_DIR=/workspace/data`** in your pod template — all models, users, and caches survive pod restarts
3. **Use spot/community GPUs** — if the pod is reclaimed, your data is safe on the network volume
4. **Recreate the pod** pointing to the same network volume — everything resumes instantly
5. **Optional:** Use the RunPod API to script automatic pod recreation on termination

---

## Configuration

All settings are controlled via environment variables with sensible defaults.

### Core

| Variable | Default | Description |
|---|---|---|
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8080` | HTTP port |
| `DATA_DIR` | `./data` | Root directory for all persistent data |
| `ADMIN_USER` | `admin` | Initial admin username |
| `ADMIN_PASSWORD` | `Admin123!` | Initial admin password (**change this!**) |

### Limits

| Variable | Default | Description |
|---|---|---|
| `MAX_UPLOAD_SIZE_MB` | `200` | Max CSV upload size |
| `MAX_AUDIO_FILE_SIZE_MB` | `200` | Max audio file size |
| `MAX_BATCH_ROWS` | `100000` | Max rows in batch prediction |
| `MAX_PREDICTION_LENGTH` | `500` | Max time series forecast steps |
| `MAX_MODELS_PER_USER` | `50` | Per-user model quota |
| `MAX_EXPORT_SIZE_MB` | `2048` | Max model export zip size |

### Sessions

| Variable | Default | Description |
|---|---|---|
| `SESSION_TTL_SECONDS` | `28800` | Session lifetime (8 hours) |
| `SESSION_IDLE_TIMEOUT` | `7200` | Idle session expiry (2 hours) |

### LLM (llama.cpp)

The platform bundles a llama.cpp server that auto-downloads and manages itself.

| Variable | Default | Description |
|---|---|---|
| `LLAMA_BUNDLED` | `auto` | `auto` / `true` / `false` — manage llama-server |
| `LLAMA_MODEL_REPO` | `unsloth/Qwen3.5-4B-GGUF` | HuggingFace model repo |
| `LLAMA_MODEL_FILE` | `Qwen3.5-4B-Q4_K_M.gguf` | GGUF file to download |
| `LLAMA_GPU_LAYERS` | `99` | Layers to offload to GPU |
| `LLAMA_CTX_SIZE` | `8192` | Context window size |
| `LLAMA_PORT` | `8081` | Internal llama-server port |
| `LLAMA_CPP_URL` | `http://localhost:8081/v1/chat/completions` | LLM API endpoint |

### Speech-to-Text (Whisper)

| Variable | Default | Description |
|---|---|---|
| `WHISPER_MODEL_DIR` | `$DATA_DIR/stt` | Model cache directory |
| `WHISPER_DEVICE` | `auto` | `auto` / `cuda` / `cpu` |
| `WHISPER_COMPUTE_TYPE` | auto-detected | `int8_float16` (GPU) or `int8` (CPU) |
| `WHISPER_IDLE_TIMEOUT` | `300` | Unload model after N seconds idle |

---

## Data & Persistence

All runtime data lives under `DATA_DIR` (default: `./data`):

```
data/
├── models/          # Trained AutoGluon models
├── temp/            # Temporary upload & processing files
├── stt/             # Whisper model cache
├── llm/             # llama.cpp binary + GGUF model cache
├── users.json       # User accounts & roles
├── sessions.json    # Active sessions
└── activity.json    # Training & prediction activity log
```

**In Docker**, mount a named volume to `/app/data` to persist data across container restarts:

```bash
docker run --gpus all -p 8080:8080 -v tahmin-data:/app/data tahmin-platformu
```

**On RunPod**, set `DATA_DIR=/workspace/data` to use the network volume.

---

## User Management

The platform has a role-based multi-user system:

| Role | Can train | Can predict | Can manage users | Can approve models |
|---|---|---|---|---|
| `master_admin` | Yes | Yes | Yes | Yes |
| `admin` | Yes | Yes | Yes | Yes |
| `user` | Yes | Yes | No | No |
| `pending` | No | No | No | No |

- First launch auto-creates the `master_admin` account
- New users register and wait for admin approval
- Each user can run **1 training job** + **1 audio job** concurrently (fair queue)
- Tabular predictions are unlimited (lightweight, no queuing)

---

## Architecture

```
┌─────────────────────────────────────────────┐
│              Browser (index.html)            │
│         Tailwind CSS + Chart.js SPA         │
└──────────────────┬──────────────────────────┘
                   │ HTTP :8080
┌──────────────────▼──────────────────────────┐
│           server.py (single file)            │
│  ┌─────────────┐  ┌──────────────────────┐  │
│  │ Auth & RBAC  │  │  Fair Job Queue      │  │
│  └─────────────┘  └──────────────────────┘  │
│  ┌─────────────┐  ┌──────────────────────┐  │
│  │  AutoGluon   │  │  faster-whisper      │  │
│  │  (Tab + TS)  │  │  (STT)              │  │
│  └─────────────┘  └──────────────────────┘  │
│  ┌─────────────┐  ┌──────────────────────┐  │
│  │  sentence-   │  │  llama.cpp           │  │
│  │  transformers│  │  (bundled LLM)       │  │
│  └─────────────┘  └──────────────────────┘  │
│  ┌──────────────────────────────────────┐   │
│  │  Resource Manager (VRAM/RAM budget)   │   │
│  └──────────────────────────────────────┘   │
└──────────────────┬──────────────────────────┘
                   │
        ┌──────────▼──────────┐
        │     DATA_DIR         │
        │  models / users /    │
        │  stt / llm / temp    │
        └─────────────────────┘
```

Everything runs in a **single process** with threading. No Redis, no Celery, no database server — just Python and the filesystem.

---

## GPU Recommendations

| GPU | VRAM | Good For |
|---|---|---|
| RTX 3090 / 4090 | 24 GB | Full stack (training + Whisper + LLM) |
| A100 40 GB | 40 GB | Heavy training + large datasets |
| RTX 4080 | 16 GB | Training + Whisper (LLM on CPU) |
| T4 | 16 GB | Budget option, inference-focused |

> The platform auto-detects GPU availability and adjusts. Everything works on CPU too — just slower.
