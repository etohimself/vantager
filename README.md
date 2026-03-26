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

```bash
git clone git@github.com:etohimself/vantager.git
cd vantager

python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python server.py
```

Open **http://localhost:8080** and log in with `admin` / `Admin123!`.

> Set `ADMIN_PASSWORD` env var in production.

---

## Deploy on a GPU Instance

This works on any GPU instance (Vast.ai, RunPod, Lambda, bare-metal, etc.)

### 1. First-time setup on the instance

```bash
# Clone the repo
git clone git@github.com:etohimself/vantager.git
cd vantager

# Create virtualenv and install deps
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# (Optional) Set environment variables
export ADMIN_PASSWORD=YourSecurePassword
export DATA_DIR=/workspace/data        # or wherever you want persistent data

# Start
chmod +x start.sh
./start.sh
```

### 2. Updating (after git push)

```bash
cd vantager
git pull
source .venv/bin/activate
pip install -r requirements.txt   # only needed if deps changed
./start.sh
```

### 3. Running with Cloudflare Tunnel

```bash
export CLOUDFLARE_TUNNEL_TOKEN=your-token
./start.sh
```

The tunnel exposes the app on your Cloudflare domain without opening ports.

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

> **Tip:** On cloud instances, point `DATA_DIR` to a persistent volume (e.g., `/workspace/data` on RunPod/Vast.ai) so trained models and user data survive instance restarts.

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
