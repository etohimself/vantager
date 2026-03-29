# Vantager

**No-code machine learning platform for tabular, time series, NLP, and call audio analysis.**

Upload a CSV or audio files, pick a target, and get a trained model with predictions, explainability, and export options — all from your browser. Built with AutoGluon, faster-whisper, sentence-transformers, and a bundled LLM, served from a single Python process.

![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue?logo=python&logoColor=white)
![CUDA 12.x](https://img.shields.io/badge/CUDA-12.x-76B900?logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/license-proprietary-gray)

---

## Features

| Capability | Engine | GPU? |
|---|---|---|
| **Classification** — categorical prediction on CSV data | AutoGluon Tabular | Optional |
| **Regression** — numerical prediction on CSV data | AutoGluon Tabular | Optional |
| **Time series forecasting** — multi-step ahead predictions | AutoGluon TimeSeries | Optional |
| **Text classification** — NLP via sentence embeddings | sentence-transformers | Optional |
| **Call audio analysis** — transcription + schema-based evaluation | faster-whisper + LLM | Recommended |
| **LLM explanations** — natural language model insights | Bundled llama.cpp (Qwen 3.5 4B) | Recommended |
| **Model export** — Airflow DAG or MSSQL stored procedure generation | Built-in | No |

Additional platform features:

- **Multi-user with roles** — admin approval flow, per-user quotas, RBAC
- **Fair job queue** — 1 training + 1 audio job per user, automatic queuing
- **Resource management** — VRAM/RAM budgeting, automatic model caching & eviction
- **Explainability** — SHAP feature importance, correlation analysis, seasonal decomposition
- **Batch prediction** — upload a CSV and get predictions for all rows
- **Example datasets** — built-in sample datasets for every task type to get started quickly

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/etohimself/vantager.git
cd vantager

python3 -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
```

Edit `.env` — at minimum, change the admin password:

```env
ADMIN_USER=admin
ADMIN_PASSWORD=YourSecurePassword123!
```

All other settings have sensible defaults. See [Configuration](#configuration) below for the full list.

### 3. Run

**Linux/Mac:**
```bash
chmod +x start.sh
./start.sh
```

**Windows:**
```
start.bat
```

**Or directly:**
```bash
python server.py
```

Open **http://localhost:8080** and log in with your admin credentials.

---

## Example Datasets

The `example/` directory contains ready-to-use sample datasets for every task type:

| File | Task Type | Description |
|---|---|---|
| `iris_flower.csv` | Classification | Classic 3-class iris flower species dataset |
| `house_pricing.csv` | Regression | House features to price prediction |
| `air_passengers.csv` | Time Series | Monthly airline passenger counts (1949-1960) |
| `*_callcenter.mp3` | Call Audio Analysis | 50 real call center audio recordings |

These are accessible from the **Ornek Veriler** (Example Datasets) page in the web UI, where you can:

- Download CSV datasets individually
- Play audio files directly in the browser
- Download audio files individually or all at once as a ZIP

To add your own example files, simply drop `.csv` or `.mp3` files into the `example/` folder — they appear automatically.

---

## Deploy on a GPU Instance

Works on any GPU instance (Vast.ai, RunPod, Lambda, bare-metal, etc.)

### First-time setup

```bash
git clone https://github.com/etohimself/vantager.git
cd vantager

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env: set ADMIN_PASSWORD, DATA_DIR, etc.

chmod +x start.sh
./start.sh
```

### Updating after changes

```bash
cd vantager
git pull
source .venv/bin/activate
pip install -r requirements.txt   # only if deps changed
./start.sh
```

### Running with Cloudflare Tunnel

```bash
# Add to .env:
CLOUDFLARE_TUNNEL_TOKEN=your-token
```

The tunnel exposes the app on your Cloudflare domain without opening ports.

---

## Configuration

All settings are via environment variables (or `.env` file). Copy `.env.example` to `.env` to get started.

### Core

| Variable | Default | Description |
|---|---|---|
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8080` | HTTP port |
| `DATA_DIR` | `./data` | Root directory for all persistent data |
| `ADMIN_USER` | `admin` | Initial admin username |
| `ADMIN_PASSWORD` | `Admin123!` | Initial admin password (**change this!**) |
| `SECURE_COOKIES` | `false` | Set `true` when behind HTTPS proxy |
| `CORS_ORIGINS` | *(empty = allow all)* | Comma-separated allowed origins |

### Limits

| Variable | Default | Description |
|---|---|---|
| `MAX_UPLOAD_SIZE_MB` | `200` | Max CSV upload size |
| `MAX_AUDIO_FILE_SIZE_MB` | `200` | Max audio file size |
| `MAX_BATCH_ROWS` | `100000` | Max rows in batch prediction |
| `MAX_PREDICTION_LENGTH` | `500` | Max time series forecast steps |
| `MAX_MODELS_PER_USER` | `50` | Per-user model quota |
| `MAX_EXPORT_SIZE_MB` | `2048` | Max model export ZIP size |

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
├── cache/           # HuggingFace & sentence-transformers cache
├── users.json       # User accounts & roles
├── sessions.json    # Active sessions
└── activity.json    # Training & prediction activity log
```

> **Tip:** On cloud instances, point `DATA_DIR` to a persistent volume (e.g., `/workspace/data` on RunPod) so trained models and user data survive instance restarts.

---

## User Management

Role-based multi-user system with admin approval:

| Role | Train | Predict | Manage Users | Endorse Models |
|---|---|---|---|---|
| `master_admin` | Yes | Yes | Yes | Yes |
| `admin` | Yes | Yes | Yes | Yes |
| `user` | Yes | Yes | No | No |
| `pending` | No | No | No | No |

- First launch auto-creates the `master_admin` account
- New users self-register and wait for admin approval
- Each user can run **1 training job** + **1 audio job** concurrently
- Additional jobs are automatically queued (fair scheduling)

---

## Architecture

```
┌─────────────────────────────────────────────┐
│              Browser (index.html)            │
│          Tailwind CSS + Chart.js SPA         │
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
        │      DATA_DIR        │
        │  models / users /    │
        │  stt / llm / temp    │
        └─────────────────────┘
```

Everything runs in a **single process** with threading. No Redis, no Celery, no database server — just Python and the filesystem.

---

## Project Structure

```
vantager/
├── server.py            # Backend — all API routes, ML pipelines, auth, job queue
├── static/
│   └── index.html       # Frontend — single-page app (Tailwind + vanilla JS)
├── example/             # Example datasets & audio files (served via UI)
│   ├── iris_flower.csv
│   ├── house_pricing.csv
│   ├── air_passengers.csv
│   └── *.mp3            # Call center audio samples
├── .env.example         # Environment variable template
├── requirements.txt     # Python dependencies
├── start.sh             # Linux/Mac startup script
└── start.bat            # Windows startup script
```

---

## GPU Recommendations

| GPU | VRAM | Good For |
|---|---|---|
| RTX 3090 / 4090 | 24 GB | Full stack (training + Whisper + LLM) |
| A100 40 GB | 40 GB | Heavy training + large datasets |
| RTX 4080 | 16 GB | Training + Whisper (LLM on CPU) |
| T4 | 16 GB | Budget option, inference-focused |

The platform auto-detects GPU availability and adjusts. Everything works on CPU too — just slower.
