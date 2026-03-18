# syntax=docker/dockerfile:1
# ── Tahmin Platformu: GPU/ML Dockerfile ──
# Multi-stage build: NVIDIA CUDA 12.4 + Python 3.11 + ML stack
# Includes Cloudflare Tunnel support for Vast.ai deployment.
#
# Build:  docker build -t tahmin-platformu .
# Run:    docker run --gpus all -p 8080:8080 tahmin-platformu
#
# With Cloudflare Tunnel:
#   docker run --gpus all -e CLOUDFLARE_TUNNEL_TOKEN=your-token tahmin-platformu
#
# Persistent data: mount or set DATA_DIR env var
#   docker run --gpus all -p 8080:8080 -v my-data:/app/data tahmin-platformu

# ═══ Stage 1: Build Python virtualenv ═══
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev \
        build-essential gcc g++ pkg-config \
        libffi-dev libssl-dev git curl && \
    rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements.txt

# ═══ Stage 2: Runtime image ═══
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

LABEL org.opencontainers.image.title="Tahmin Platformu" \
      org.opencontainers.image.description="AutoGluon ML prediction platform with Whisper STT and LLM" \
      org.opencontainers.image.source="https://github.com/etohimself/vantager"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install runtime deps + cloudflared for tunnel support
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv \
        ffmpeg libsndfile1 libgomp1 libffi8 libssl3 curl && \
    curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb \
        -o /tmp/cloudflared.deb && dpkg -i /tmp/cloudflared.deb && rm /tmp/cloudflared.deb && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# GPU visibility
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# ML cache (HuggingFace models, sentence-transformers, etc.)
ENV HF_HOME=/root/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/root/.cache/torch/sentence_transformers

# Application defaults
ENV HOST=0.0.0.0 \
    PORT=8080 \
    DATA_DIR=/app/data

WORKDIR /app

# Pre-create data subdirectories
RUN mkdir -p /app/data/models /app/data/temp /app/data/stt /app/data/llm

# Copy application code (data/ excluded by .dockerignore)
COPY . .
RUN chmod +x start.sh

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -sf http://localhost:8080/ || exit 1

CMD ["./start.sh"]
