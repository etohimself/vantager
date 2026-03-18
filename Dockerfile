# syntax=docker/dockerfile:1
# ── Tahmin Platformu: GPU/ML Dockerfile ──
# Multi-stage build: NVIDIA CUDA 12.1 + Python 3.11 + ML stack
# Compiles llama.cpp from source with CUDA 12.1 for GPU-accelerated LLM.
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

# ═══ Stage 1: Build Python virtualenv + llama.cpp ═══
# Using devel image for CUDA compiler (nvcc) needed to build llama.cpp
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev \
        build-essential gcc g++ pkg-config cmake \
        libffi-dev libssl-dev libcurl4-openssl-dev git curl && \
    rm -rf /var/lib/apt/lists/*

# Build Python virtualenv
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements.txt

# Build llama.cpp with CUDA support (compiled for our exact CUDA 12.1)
RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp.git /tmp/llama.cpp && \
    cd /tmp/llama.cpp && \
    cmake -B build \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="61;70;75;80;86;89;90" \
        -DLLAMA_CURL=ON \
        -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build --config Release -j$(nproc) --target llama-server && \
    cp build/bin/llama-server /usr/local/bin/llama-server && \
    rm -rf /tmp/llama.cpp

# ═══ Stage 2: Runtime image ═══
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

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
        ffmpeg libsndfile1 libgomp1 libffi8 libssl3 libcurl4 curl && \
    curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb \
        -o /tmp/cloudflared.deb && dpkg -i /tmp/cloudflared.deb && rm /tmp/cloudflared.deb && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /usr/local/bin/llama-server /usr/local/bin/llama-server
ENV PATH="/opt/venv/bin:$PATH"

# GPU visibility
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# ML cache — inside DATA_DIR so it persists on mounted volumes (Vast.ai /workspace)
ENV HF_HOME=/app/data/cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/data/cache/sentence_transformers

# Application defaults
ENV HOST=0.0.0.0 \
    PORT=8080 \
    DATA_DIR=/app/data

WORKDIR /app

# Pre-create data subdirectories
RUN mkdir -p /app/data/models /app/data/temp /app/data/stt /app/data/llm \
    /app/data/cache/huggingface /app/data/cache/sentence_transformers

# Copy application code (data/ excluded by .dockerignore)
COPY . .
RUN chmod +x start.sh

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -sf http://localhost:8080/ || exit 1

CMD ["./start.sh"]
