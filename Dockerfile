# ── Tahmin Platformu: GPU/ML Python Dockerfile ──
# NVIDIA CUDA tabanlı — PyTorch, AutoGluon, Whisper, sentence-transformers destekli.
# DeployShield uyumlu: app-data:/app/data + ml-cache:/home/appuser/.cache

# ═══ Aşama 1: Derleme ═══
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev \
    build-essential gcc g++ pkg-config \
    libffi-dev libssl-dev git curl \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ═══ Aşama 2: Çalışma Zamanı ═══
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv \
    ffmpeg libsndfile1 libgomp1 libffi8 libssl3 curl \
    && rm -rf /var/lib/apt/lists/*

# Güvenlik: root olmayan kullanıcı
RUN groupadd -r appgroup && useradd -r -g appgroup -m appuser

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# GPU erişimi
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# HuggingFace / sentence-transformers cache (DeployShield ml-cache volume ile eşleşir)
ENV HF_HOME=/home/appuser/.cache/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/home/appuser/.cache/torch/sentence_transformers

WORKDIR /app

# ── Data dizini: Docker named volume mount noktası ──
# DeployShield docker-compose: "app-data:/app/data"
# Bu dizin IMAGE'da oluşturulmalı ki volume ilk mount'ta appuser ownership'i alsın.
RUN mkdir -p /app/data/models /app/data/temp /app/data/stt \
    && chown -R appuser:appgroup /app/data

# Uygulama kodunu kopyala (data/ .dockerignore ile hariç tutulur)
COPY --chown=appuser:appgroup . .

USER appuser
EXPOSE 8080
CMD ["python3.11", "server.py"]
