#!/bin/bash
# ── Tahmin Platformu Startup Script ──
# Handles: Data dirs, ML cache, Cloudflare Tunnel, Application server

set -e

# Create data directories
DATA="${DATA_DIR:-./data}"
mkdir -p "${DATA}"/{models,temp,stt,llm,cache/huggingface,cache/sentence_transformers}

# Point ML model caches to DATA_DIR so they persist
export HF_HOME="${DATA}/cache/huggingface"
export SENTENCE_TRANSFORMERS_HOME="${DATA}/cache/sentence_transformers"

# ── Cloudflare Tunnel (if token provided) ──
if [ -n "$CLOUDFLARE_TUNNEL_TOKEN" ]; then
    echo "[Tunnel] Starting Cloudflare Tunnel..."
    cloudflared tunnel run --token "$CLOUDFLARE_TUNNEL_TOKEN" &
    TUNNEL_PID=$!
    echo "[Tunnel] Started (PID $TUNNEL_PID)"
else
    echo "[Tunnel] No CLOUDFLARE_TUNNEL_TOKEN set — skipping tunnel"
fi

# ── Start application ──
exec python3.11 server.py
