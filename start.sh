#!/bin/bash
# ── Tahmin Platformu Startup Script ──
# Handles: Data dirs, ML cache, Cloudflare Tunnel, Application server

set -e

# Load .env file if present
if [ -f "$(dirname "$0")/.env" ]; then
    set -a
    . "$(dirname "$0")/.env"
    set +a
fi

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

# ── Start application (auto-restart on crash) ──
RESTART_DELAY=3
while true; do
    echo "[Server] Starting Vantager..."
    python3 server.py
    EXIT_CODE=$?
    if [ "$EXIT_CODE" -eq 0 ]; then
        echo "[Server] Exited cleanly (code 0). Stopping."
        break
    fi
    echo "[Server] Crashed with exit code $EXIT_CODE. Restarting in ${RESTART_DELAY}s..."
    sleep "$RESTART_DELAY"
done
