#!/bin/bash
# ── Tahmin Platformu Startup Script ──
# Handles: Cloudflare Tunnel (optional) + Application server

set -e

# Create data directories (handles both /app/data and custom DATA_DIR)
mkdir -p "${DATA_DIR:-/app/data}"/{models,temp,stt,llm}

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
