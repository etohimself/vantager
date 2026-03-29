@echo off
setlocal enabledelayedexpansion

:: ── Tahmin Platformu Startup Script (Windows) ──
:: Handles: Data dirs, ML cache, Cloudflare Tunnel, Application server

:: Load .env file if present
if exist "%~dp0.env" (
    echo [Setup] Loading .env file...
    for /f "usebackq eol=# tokens=1,* delims==" %%A in ("%~dp0.env") do (
        set "VAR_NAME=%%A"
        set "VAR_VAL=%%B"
        :: Remove surrounding quotes if they exist in the .env file
        if defined VAR_VAL (
            set "VAR_VAL=!VAR_VAL:"=!"
            set "!VAR_NAME!=!VAR_VAL!"
        )
    )
)

:: Create data directories
if "!DATA_DIR!"=="" set "DATA_DIR=.\data"

:: Windows mkdir automatically creates intermediate directories (like mkdir -p)
:: 2>nul hides the "Directory already exists" error on subsequent runs
mkdir "!DATA_DIR!\models" 2>nul
mkdir "!DATA_DIR!\temp" 2>nul
mkdir "!DATA_DIR!\stt" 2>nul
mkdir "!DATA_DIR!\llm" 2>nul
mkdir "!DATA_DIR!\cache\huggingface" 2>nul
mkdir "!DATA_DIR!\cache\sentence_transformers" 2>nul

:: Point ML model caches to DATA_DIR so they persist
set "HF_HOME=!DATA_DIR!\cache\huggingface"
set "SENTENCE_TRANSFORMERS_HOME=!DATA_DIR!\cache\sentence_transformers"

:: ── Cloudflare Tunnel (if token provided) ──
if not "!CLOUDFLARE_TUNNEL_TOKEN!"=="" (
    echo [Tunnel] Starting Cloudflare Tunnel...
    :: 'start /b' runs the process in the background, similar to '&' in Bash
    start /b cloudflared tunnel run --token "!CLOUDFLARE_TUNNEL_TOKEN!"
    echo [Tunnel] Started in background
) else (
    echo [Tunnel] No CLOUDFLARE_TUNNEL_TOKEN set -- skipping tunnel
)

:: ── Start application (auto-restart on crash) ──
set "RESTART_DELAY=3"
:restart_loop
echo [Server] Starting Vantager...
python server.py
if !ERRORLEVEL! == 0 (
    echo [Server] Exited cleanly. Stopping.
    goto :end
)
echo [Server] Crashed with exit code !ERRORLEVEL!. Restarting in %RESTART_DELAY%s...
timeout /t %RESTART_DELAY% /nobreak >nul
goto :restart_loop

:end
endlocal