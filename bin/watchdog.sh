#!/usr/bin/env bash
# LegalMind AI Self-Healing Watchdog
# Checks https://127.0.0.1:8443/health every run.
# If unhealthy AND no server process is already running, restarts.
# IMPORTANT: Model loading takes ~40s. Never restart if a process exists.

CWD="/home/sece2026-student12/LegalMindAI"
LOG="${CWD}/logs/watchdog.log"
DATE="$(date '+%Y-%m-%d %H:%M:%S')"

mkdir -p "${CWD}/logs"

# ── Step 1: Check if a server process is already running ────────────────────
# Match both "python3.12 .venv/bin/uvicorn" and "python3.12 /path/uvicorn"
# on port 8443. If ANY process is found, do NOT restart.
RUNNING_PID="$(pgrep -f "uvicorn app.main:app.*8443\|8443.*uvicorn app.main:app" 2>/dev/null | head -1)"
if [ -z "$RUNNING_PID" ]; then
    # Also try matching by port argument directly
    RUNNING_PID="$(pgrep -fa "uvicorn" 2>/dev/null | grep "8443" | grep "app.main:app" | awk '{print $1}' | head -1)"
fi
if [ -n "$RUNNING_PID" ]; then
    echo "[$DATE] Server process $RUNNING_PID is running (may still be loading model). Skipping restart." >> "$LOG"
    exit 0
fi

# ── Step 2: No process running — check health to confirm outage ─────────────
HTTP_CODE="$(curl -k -s -m 6 -o /dev/null -w "%{http_code}" https://127.0.0.1:8443/health 2>/dev/null || true)"
if [ "$HTTP_CODE" != "200" ]; then
    HTTP_CODE="$(curl -s -m 6 -o /dev/null -w "%{http_code}" http://127.0.0.1:8443/health 2>/dev/null || echo "000")"
fi

if [ "$HTTP_CODE" = "200" ]; then
    # Server is healthy
    exit 0
fi

echo "[$DATE] [WARNING] No process running and health check failed (HTTP $HTTP_CODE). Initiating recovery..." >> "$LOG"

# ── Step 3: Restart (only reached when process is truly gone) ────────────────
if systemctl --user is-active --quiet legalmind.service 2>/dev/null; then
    echo "[$DATE] Restarting systemd service legalmind..." >> "$LOG"
    systemctl --user restart legalmind.service 2>&1 >> "$LOG"
else
    echo "[$DATE] Launching detached server via start_server.py..." >> "$LOG"
    "${CWD}/.venv/bin/python3" "${CWD}/scratch/start_server.py" 2>&1 >> "$LOG"
fi

echo "[$DATE] Recovery procedure executed." >> "$LOG"
