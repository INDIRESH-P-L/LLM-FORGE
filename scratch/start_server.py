#!/usr/bin/env python3
"""
start_server.py — LegalMind AI server launcher.
IDEMPOTENT: will not start a second instance if one is already running.
"""
import os
import subprocess
import sys
import time

CWD = "/home/sece2026-student12/LegalMindAI"
LOG_FILE = os.path.join(CWD, "logs", "legalmind_server.log")

# ── Idempotency guard ────────────────────────────────────────────────────────
# If any uvicorn process for this app is already running on port 8443,
# exit immediately. This prevents duplicate instances during model load.
try:
    result = subprocess.run(
        ["pgrep", "-fa", "uvicorn"],
        capture_output=True, text=True, timeout=5
    )
    for line in result.stdout.splitlines():
        if "app.main:app" in line and "8443" in line:
            pid = line.split()[0]
            print(f"Server already running (PID {pid}). Not starting another instance.")
            sys.exit(0)
except Exception:
    pass  # If pgrep fails, proceed with start attempt

# ── Environment ──────────────────────────────────────────────────────────────
env = os.environ.copy()
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
env["LLM_GPU"] = "2"
env["EMBED_GPU"] = "4"
env["PYTHONUNBUFFERED"] = "1"

cmd = [
    "/home/sece2026-student12/LegalMindAI/.venv/bin/uvicorn",
    "app.main:app",
    "--host", "0.0.0.0",
    "--port", "8443",
    "--workers", "1",
]

crt = os.path.join(CWD, "certs", "legalmind.crt")
key = os.path.join(CWD, "certs", "legalmind.key")
if os.path.exists(crt) and os.path.exists(key):
    cmd.extend(["--ssl-certfile", crt, "--ssl-keyfile", key])

with open(LOG_FILE, "a") as logf:
    logf.write(f"\n--- Starting LegalMind AI at {time.ctime()} ---\n")
    logf.flush()
    proc = subprocess.Popen(
        cmd,
        cwd=CWD,
        env=env,
        stdout=logf,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )

print(f"Server launched with PID {proc.pid}. Logging to {LOG_FILE}")
