#!/usr/bin/env python3
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

def get_pid_on_port(port=8080):
    try:
        out = subprocess.check_output(f"lsof -ti :{port}", shell=True).decode().strip()
        pids = [int(p) for p in out.split() if p.isdigit()]
        return pids
    except Exception:
        return []

def main():
    port = 8080
    pids = get_pid_on_port(port)
    print(f"PIDs on port {port}: {pids}")
    for pid in pids:
        print(f"Terminating PID {pid}...")
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception as e:
            print(f"Error killing {pid}: {e}")

    # Wait for port to be released
    for _ in range(15):
        if not get_pid_on_port(port):
            break
        time.sleep(1)
    
    # If still alive, SIGKILL
    for pid in get_pid_on_port(port):
        print(f"Force killing PID {pid}...")
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            pass

    time.sleep(2)

    # Launch server
    venv_uvicorn = "/home/sece2026-student12/LegalMindAI/.venv/bin/uvicorn"
    if not os.path.exists(venv_uvicorn):
        venv_uvicorn = "uvicorn"

    log_path = "/home/sece2026-student12/LegalMindAI/logs/server.log"
    log_file = open(log_path, "a")

    cmd = [
        venv_uvicorn,
        "app.main:app",
        "--host", "0.0.0.0",
        "--port", "8080",
        "--workers", "1"
    ]

    print(f"Starting server: {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONPATH"] = "/home/sece2026-student12/LegalMindAI:/home/sece2026-student12/LegalMindAI/scripts"
    proc = subprocess.Popen(
        cmd,
        cwd="/home/sece2026-student12/LegalMindAI",
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env
    )
    print(f"Server started with PID: {proc.pid}")

    # Poll health
    url = f"http://localhost:{port}/health"
    print(f"Waiting for {url} to be ready...")
    ready = False
    for i in range(40):
        time.sleep(2)
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode())
                    print(f"Health response: {data}")
                    if data.get("status") == "ok":
                        ready = True
                        break
        except Exception as e:
            print(f"Waiting ({i+1}/40)... {e}")

    if ready:
        print("Server is UP and HEALTHY on port 8080!")
        sys.exit(0)
    else:
        print("Server failed to become healthy within timeout.")
        sys.exit(1)

if __name__ == "__main__":
    main()
