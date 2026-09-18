import os
import signal
import subprocess
import time

# Find running uvicorn on port 8443
output = subprocess.check_output(["pgrep", "-f", "8443"]).decode().strip()
pids = [int(p) for p in output.split()]
print(f"Found PIDs: {pids}")

for p in pids:
    try:
        os.kill(p, signal.SIGTERM)
        print(f"Sent SIGTERM to {p}")
    except Exception as e:
        print(f"Kill failed for {p}: {e}")

time.sleep(3)

# Spawn uvicorn cleanly
cmd = [
    "/home/sece2026-student12/LegalMindAI/.venv/bin/uvicorn",
    "app.main:app",
    "--host", "0.0.0.0",
    "--port", "8443",
    "--workers", "1",
    "--ssl-certfile", "certs/legalmind.crt",
    "--ssl-keyfile", "certs/legalmind.key",
]

with open("/home/sece2026-student12/LegalMindAI/logs/uvicorn-https.log", "a") as logfile:
    p = subprocess.Popen(
        cmd,
        cwd="/home/sece2026-student12/LegalMindAI",
        stdout=logfile,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    print(f"Spawned new uvicorn server with PID {p.pid}")

time.sleep(2)
print("Restart script finished.")
