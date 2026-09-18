#!/usr/bin/env bash
#
# bin/serve_https.sh
# ==================
# Serve LegalMind AI over HTTPS so that voice input works off-localhost.
#
# Port 8443 rather than 443: binding a port below 1024 needs root, and this
# account does not have it. Nothing about the secure-context rule cares which
# port is used — only that the scheme is https.
#
#   LEGALMIND_HTTPS_PORT   listen port          (default 8443)
#   LEGALMIND_HOST         bind address         (default 0.0.0.0)
#   LEGALMIND_DB_PATH      chat database        (default ./data/legalmind.db)

set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${LEGALMIND_HTTPS_PORT:-8443}"
HOST="${LEGALMIND_HOST:-0.0.0.0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
CRT="certs/legalmind.crt"
KEY="certs/legalmind.key"

if [ ! -f "$CRT" ] || [ ! -f "$KEY" ]; then
    echo "No certificate found. Run:  bash bin/setup_https.sh" >&2
    exit 1
fi

LAN_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
echo "⚖️  LegalMind AI over HTTPS"
echo "   local:   https://localhost:$PORT"
[ -n "$LAN_IP" ] && echo "   network: https://$LAN_IP:$PORT      <- share this one"
echo "   (first visit shows a certificate warning unless certs/legalmind-ca.crt"
echo "    is installed on the device — see docs/https_and_voice.md)"
echo

exec .venv/bin/uvicorn app.main:app \
    --host "$HOST" --port "$PORT" --workers 1 \
    --ssl-certfile "$CRT" --ssl-keyfile "$KEY"
