#!/usr/bin/env bash
# Redirect plain-HTTP visitors to the HTTPS app. Loads no model.
set -euo pipefail
cd "$(dirname "$0")/.."
PORT="${LEGALMIND_HTTP_PORT:-8090}"
echo "HTTP redirector on :$PORT  ->  https://<host>:${LEGALMIND_HTTPS_PORT:-8443}"
exec .venv/bin/uvicorn bin.http_redirect:app --host "${LEGALMIND_HOST:-0.0.0.0}" --port "$PORT"
