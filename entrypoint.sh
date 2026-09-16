#!/usr/bin/env bash
# Container entrypoint: pull weights, then hand the PID to uvicorn.
#
# `exec` on the last line matters — uvicorn becomes PID 1 and receives
# SIGTERM directly, so the platform's graceful-shutdown window actually
# drains in-flight requests instead of killing a shell wrapper.
set -euo pipefail

echo "[entrypoint] $(date -u +%FT%TZ) starting ${OCR_APP_NAME:-ocr-platform}"

python scripts/fetch_models.py

PORT="${PORT:-8000}"
WORKERS="${WEB_CONCURRENCY:-1}"

# One worker by default and that is deliberate, not laziness: each worker
# loads its own full copy of every model into memory. On a 2GB instance a
# second worker is an OOM, not throughput. Scale by adding replicas.
echo "[entrypoint] uvicorn on 0.0.0.0:${PORT} (workers=${WORKERS})"
exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --workers "${WORKERS}" \
    --timeout-keep-alive 75 \
    --proxy-headers \
    --forwarded-allow-ips '*'
