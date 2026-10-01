#!/bin/sh
set -e

alembic upgrade head

if [ "${UVICORN_RELOAD:-false}" = "true" ]; then
    exec uvicorn app.main:app --host 0.0.0.0 --port 8000 \
        --workers 1 --reload --timeout-graceful-shutdown 30
else
    exec uvicorn app.main:app --host 0.0.0.0 --port 8000 \
        --workers "${UVICORN_WORKERS:-2}" --timeout-graceful-shutdown 30
fi
