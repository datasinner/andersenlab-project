#!/bin/sh
set -e

alembic upgrade head

# Ingest bundled tariff documents (idempotent). A failure is logged but does
# not stop the API: documents can still be uploaded through it.
python scripts/ingest.py --dir "${TARIFFS_DIR:-./data/tariffs}" \
    || echo "warning: tariff ingestion failed; see the log above" >&2

if [ "${UVICORN_RELOAD:-false}" = "true" ]; then
    exec uvicorn app.main:app --host 0.0.0.0 --port 8000 \
        --workers 1 --reload --timeout-graceful-shutdown 30
else
    exec uvicorn app.main:app --host 0.0.0.0 --port 8000 \
        --workers "${UVICORN_WORKERS:-2}" --timeout-graceful-shutdown 30
fi
