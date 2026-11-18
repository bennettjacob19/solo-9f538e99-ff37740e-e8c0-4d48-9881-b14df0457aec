#!/bin/sh
# Coordinator entrypoint. On container restart the FastAPI lifespan detects
# any round that was in flight, marks it paused, and waits for an explicit
# resume so no unconfirmed update can silently advance.
set -eu
cd /app
mkdir -p "$COORD_DATA" "$COORD_DATA/weights"
exec python -I -m uvicorn coordinator:app --host 0.0.0.0 --port 8000
