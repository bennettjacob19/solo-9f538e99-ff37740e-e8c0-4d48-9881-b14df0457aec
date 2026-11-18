#!/bin/sh
# Worker entrypoint: a tiny supervisor that respawns the worker process after
# a strong kill (SIGKILL via /kill or `docker compose kill`).
#
# The TRAINING GROUP still pauses: the fresh process reloads its durable shard
# and state, but has no active generation until the coordinator's /prepare
# fences it again. Late messages from the dead generation are rejected (409).
set -u

cd /app

# Handle `docker compose stop` (SIGTERM) gracefully: stop supervising.
TERMINATED=0
trap 'TERMINATED=1; kill "$CHILD" 2>/dev/null || true' TERM INT

while [ "$TERMINATED" -eq 0 ]; do
    echo "[entrypoint] starting worker $WORKER_ID (pid $$)"
    python -I -m uvicorn worker:app --host 0.0.0.0 --port 8000 &
    CHILD=$!
    wait "$CHILD"
    code=$?
    if [ "$TERMINATED" -eq 1 ]; then
        break
    fi
    echo "[entrypoint] worker $WORKER_ID exited ($code); respawning in 1s"
    sleep 1
done
echo "[entrypoint] supervisor for $WORKER_ID exiting"
