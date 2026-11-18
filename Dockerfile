FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Numeric + web stack only. Pinned to known-good majors; image rebuilds
# resolve patch versions via the lock constraints below.
RUN pip install --no-cache-dir \
        "numpy==1.26.4" \
        "fastapi==0.115.6" \
        "uvicorn[standard]==0.34.0" \
        "httpx==0.28.1" \
        "pydantic==2.10.4" \
        "python-multipart==0.0.20"

COPY app/ /app/

RUN chmod +x /app/entrypoint-worker.sh /app/entrypoint-coordinator.sh

EXPOSE 8000
