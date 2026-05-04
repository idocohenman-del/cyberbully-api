#!/bin/bash
set -e

# gdown downloads into /data/<folder-name>, so MODEL_DIR points there directly
MODEL_DIR="${MODEL_DIR:-/data/distilbert_cyberbullying}"
MODEL_DRIVE_URL="${MODEL_DRIVE_URL:-}"

# Download model from Google Drive on first boot (skipped on subsequent restarts)
if [ ! -f "$MODEL_DIR/model.safetensors" ]; then
    echo "Model not found at $MODEL_DIR"
    if [ -n "$MODEL_DRIVE_URL" ]; then
        echo "Downloading model from Google Drive (~260 MB, one-time)..."
        mkdir -p /data
        gdown --folder "$MODEL_DRIVE_URL" -O /data --remaining-ok
        echo "Model download complete."
    else
        echo "WARNING: MODEL_DRIVE_URL is not set — classify endpoint will be unavailable."
    fi
else
    echo "Model already present at $MODEL_DIR — skipping download."
fi

exec python -m uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8000}"
