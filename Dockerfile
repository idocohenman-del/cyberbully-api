FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt gdown

# CPU-only PyTorch + ML inference deps (separate layer for caching)
RUN pip install --no-cache-dir numpy transformers accelerate \
    torch --index-url https://download.pytorch.org/whl/cpu

COPY api/ ./api/
COPY src/ ./src/
# cache-bust: force rebuild of app layers

COPY start.sh ./start.sh
RUN chmod +x ./start.sh

EXPOSE 8000

CMD ["./start.sh"]
