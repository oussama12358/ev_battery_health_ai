# ─────────────────────────────────────────────────────────────────────────────
# EV Battery Health AI — Dockerfile
# Multi-stage build: slim base + only required packages
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim AS base

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ─────────────────────────────────────────────────────────────────────────────
# Dependencies
# ─────────────────────────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ─────────────────────────────────────────────────────────────────────────────
# Application code
# ─────────────────────────────────────────────────────────────────────────────
COPY . .

# Pre-create directories
RUN mkdir -p data/raw data/processed models/saved

# ─────────────────────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# ─────────────────────────────────────────────────────────────────────────────
# Expose ports
#   8000 → FastAPI
#   8501 → Streamlit
# ─────────────────────────────────────────────────────────────────────────────
EXPOSE 8000 8501

# Default: train then launch API
CMD ["sh", "-c", "python training/run_all.py && uvicorn api.main:app --host 0.0.0.0 --port 8000"]
