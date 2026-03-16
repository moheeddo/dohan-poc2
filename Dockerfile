# KHNP Education AI Platform - Application Container
# Python 3.13 + FastAPI + 전체 파이프라인 모듈
FROM python:3.13-slim AS base

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY src/ ./src/
COPY web/ ./web/
COPY demo/ ./demo/
COPY tests/ ./tests/

# Data directories
RUN mkdir -p data/raw_slides data/question_versions data/sme_reviews data/few_shot_examples

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run
EXPOSE 8080
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
