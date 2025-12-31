#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=/app/src

# Streamlit should call the internal FastAPI service inside the container
export CHURN_API_URL=${CHURN_API_URL:-http://127.0.0.1:8000}

# Path to the trained model artifact
export MODEL_PATH=${MODEL_PATH:-/app/artifacts/telco_churn_pipeline.joblib}

echo "Starting FastAPI on :8000 ..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 &

echo "Starting Streamlit on :7860 ..."
streamlit run ui/streamlit_app.py \
  --server.port 7860 \
  --server.address 0.0.0.0
