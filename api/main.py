from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from churn.predict import load_pipeline, predict_one
from api.schemas import PredictRequest, PredictResponse

APP_NAME = "telco-churn-api"

# Allow override for deployment environments
ARTIFACT_PATH = Path(os.getenv("MODEL_PATH", "artifacts/telco_churn_pipeline.joblib"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "telco-churn-hgbdt-v1")

app = FastAPI(title=APP_NAME, version=MODEL_VERSION)

# Streamlit runs in a browser, permissive CORS for demo purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Lazily loaded, cached pipeline for inference
_PIPELINE = None


def get_pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        if not ARTIFACT_PATH.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {ARTIFACT_PATH}. "
                "Ensure the trained pipeline artifact is available at startup."
            )
        _PIPELINE = load_pipeline(ARTIFACT_PATH)
    return _PIPELINE


@app.get("/health")
def health() -> Dict[str, Any]:
    """
    Health endpoint for local checks and HF deployment monitoring.
    """
    try:
        _ = get_pipeline()
        return {"status": "ok", "model_version": MODEL_VERSION}
    except Exception as e:
        return {"status": "error", "error": str(e), "model_version": MODEL_VERSION}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """
    Predict churn probability + class.
    """
    try:
        pipeline = get_pipeline()
        result = predict_one(pipeline, req.model_dump())
        return PredictResponse(**result, model_version=MODEL_VERSION)
    except ValueError as ve:
        # Validation or feature missing error
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
