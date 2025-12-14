from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

from churn.config import ALL_FEATURES


def load_pipeline(path: str | Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found at: {path}")
    return joblib.load(path)


def predict_one(pipeline, payload: Dict[str, Any]) -> Dict[str, Any]:
    # Ensure column order / presence
    missing = [f for f in ALL_FEATURES if f not in payload]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    X = pd.DataFrame([{k: payload[k] for k in ALL_FEATURES}])
    proba = float(pipeline.predict_proba(X)[:, 1][0])
    pred = int(proba >= 0.5)
    return {
        "churn_probability": proba,
        "churn_prediction": pred,
        "threshold": 0.5,
    }
