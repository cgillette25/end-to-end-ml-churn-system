from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from churn.config import ALL_FEATURES, ProjectConfig
from churn.data import clean_telco, load_raw_csv, make_split
from churn.features import build_preprocessor
from churn.metrics import compute_binary_metrics
from churn.model import get_model_specs


def build_pipeline(estimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("model", estimator),
        ]
    )


def fit_best_model(
    X_train,
    y_train,
    random_state: int = 42,
    n_iter: int = 20,
) -> Tuple[Pipeline, Dict]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    best_score = -np.inf
    best_pipeline = None
    best_info: Dict = {}

    for spec in get_model_specs(random_state=random_state):
        pipe = build_pipeline(spec.estimator)

        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=spec.search_space,
            n_iter=n_iter,
            scoring="roc_auc",
            cv=cv,
            random_state=random_state,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X_train, y_train)

        if search.best_score_ > best_score:
            best_score = float(search.best_score_)
            best_pipeline = search.best_estimator_
            best_info = {
                "model_name": spec.name,
                "cv_best_roc_auc": float(search.best_score_),
                "best_params": search.best_params_,
            }

    if best_pipeline is None:
        raise RuntimeError("Failed to fit any model.")
    return best_pipeline, best_info


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train Telco churn model and export artifacts.")
    parser.add_argument("--data", required=True, help="Path to Telco-Customer-Churn.csv")
    parser.add_argument("--out", required=True, help="Artifacts output directory")
    parser.add_argument("--n_iter", type=int, default=20, help="RandomizedSearch iterations per model")
    args = parser.parse_args(argv)

    cfg = ProjectConfig()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_raw = load_raw_csv(args.data)
    df = clean_telco(df_raw)
    split = make_split(df, test_size=cfg.test_size, random_state=cfg.random_state)

    best_pipeline, best_info = fit_best_model(
        split.X_train,
        split.y_train,
        random_state=cfg.random_state,
        n_iter=args.n_iter,
    )

    # Evaluate on held-out test
    proba = best_pipeline.predict_proba(split.X_test)[:, 1]
    test_metrics = compute_binary_metrics(split.y_test, proba, threshold=0.5)

    # Save pipeline
    model_path = out_dir / cfg.artifact_name
    joblib.dump(best_pipeline, model_path)

    # Save metrics + schema
    metrics_payload = {
        "dataset": "IBM Telco Customer Churn",
        "rows": int(df.shape[0]),
        "features": ALL_FEATURES,
        "target": "Churn",
        "best_model": best_info,
        "test_metrics": test_metrics,
    }
    write_json(out_dir / cfg.metrics_name, metrics_payload)

    schema_payload = {
        "title": "Telco Churn Prediction Input",
        "type": "object",
        "required": ALL_FEATURES,
        "properties": {
            # Keep types broad; FastAPI/Pydantic will enforce more strictly later
            "SeniorCitizen": {"type": "integer"},
            "tenure": {"type": "number"},
            "MonthlyCharges": {"type": "number"},
            "TotalCharges": {"type": "number"},
        },
    }
    # Add categorical properties
    for c in [f for f in ALL_FEATURES if f not in schema_payload["properties"]]:
        schema_payload["properties"][c] = {"type": "string"}

    write_json(out_dir / cfg.schema_name, schema_payload)

    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {out_dir / cfg.metrics_name}")
    print(f"Saved schema: {out_dir / cfg.schema_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
