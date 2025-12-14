from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class ProjectConfig:
    target_col: str = "Churn"
    id_cols: List[str] = None  # set in __post_init__ style below
    artifact_name: str = "telco_churn_pipeline.joblib"
    metrics_name: str = "metrics.json"
    schema_name: str = "feature_schema.json"
    random_state: int = 42
    test_size: float = 0.2

    def __post_init__(self):
        # dataclass is frozen; not mutating here. We'll just provide a helper.
        pass


def get_id_cols() -> List[str]:
    return ["customerID"]


NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
