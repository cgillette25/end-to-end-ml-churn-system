from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from churn.config import ALL_FEATURES, NUMERIC_FEATURES, get_id_cols


@dataclass(frozen=True)
class DatasetSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def load_raw_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")
    return pd.read_csv(path)


def clean_telco(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop identifier(s) if present
    for col in get_id_cols():
        if col in df.columns:
            df = df.drop(columns=[col])

    # Target mapping: Yes/No -> 1/0 (keep original column name)
    if "Churn" not in df.columns:
        raise ValueError("Expected target column 'Churn' not found in dataset.")
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    if df["Churn"].isna().any():
        raise ValueError("Unexpected values in 'Churn' column. Expected only Yes/No.")

    # TotalCharges is often a string with blanks; coerce to numeric
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Ensure numeric columns are numeric (coerce safe)
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep only the expected features + target (guards against extra cols)
    expected_cols = set(ALL_FEATURES + ["Churn"])
    missing = [c for c in (ALL_FEATURES + ["Churn"]) if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing expected columns: {missing}")

    df = df[list(ALL_FEATURES) + ["Churn"]]
    return df


def make_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> DatasetSplit:
    X = df.drop(columns=["Churn"])
    y = df["Churn"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return DatasetSplit(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
