from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: Any
    search_space: Dict[str, Any]


def get_model_specs(random_state: int = 42) -> Tuple[ModelSpec, ...]:
    # Note: parameters are prefixed with "model__" because in our pipeline
    # the final step will be named "model".
    return (
        ModelSpec(
            name="logreg",
            estimator=LogisticRegression(
                max_iter=2000,
                solver="lbfgs",
                n_jobs=None,
            ),
            search_space={
                "model__C": [0.1, 0.3, 1.0, 3.0, 10.0],
                "model__class_weight": [None, "balanced"],
            },
        ),
        ModelSpec(
            name="random_forest",
            estimator=RandomForestClassifier(
                n_estimators=400,
                random_state=random_state,
                n_jobs=-1,
            ),
            search_space={
                "model__max_depth": [None, 4, 6, 10, 16],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", "log2", None],
                "model__class_weight": [None, "balanced"],
            },
        ),
        ModelSpec(
            name="hist_gbdt",
            estimator=HistGradientBoostingClassifier(
                random_state=random_state,
            ),
            search_space={
                "model__learning_rate": [0.03, 0.05, 0.1, 0.2],
                "model__max_depth": [None, 3, 5, 8],
                "model__max_leaf_nodes": [15, 31, 63],
                "model__min_samples_leaf": [20, 40, 80],
                "model__l2_regularization": [0.0, 0.1, 1.0],
            },
        ),
    )
