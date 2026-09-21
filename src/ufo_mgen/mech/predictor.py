"""Optional K/G/E surrogate used only for candidate prioritization."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

#: Mechanical targets used by UFO-Mech.
TARGET_COLUMNS: List[str] = ["log_K", "log_G", "log_E"]

#: Hyper-parameters of the released surrogate.
PRODUCTION_ARCH: Dict[str, Any] = {
    "n_estimators": 220,
    "max_features": 0.75,
    "min_samples_leaf": 1,
    "bootstrap": False,
    "imputation": "median",
}


def build_predictor(random_state: int = 0, **overrides: Any):
    """Instantiate the surrogate at its production configuration."""
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    cfg = dict(PRODUCTION_ARCH)
    cfg.update(overrides)
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=cfg["imputation"])),
            (
                "regressor",
                ExtraTreesRegressor(
                    n_estimators=int(cfg["n_estimators"]),
                    max_features=cfg["max_features"],
                    min_samples_leaf=int(cfg["min_samples_leaf"]),
                    bootstrap=bool(cfg["bootstrap"]),
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def fit_predictor(
    features,
    targets,
    sample_weight: Optional[Sequence[float]] = None,
    random_state: int = 0,
):
    """Fit the surrogate. ``targets`` must carry :data:`TARGET_COLUMNS`."""
    model = build_predictor(random_state=random_state)
    kwargs = {}
    if sample_weight is not None:
        kwargs["regressor__sample_weight"] = sample_weight
    model.fit(features, targets, **kwargs)
    return model


def predict_moduli(model, features):
    """Predict K, G and E and expand the logs back to GPa."""
    import numpy as np
    import pandas as pd

    raw = np.asarray(model.predict(features), dtype=float)
    out = pd.DataFrame(raw, columns=TARGET_COLUMNS)
    for log_col, plain in (("log_K", "K"), ("log_G", "G"), ("log_E", "E")):
        out[plain] = np.exp(out[log_col])
    return out
