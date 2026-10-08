"""The demand forecaster: a LightGBM model with recursive multi-step forecasting."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from .features import (
    FEATURE_COLUMNS,
    LAGS,
    ROLLING,
    TARGET,
    add_calendar,
    build_features,
    split_xy,
)

_DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 400,
    "num_leaves": 48,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "min_child_samples": 30,
    "random_state": 42,
    "n_jobs": 1,
    "verbose": -1,
}


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


@dataclass
class TrainingReport:
    version: str
    trained_at: str
    n_rows: int
    metrics: dict[str, float]
    params: dict[str, Any]


class DemandForecaster:
    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = {**_DEFAULT_PARAMS, **(params or {})}
        self.model: LGBMRegressor | None = None
        self.version: str | None = None
        self.metadata: dict[str, Any] = {}

    def _version(self, df: pd.DataFrame) -> str:
        fingerprint = hashlib.sha256()
        fingerprint.update(str(df.shape).encode())
        fingerprint.update(pd.util.hash_pandas_object(df[TARGET], index=False).values.tobytes())
        payload = json.dumps({"params": self.params, "features": FEATURE_COLUMNS}, sort_keys=True)
        fingerprint.update(payload.encode())
        return "m_" + fingerprint.hexdigest()[:12]

    def train(self, df: pd.DataFrame, test_fraction: float = 0.2) -> TrainingReport:
        feat = build_features(df)
        split = int(len(feat) * (1 - test_fraction))
        train_df, test_df = feat.iloc[:split], feat.iloc[split:]
        x_train, y_train = split_xy(train_df)
        x_test, y_test = split_xy(test_df)

        self.model = LGBMRegressor(**self.params)
        self.model.fit(x_train, y_train)
        pred = self.model.predict(x_test)

        metrics = {
            "mae": round(mae(y_test, pred), 4),
            "rmse": round(rmse(y_test, pred), 4),
            "mape": round(mape(y_test, pred), 4),
        }
        self.version = self._version(df)
        self.metadata = {
            "version": self.version,
            "trained_at": datetime.now(UTC).isoformat(),
            "n_rows": int(len(feat)),
            "metrics": metrics,
            "params": self.params,
            "features": FEATURE_COLUMNS,
        }
        return TrainingReport(
            self.version, self.metadata["trained_at"], len(feat), metrics, self.params
        )

    def predict_features(self, feat: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("model is not trained or loaded")
        return self.model.predict(feat[FEATURE_COLUMNS])

    def forecast(self, history: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
        """Forecast demand for each row in ``future`` (which has timestamp + temperature).

        Uses a recursive strategy: predict one hour, append it, and use it to build the
        lags for the next hour.
        """
        if self.model is None:
            raise RuntimeError("model is not trained or loaded")

        work = pd.concat(
            [history[["timestamp", "temperature", TARGET]].copy(),
             future[["timestamp", "temperature"]].assign(**{TARGET: np.nan})],
            ignore_index=True,
        )
        work = add_calendar(work)
        max_lag = max(LAGS + ROLLING)
        start = len(history)

        demand = work[TARGET].to_numpy(dtype=float).copy()
        preds = []
        for i in range(start, len(work)):
            row = {
                "hour": work.at[i, "hour"],
                "dayofweek": work.at[i, "dayofweek"],
                "month": work.at[i, "month"],
                "is_weekend": work.at[i, "is_weekend"],
                "temperature": work.at[i, "temperature"],
            }
            for lag in LAGS:
                row[f"lag_{lag}"] = demand[i - lag]
            for window in ROLLING:
                row[f"roll_mean_{window}"] = np.mean(demand[i - window : i])
            x = pd.DataFrame([row])[FEATURE_COLUMNS]
            yhat = float(self.model.predict(x)[0])
            demand[i] = yhat
            preds.append(yhat)

        _ = max_lag  # history must be at least max_lag long; validated by callers/tests
        return pd.DataFrame({"timestamp": future["timestamp"].to_numpy(), "forecast": preds})

    def save(self, model_dir: str | Path) -> Path:
        if self.model is None or self.version is None:
            raise RuntimeError("nothing to save; train first")
        target = Path(model_dir) / self.version
        target.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, target / "model.joblib")
        (target / "metadata.json").write_text(json.dumps(self.metadata, indent=2))
        return target

    @classmethod
    def load(cls, model_dir: str | Path, version: str | None = None) -> DemandForecaster:
        base = Path(model_dir)
        resolved = version or latest_version(base)
        if resolved is None:
            raise FileNotFoundError(f"no trained model under {base}")
        path = base / resolved
        obj = cls()
        obj.model = joblib.load(path / "model.joblib")
        obj.metadata = json.loads((path / "metadata.json").read_text())
        obj.version = obj.metadata["version"]
        obj.params = obj.metadata.get("params", obj.params)
        return obj


def latest_version(model_dir: str | Path) -> str | None:
    base = Path(model_dir)
    if not base.exists():
        return None
    candidates = [p for p in base.iterdir() if p.is_dir() and (p / "metadata.json").exists()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: json.loads((p / "metadata.json").read_text())["trained_at"])
    return candidates[-1].name
