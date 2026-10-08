"""Rolling-origin backtesting.

A single train/test split can be lucky. A backtest instead trains several times,
each time on data up to a cut point, then forecasts the next window and measures
the error. Rolling the cut point forward gives a fairer picture of how the model
does over time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .features import TARGET
from .model import DemandForecaster, mae, mape, rmse


@dataclass
class BacktestResult:
    folds: list[dict] = field(default_factory=list)
    mae: float = 0.0
    rmse: float = 0.0
    mape: float = 0.0

    def as_dict(self) -> dict:
        return {
            "n_folds": len(self.folds),
            "mae": round(self.mae, 4),
            "rmse": round(self.rmse, 4),
            "mape": round(self.mape, 4),
            "folds": self.folds,
        }


def rolling_backtest(
    df: pd.DataFrame,
    n_folds: int = 3,
    horizon: int = 168,
    params: dict[str, Any] | None = None,
) -> BacktestResult:
    """Train on a growing history, forecast ``horizon`` hours, repeated ``n_folds`` times."""
    total = len(df)
    min_train = total - n_folds * horizon
    if min_train <= max(336, horizon):
        raise ValueError("series too short for this many folds and horizon")

    result = BacktestResult()
    maes, rmses, mapes = [], [], []
    for fold in range(n_folds):
        cut = min_train + fold * horizon
        history = df.iloc[:cut].copy()
        future = df.iloc[cut : cut + horizon]
        if len(future) < horizon:
            break
        forecaster = DemandForecaster(params)
        forecaster.train(history)
        fc = forecaster.forecast(history, future[["timestamp", "temperature"]].copy())
        actual = future[TARGET].to_numpy()
        pred = fc["forecast"].to_numpy()
        f_mae, f_rmse, f_mape = mae(actual, pred), rmse(actual, pred), mape(actual, pred)
        maes.append(f_mae)
        rmses.append(f_rmse)
        mapes.append(f_mape)
        result.folds.append(
            {"fold": fold, "cut": str(df.iloc[cut]["timestamp"]),
             "mae": round(f_mae, 4), "rmse": round(f_rmse, 4), "mape": round(f_mape, 4)}
        )

    result.mae = float(np.mean(maes)) if maes else 0.0
    result.rmse = float(np.mean(rmses)) if rmses else 0.0
    result.mape = float(np.mean(mapes)) if mapes else 0.0
    return result
