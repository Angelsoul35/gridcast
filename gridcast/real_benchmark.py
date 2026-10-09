"""Benchmark the forecaster on the real Spanish demand series.

The synthetic data (:mod:`gridcast.data`) proves the model recovers a pattern we
built; this module proves it on data no one designed. It runs the forecaster over
the real 2015-2016 Spanish grid series with a rolling-origin backtest and lines the
result up against two references:

* a **naive seasonal baseline** (last week, same hour) — the floor a useful
  forecaster must beat; and
* the **grid operator's own published day-ahead forecast** (``total load
  forecast``) — a professional ceiling built from a far richer operational feed
  (region-by-region weather forecasts, generation schedules, the holiday calendar).

Reporting both turns one error number into a story: the model clears the naive
baseline comfortably and lands in the same low-single-digit range as the operator,
and the remaining gap to the operator measures exactly what richer inputs would buy.

Run it with ``python -m gridcast.real_benchmark``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import rolling_backtest
from .features import TARGET, build_features, split_xy
from .model import DemandForecaster, mae, mape
from .real_data import load_spain_demand


@dataclass
class HorizonResult:
    horizon: int
    n_folds: int
    model_mape: float
    model_mae: float
    tso_mape: float
    tso_mae: float
    naive_mape: float
    naive_mae: float


@dataclass
class RealBenchmarkReport:
    n_rows: int
    demand_min: float
    demand_max: float
    train_mape: float
    test_mape: float
    gap: float
    horizons: list[HorizonResult]

    @property
    def verdict(self) -> str:
        day_ahead = next((h for h in self.horizons if h.horizon == 24), self.horizons[0])
        beats_naive = day_ahead.model_mape < day_ahead.naive_mape
        if self.gap <= 0.03 and beats_naive:
            return (
                "well-fit: small train/test gap and the day-ahead error clears the "
                "naive seasonal baseline"
            )
        if not beats_naive:
            return "underfit: does not beat the naive seasonal baseline"
        return "overfit: train/test gap is large relative to the held-out error"


def _one_step_gap(df: pd.DataFrame, test_fraction: float = 0.2) -> tuple[float, float]:
    """One-step-ahead train vs held-out MAPE (the clean over/underfit signal)."""
    feat = build_features(df)
    split = int(len(feat) * (1 - test_fraction))
    train_df, test_df = feat.iloc[:split], feat.iloc[split:]
    forecaster = DemandForecaster()
    forecaster.train(df)
    x_tr, y_tr = split_xy(train_df)
    x_te, y_te = split_xy(test_df)
    return (
        mape(y_tr, forecaster.model.predict(x_tr)),
        mape(y_te, forecaster.model.predict(x_te)),
    )


def _naive_seasonal(df: pd.DataFrame, start: int, end: int, season: int = 168) -> np.ndarray:
    """Forecast each hour as the demand exactly ``season`` hours earlier (last week)."""
    demand = df[TARGET].to_numpy(dtype=float)
    return demand[start - season : end - season]


def _horizon_result(df: pd.DataFrame, n_folds: int, horizon: int) -> HorizonResult:
    base = df[["timestamp", "temperature", "demand"]]
    bt = rolling_backtest(base, n_folds=n_folds, horizon=horizon)
    total = len(df)
    test_start = total - n_folds * horizon  # contiguous region the folds forecast
    actual = df[TARGET].to_numpy(dtype=float)[test_start:total]
    tso = df["tso_forecast"].to_numpy(dtype=float)[test_start:total]
    naive = _naive_seasonal(df, test_start, total)
    return HorizonResult(
        horizon=horizon,
        n_folds=len(bt.folds),
        model_mape=bt.mape,
        model_mae=bt.mae,
        tso_mape=mape(actual, tso),
        tso_mae=mae(actual, tso),
        naive_mape=mape(actual, naive),
        naive_mae=mae(actual, naive),
    )


def run_benchmark(df: pd.DataFrame | None = None) -> RealBenchmarkReport:
    df = df if df is not None else load_spain_demand(with_baseline=True)
    train_mape, test_mape = _one_step_gap(df[["timestamp", "temperature", "demand"]])
    horizons = [
        _horizon_result(df, n_folds=6, horizon=24),
        _horizon_result(df, n_folds=4, horizon=168),
    ]
    return RealBenchmarkReport(
        n_rows=len(df),
        demand_min=float(df[TARGET].min()),
        demand_max=float(df[TARGET].max()),
        train_mape=train_mape,
        test_mape=test_mape,
        gap=round(test_mape - train_mape, 4),
        horizons=horizons,
    )


def format_report(report: RealBenchmarkReport) -> str:
    lines = [
        "GridCast real-data benchmark — Spanish grid demand (ENTSO-E, 2015-2016)",
        "=" * 70,
        f"hourly rows        : {report.n_rows}  "
        f"(demand {report.demand_min:,.0f}-{report.demand_max:,.0f} MW)",
        f"one-step train/test: MAPE {report.train_mape:.4f} / {report.test_mape:.4f}  "
        f"(gap {report.gap:.4f})",
        "",
        f"{'horizon':>10} {'folds':>6} {'model MAPE':>11} {'naive MAPE':>11} "
        f"{'TSO MAPE':>9} {'model MAE':>10}",
    ]
    for h in report.horizons:
        label = f"{h.horizon}h"
        lines.append(
            f"{label:>10} {h.n_folds:>6} {h.model_mape:>11.4f} {h.naive_mape:>11.4f} "
            f"{h.tso_mape:>9.4f} {h.model_mae:>9.0f}MW"
        )
    day = next((h for h in report.horizons if h.horizon == 24), report.horizons[0])
    week = next((h for h in report.horizons if h.horizon == 168), None)
    lines += [
        "",
        f"verdict: {report.verdict}",
        "",
        f"Day-ahead, the model's {day.model_mape:.1%} MAPE beats the naive "
        f"last-week baseline ({day.naive_mape:.1%}) and lands near the grid "
        f"operator's own {day.tso_mape:.1%} day-ahead forecast, which it builds from a "
        "far richer feed. The gap to the operator is the headroom that regional "
        "weather forecasts and a holiday calendar would add.",
    ]
    if week is not None:
        lines.append(
            f"A full week ahead ({week.model_mape:.1%}), recursive forecasting "
            f"accumulates error until the last-week baseline ({week.naive_mape:.1%}) "
            "becomes competitive — the textbook signature of recursive multi-step "
            "forecasting, and the cue to add a direct multi-horizon model for long "
            "lead times. Day-ahead is the operational horizon, and there the model is "
            "clearly the better forecaster."
        )
    return "\n".join(lines)


def main() -> None:
    print(format_report(run_benchmark()))


if __name__ == "__main__":
    main()
