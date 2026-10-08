from __future__ import annotations

import numpy as np
import pytest

from gridcast.backtest import rolling_backtest
from gridcast.data import make_demand_series
from gridcast.drift import DriftReport, FeatureDrift, detect_drift, ks_statistic, psi
from gridcast.retrain import decide
from gridcast.service import GridCastService


def test_psi_low_for_same_distribution():
    rng = np.random.default_rng(0)
    assert psi(rng.normal(size=5000), rng.normal(size=5000)) < 0.1


def test_psi_high_for_shifted_distribution():
    rng = np.random.default_rng(0)
    assert psi(rng.normal(size=5000), rng.normal(5, size=5000)) > 0.25


def test_ks_statistic_range():
    rng = np.random.default_rng(0)
    x = rng.normal(size=1000)
    assert ks_statistic(x, x) == 0.0
    assert 0.0 <= ks_statistic(x, rng.normal(3, size=1000)) <= 1.0


def test_detect_drift_flags_shift():
    df = GridCastService.drifted_series(seed=7)
    half = len(df) // 2
    report = detect_drift(df.iloc[:half], df.iloc[half:], ["temperature", "demand"])
    assert report.drifted is True


def test_backtest_runs(series):
    result = rolling_backtest(series, n_folds=2, horizon=48, params={"n_estimators": 60})
    assert len(result.folds) == 2
    assert result.mape > 0


def test_backtest_too_short_raises():
    with pytest.raises(ValueError):
        rolling_backtest(make_demand_series(days=5), n_folds=5, horizon=168)


def test_retrain_decisions():
    calm = DriftReport()
    calm.features = [FeatureDrift("temperature", 0.01, 0.01, False)]
    assert decide(0.05, calm, 0.12).should_retrain is False
    assert decide(0.20, calm, 0.12).should_retrain is True  # high error
    drifted = DriftReport()
    drifted.features = [FeatureDrift("temperature", 0.5, 0.3, True)]
    assert decide(0.05, drifted, 0.12).should_retrain is True  # drift
