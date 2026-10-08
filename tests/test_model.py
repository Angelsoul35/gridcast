from __future__ import annotations

import pytest

from gridcast.model import DemandForecaster, latest_version, mae, mape, rmse


def test_training_learns_signal(series):
    model = DemandForecaster(params={"n_estimators": 80})
    report = model.train(series)
    assert report.version.startswith("m_")
    assert report.metrics["mape"] < 0.12  # a decent forecast on the holdout


def test_forecast_length_and_values(series):
    model = DemandForecaster(params={"n_estimators": 80})
    model.train(series)
    history = series.iloc[:-24]
    future = series.iloc[-24:][["timestamp", "temperature"]].copy()
    fc = model.forecast(history, future)
    assert len(fc) == 24
    assert (fc["forecast"] > 0).all()


def test_save_and_load(series, tmp_path):
    model = DemandForecaster(params={"n_estimators": 60})
    model.train(series)
    model.save(tmp_path)
    reloaded = DemandForecaster.load(tmp_path, model.version)
    assert reloaded.version == model.version
    assert latest_version(tmp_path) == model.version


def test_metric_helpers():
    assert mae([1, 2, 3], [1, 2, 3]) == 0.0
    assert rmse([0, 0], [3, 4]) == pytest.approx(3.5355, rel=1e-3)
    # mape ignores zero-valued true points
    assert mape([0, 100], [50, 110]) == pytest.approx(0.1, rel=1e-6)
