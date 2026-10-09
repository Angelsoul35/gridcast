"""Functional tests for the real Spanish-demand benchmark.

They prove the vendored data is the real series, that the forecaster is well-fit on
it (small train/test gap, day-ahead error below the naive seasonal baseline and near
the grid operator's own forecast), that the signal is real (shuffling the demand
destroys the model's edge), and that the fetch/join path works.
"""

from __future__ import annotations

import numpy as np
import pytest

from gridcast.backtest import rolling_backtest
from gridcast.real_benchmark import format_report, run_benchmark
from gridcast.real_data import load_spain_demand


@pytest.fixture(scope="module")
def df():
    return load_spain_demand(with_baseline=True)


@pytest.fixture(scope="module")
def report(df):
    return run_benchmark(df)


def test_loads_the_real_series(df):
    assert len(df) > 17000  # ~2 years of hourly readings
    assert list(df.columns) == ["timestamp", "temperature", "demand", "tso_forecast"]
    # Real Spanish grid demand sits in the tens of GW; Madrid temperature is sane.
    assert 5_000 < df["demand"].min() and df["demand"].max() < 60_000
    assert -20 < df["temperature"].min() and df["temperature"].max() < 50
    # Hourly cadence: all but a handful of steps are exactly one hour. The few
    # exceptions are the two annual daylight-saving transitions (local civil time
    # jumps forward/repeats) plus a couple of missing hours in the ENTSO-E source.
    gaps = df["timestamp"].diff().dropna().dt.total_seconds()
    assert (gaps == 3600.0).mean() > 0.99
    assert df["timestamp"].is_monotonic_increasing or (gaps == 0).sum() <= 2


def test_load_without_baseline_drops_the_tso_column():
    base = load_spain_demand()
    assert list(base.columns) == ["timestamp", "temperature", "demand"]


def test_forecaster_is_well_fit_on_real_demand(report):
    # Small one-step train/test gap: regularised, not pathologically overfit.
    assert report.train_mape < report.test_mape  # some generalisation cost
    assert report.gap <= 0.03, report
    assert report.test_mape < 0.06, report
    assert report.verdict.startswith("well-fit"), report.verdict


def test_day_ahead_beats_naive_and_trails_the_operator(report):
    day = next(h for h in report.horizons if h.horizon == 24)
    # Clears the naive last-week baseline comfortably...
    assert day.model_mape < day.naive_mape
    assert day.model_mape < 0.08
    # ...and the operator's richer-feed forecast is the professional reference below it.
    assert day.tso_mape < day.model_mape
    assert day.tso_mape < 0.03


def test_week_ahead_is_reported_and_harder(report):
    week = next(h for h in report.horizons if h.horizon == 168)
    day = next(h for h in report.horizons if h.horizon == 24)
    # Recursive error accumulates: a week out is harder than a day out.
    assert week.model_mape > day.model_mape


def test_shuffling_demand_destroys_the_models_edge(df):
    """The model learns real temporal structure: once the demand column is shuffled
    (lags and seasonality gone), the same backtest error inflates sharply."""
    sl = df.iloc[:2000].reset_index(drop=True)
    horizon, folds = 24, 3
    real_bt = rolling_backtest(sl[["timestamp", "temperature", "demand"]], folds, horizon)

    rng = np.random.default_rng(0)
    shuffled = sl.copy()
    shuffled["demand"] = rng.permutation(shuffled["demand"].to_numpy())
    shuf_bt = rolling_backtest(
        shuffled[["timestamp", "temperature", "demand"]], folds, horizon
    )

    assert shuf_bt.mape > real_bt.mape * 1.5  # destroying the structure wrecks it


def test_format_report_renders_the_references(report):
    text = format_report(report)
    assert "Spanish grid demand" in text
    assert "naive" in text and "operator" in text
    assert f"{report.test_mape:.4f}" in text


def test_main_prints_the_report(capsys):
    from gridcast.real_benchmark import main

    main()
    assert "real-data benchmark" in capsys.readouterr().out


def test_loader_rejects_a_missing_dataset(monkeypatch, tmp_path):
    import gridcast.real_data as rd

    monkeypatch.setattr(rd, "_DATASET_PATH", tmp_path / "missing.csv")
    with pytest.raises(FileNotFoundError):
        rd.load_spain_demand()


def test_fetch_joins_demand_and_temperature(tmp_path, monkeypatch):
    import gridcast.real_data as rd

    energy_rows = [
        {"row": {"time": "2015-01-01 00:00:00+01:00",
                 "total load actual": 25000.0, "total load forecast": 25500.0}},
        {"row": {"time": "2015-01-01 01:00:00+01:00",
                 "total load actual": 24000.0, "total load forecast": 24200.0}},
        # Dropped: no actual reading.
        {"row": {"time": "2015-01-01 02:00:00+01:00",
                 "total load actual": None, "total load forecast": 23000.0}},
    ]
    weather_rows = [
        {"row": {"city_name": "Valencia", "dt_iso": "2015-01-01 00:00:00+01:00",
                 "temp": 300.0}},  # wrong city, ignored
        {"row": {"city_name": "Madrid", "dt_iso": "2015-01-01 00:00:00+01:00",
                 "temp": 283.15}},  # 10 C
        {"row": {"city_name": "Madrid", "dt_iso": "2015-01-01 01:00:00+01:00",
                 "temp": 278.15}},  # 5 C
    ]
    calls = {"energy": 0, "weather": 0}

    def fake_get(ds, off, tries=5):
        if ds == rd._ENERGY_DS:
            calls["energy"] += 1
            return {"rows": energy_rows if calls["energy"] == 1 else []}
        calls["weather"] += 1
        return {"rows": weather_rows if calls["weather"] == 1 else []}

    monkeypatch.setattr(rd, "_get", fake_get)
    dest = tmp_path / "spain.csv"
    n = rd.fetch_spain_demand(end_hour_prefix="2015-01-01 23", dest=dest)
    assert n == 2  # two joined hours (the None-actual row is dropped)

    monkeypatch.setattr(rd, "_DATASET_PATH", dest)
    out = rd.load_spain_demand(with_baseline=True)
    assert out["demand"].tolist() == [25000.0, 24000.0]
    assert out["temperature"].tolist() == [10.0, 5.0]
    assert out["tso_forecast"].tolist() == [25500.0, 24200.0]
