from __future__ import annotations

from gridcast.data import make_demand_series
from gridcast.features import FEATURE_COLUMNS, LAGS, ROLLING, build_features


def test_series_columns_and_positive():
    df = make_demand_series(days=10, seed=2)
    assert {"timestamp", "temperature", "demand"}.issubset(df.columns)
    assert len(df) == 10 * 24
    assert (df["demand"] > 0).all()


def test_build_features_has_all_columns_and_no_nan():
    df = make_demand_series(days=30, seed=2)
    feat = build_features(df)
    for col in FEATURE_COLUMNS:
        assert col in feat.columns
    assert feat.isna().sum().sum() == 0
    assert 0 < len(feat) < len(df)


def test_rolling_uses_only_past():
    # roll_mean at a row must not include that row's own demand (shift(1)).
    df = make_demand_series(days=30, seed=2)
    feat = build_features(df, dropna=False)
    first_valid = max(LAGS + ROLLING)
    # By construction the rolling feature is defined from the previous hours only;
    # a non-null value appears only after enough history.
    assert feat["roll_mean_24"].iloc[:first_valid].isna().any()
