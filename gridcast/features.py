"""Turn the raw demand series into model features.

Forecasting demand is a tabular regression once you build the right features:
calendar parts (hour, day of week, month), the temperature, and past demand as
lags and rolling means. The model learns the daily and weekly shape from the
calendar parts and the recent level from the lags.
"""

from __future__ import annotations

import pandas as pd

TARGET = "demand"
LAGS = [24, 48, 168]  # 1 day, 2 days, 1 week
ROLLING = [24, 168]  # 1-day and 1-week moving averages

CALENDAR_FEATURES = ["hour", "dayofweek", "month", "is_weekend"]
EXOG_FEATURES = ["temperature"]
LAG_FEATURES = [f"lag_{lag}" for lag in LAGS]
ROLL_FEATURES = [f"roll_mean_{w}" for w in ROLLING]

FEATURE_COLUMNS = CALENDAR_FEATURES + EXOG_FEATURES + LAG_FEATURES + ROLL_FEATURES


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df["timestamp"])
    df = df.copy()
    df["hour"] = ts.dt.hour
    df["dayofweek"] = ts.dt.dayofweek
    df["month"] = ts.dt.month
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    return df


def build_features(df: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
    """Return the DataFrame with calendar, lag, and rolling features added."""
    out = add_calendar(df)
    for lag in LAGS:
        out[f"lag_{lag}"] = out[TARGET].shift(lag)
    for window in ROLLING:
        # shift(1) so the rolling mean uses only the past, not the current hour.
        out[f"roll_mean_{window}"] = out[TARGET].shift(1).rolling(window).mean()
    if dropna:
        out = out.dropna().reset_index(drop=True)
    return out


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURE_COLUMNS], df[TARGET]
