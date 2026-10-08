"""Synthetic electricity-demand time series.

The series is hourly and fully synthetic, but it is built to look like real grid
demand: a daily shape (low at night, peaks in the morning and evening), a weekly
shape (weekends lower), a yearly shape, a temperature effect (demand rises when
it is very cold or very hot), a slow upward trend, and noise. Because we build it
from known parts, we can check that the model recovers the real pattern and we
can inject drift on purpose to test the drift detector.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _temperature(index: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    hours = np.arange(len(index))
    day_of_year = index.dayofyear.to_numpy()
    hour = index.hour.to_numpy()
    # Yearly season (warmer mid-year) plus a daily swing plus noise.
    yearly = 8.0 * np.sin(2 * np.pi * (day_of_year - 100) / 365.0)
    daily = 4.0 * np.sin(2 * np.pi * (hour - 15) / 24.0)
    base = 18.0
    noise = rng.normal(0, 1.5, size=len(index))
    return base + yearly + daily + noise + 0.0 * hours


def make_demand_series(
    days: int = 730,
    start: str = "2023-01-01",
    seed: int = 42,
    demand_scale: float = 1.0,
) -> pd.DataFrame:
    """Return an hourly DataFrame with columns: timestamp, temperature, demand (MW)."""
    rng = np.random.default_rng(seed)
    periods = days * 24
    index = pd.date_range(start=start, periods=periods, freq="h")

    hour = index.hour.to_numpy()
    dow = index.dayofweek.to_numpy()
    day_of_year = index.dayofyear.to_numpy()

    temperature = _temperature(index, rng)

    # Daily demand shape: two peaks (morning, evening), low overnight.
    daily_shape = 0.6 * np.sin(2 * np.pi * (hour - 8) / 24.0) + 0.4 * np.sin(
        2 * np.pi * (hour - 19) / 12.0
    )
    weekly = np.where(dow >= 5, -0.25, 0.0)  # weekends lower
    yearly = 0.15 * np.sin(2 * np.pi * (day_of_year - 15) / 365.0)

    # Temperature effect: U-shape around a comfortable ~20C.
    temp_effect = 0.02 * (temperature - 20.0) ** 2

    trend = np.linspace(0, 0.1, periods)  # slow growth
    base_load = 5.0

    demand = (
        base_load * (1.0 + 0.35 * daily_shape + weekly + yearly + trend)
        + temp_effect
        + rng.normal(0, 0.15, size=periods)
    )
    demand = np.clip(demand, 0.5, None) * demand_scale

    return pd.DataFrame(
        {
            "timestamp": index,
            "temperature": np.round(temperature, 3),
            "demand": np.round(demand, 4),
        }
    )
