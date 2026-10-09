"""Real Spanish grid demand, for validating the forecaster on data it did not shape.

The default demand series (:mod:`gridcast.data`) is synthetic — useful because every
component (daily shape, temperature effect, drift) is known, so a test can check that
the model recovers it and drift can be injected on purpose. This module adds the other
half: **real** hourly electricity demand for the Spanish grid, so the model is also
measured on data no one designed.

Source: the ENTSO-E transparency figures for Spain, via the Hugging Face mirror
``vitaliy-sharandin/energy-consumption-hourly-spain`` (hourly ``total load actual`` in
MW and the TSO's own day-ahead ``total load forecast``), joined on the timestamp with
Madrid temperature from ``vitaliy-sharandin/energy-consumption-weather-hourly-spain``
(converted from Kelvin to Celsius). A 2015–2016 slice is vendored under
``datasets/spain_demand_real.csv``; ``fetch_spain_demand`` refreshes it.

The frame has exactly the columns the rest of GridCast expects — ``timestamp``,
``temperature``, ``demand`` — plus ``tso_forecast``, the grid operator's own published
day-ahead forecast, which the real benchmark uses as a professional baseline.
"""

from __future__ import annotations

import csv
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATASET_PATH = _REPO_ROOT / "datasets" / "spain_demand_real.csv"

_ENERGY_DS = "vitaliy-sharandin%2Fenergy-consumption-hourly-spain"
_WEATHER_DS = "vitaliy-sharandin%2Fenergy-consumption-weather-hourly-spain"
_ROWS_URL = (
    "https://datasets-server.huggingface.co/rows?dataset={ds}"
    "&config=default&split=train&offset={off}&length=100"
)


def load_spain_demand(with_baseline: bool = False) -> pd.DataFrame:
    """Load the vendored real Spanish demand series.

    Returns a frame with ``timestamp`` (datetime), ``temperature`` (Celsius) and
    ``demand`` (MW); when ``with_baseline`` is set, also ``tso_forecast`` (MW), the
    grid operator's published day-ahead forecast.
    """
    if not _DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Vendored Spain demand not found at {_DATASET_PATH}. Run "
            "`python -c 'from gridcast.real_data import fetch_spain_demand; fetch_spain_demand()'`."
        )
    df = pd.read_csv(_DATASET_PATH)
    # The stored timestamps carry the local UTC offset, which flips with daylight
    # saving (+01:00 winter, +02:00 summer). Demand patterns follow the local
    # wall-clock (when people wake, cook, sleep), so parse the local time directly
    # (the first 19 chars) as a naive datetime — the hour-of-day the model needs.
    df["timestamp"] = pd.to_datetime(df["timestamp"].str.slice(0, 19))
    cols = ["timestamp", "temperature", "demand"]
    if with_baseline:
        cols.append("tso_forecast")
    return df[cols].reset_index(drop=True)


def _get(ds: str, off: int, tries: int = 5) -> dict:
    url = _ROWS_URL.format(ds=ds, off=off)
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return json.load(resp)
        except Exception:  # noqa: BLE001 - network retry
            if i == tries - 1:
                raise
            time.sleep(2**i)
    return {}


def fetch_spain_demand(
    start: str = "2015-01-01",
    end_hour_prefix: str = "2016-12-31 23",
    dest: str | Path | None = None,
) -> int:
    """Download and join real demand + Madrid temperature into the vendored CSV.

    Pages the Hugging Face datasets-server. ``demand`` is ``total load actual`` (MW),
    ``tso_forecast`` is the day-ahead ``total load forecast`` (MW), and ``temperature``
    is Madrid's, converted to Celsius. Returns the number of joined hourly rows.
    """
    target = Path(dest) if dest is not None else _DATASET_PATH

    def in_window(ts: str) -> bool:
        return ts >= start and ts[:13] <= end_hour_prefix

    # Demand + the TSO's own day-ahead forecast (chronological from the start).
    demand: dict[str, tuple[float, float | None]] = {}
    off = 0
    while off <= 20000:
        rows = _get(_ENERGY_DS, off).get("rows", [])
        if not rows:
            break
        last_ts = None
        for row in rows:
            r = row["row"]
            ts = r["time"]
            last_ts = ts
            la = r.get("total load actual")
            if not in_window(ts) or la is None:
                continue
            lf = r.get("total load forecast")
            demand[ts] = (float(la), None if lf is None else float(lf))
        off += 100
        if last_ts and last_ts[:13] > end_hour_prefix:
            break

    # Madrid temperature (Kelvin -> Celsius), scanned from the Madrid block.
    temp: dict[str, float] = {}
    off = 35100
    seen_madrid = False
    while off <= 70200:
        rows = _get(_WEATHER_DS, off).get("rows", [])
        if not rows:
            break
        last_city_ts = None
        for row in rows:
            r = row["row"]
            if (r.get("city_name") or "").strip() != "Madrid":
                continue
            seen_madrid = True
            ts = r["dt_iso"]
            last_city_ts = ts
            if in_window(ts):
                temp[ts] = round(float(r["temp"]) - 273.15, 2)
        off += 100
        if seen_madrid and last_city_ts and last_city_ts[:13] > end_hour_prefix:
            break

    joined = [
        (ts, temp[ts], round(demand[ts][0], 2), demand[ts][1])
        for ts in sorted(demand)
        if ts in temp
    ]
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "temperature", "demand", "tso_forecast"])
        for ts, t, la, lf in joined:
            w.writerow([ts, t, la, "" if lf is None else round(lf, 2)])
    return len(joined)
