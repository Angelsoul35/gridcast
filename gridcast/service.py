"""Orchestration: train/register, forecast, drift check, and retrain decision."""

from __future__ import annotations

import pandas as pd

from .backtest import rolling_backtest
from .config import Settings, get_settings
from .data import make_demand_series
from .drift import detect_drift
from .features import LAGS, ROLLING
from .model import DemandForecaster, latest_version
from .retrain import decide

_MAX_LAG = max(LAGS + ROLLING)


class GridCastService:
    def __init__(
        self,
        settings: Settings | None = None,
        series: pd.DataFrame | None = None,
        train_params: dict | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.series = series if series is not None else make_demand_series(days=365, seed=42)
        self.train_params = train_params or {"n_estimators": 300}
        self.model: DemandForecaster | None = None
        self._load_or_train()

    def _load_or_train(self) -> None:
        if latest_version(self.settings.model_dir) is not None:
            self.model = DemandForecaster.load(self.settings.model_dir)
        else:
            self.model = DemandForecaster(params=self.train_params)
            self.model.train(self.series)

    def model_info(self) -> dict:
        assert self.model is not None
        return {"version": self.model.version, "metadata": self.model.metadata}

    def _future_frame(self, horizon: int) -> pd.DataFrame:
        last_ts = pd.to_datetime(self.series["timestamp"].iloc[-1])
        timestamps = pd.date_range(start=last_ts + pd.Timedelta(hours=1), periods=horizon, freq="h")
        # Naive future temperature: repeat the last week, hour by hour.
        last_week = self.series["temperature"].to_numpy()[-168:]
        temps = [last_week[i % len(last_week)] for i in range(horizon)]
        return pd.DataFrame({"timestamp": timestamps, "temperature": temps})

    def forecast(self, horizon: int = 48) -> dict:
        assert self.model is not None
        future = self._future_frame(horizon)
        fc = self.model.forecast(self.series, future)
        return {
            "horizon": horizon,
            "model_version": self.model.version,
            "forecast": [
                {"timestamp": str(t), "forecast": round(float(v), 4)}
                for t, v in zip(fc["timestamp"], fc["forecast"], strict=True)
            ],
        }

    def check_drift(self, current: pd.DataFrame | None = None) -> dict:
        reference = self.series.iloc[: len(self.series) // 2]
        current = current if current is not None else self.series.iloc[-168 * 4 :]
        report = detect_drift(
            reference,
            current,
            features=["temperature", "demand"],
            threshold=self.settings.drift_psi_threshold,
        )
        return report.as_dict()

    def evaluate(self, n_folds: int = 3, horizon: int = 168) -> dict:
        backtest = rolling_backtest(
            self.series, n_folds=n_folds, horizon=horizon, params={"n_estimators": 200}
        )
        drift_dict = self.check_drift()
        # Rebuild a DriftReport-like object for the decision from the dict.
        from .drift import DriftReport, FeatureDrift

        report = DriftReport(threshold=self.settings.drift_psi_threshold)
        report.features = [
            FeatureDrift(f["feature"], f["psi"], f["ks"], f["drifted"])
            for f in drift_dict["features"]
        ]
        decision = decide(backtest.mape, report, self.settings.retrain_mape_threshold)
        return {
            "backtest": backtest.as_dict(),
            "drift": drift_dict,
            "retrain": decision.as_dict(),
        }

    @staticmethod
    def drifted_series(seed: int = 7) -> pd.DataFrame:
        """A series whose second half is hotter — used to demonstrate drift."""
        base = make_demand_series(days=200, seed=seed)
        half = len(base) // 2
        shifted = base.copy()
        shifted.loc[half:, "temperature"] = shifted.loc[half:, "temperature"] + 10.0
        return shifted
