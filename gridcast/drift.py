"""Data-drift detection with the Population Stability Index (PSI).

A model is trained on past data. If the live data moves away from that past
(hotter weather, a new demand pattern), the model quietly gets worse. Drift
detection watches for that move, so you can retrain before quality drops.

PSI compares the distribution of a feature in a reference window against a
current window. Rule of thumb: < 0.1 no real change, 0.1-0.25 some change,
> 0.25 a big change.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two samples of one feature."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(expected, quantiles))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    eps = 1e-6
    e_perc = np.histogram(expected, bins=edges)[0] / len(expected)
    a_perc = np.histogram(actual, bins=edges)[0] / len(actual)
    e_perc = np.clip(e_perc, eps, None)
    a_perc = np.clip(a_perc, eps, None)
    return float(np.sum((a_perc - e_perc) * np.log(a_perc / e_perc)))


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Kolmogorov-Smirnov distance between two empirical distributions."""
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    if len(a) == 0 or len(b) == 0:
        return 0.0
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / len(a)
    cdf_b = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


@dataclass
class FeatureDrift:
    feature: str
    psi: float
    ks: float
    drifted: bool


@dataclass
class DriftReport:
    features: list[FeatureDrift] = field(default_factory=list)
    threshold: float = 0.20

    @property
    def drifted(self) -> bool:
        return any(f.drifted for f in self.features)

    def as_dict(self) -> dict:
        return {
            "drifted": self.drifted,
            "threshold": self.threshold,
            "features": [
                {
                    "feature": f.feature,
                    "psi": round(f.psi, 4),
                    "ks": round(f.ks, 4),
                    "drifted": f.drifted,
                }
                for f in self.features
            ],
        }


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str],
    threshold: float = 0.20,
) -> DriftReport:
    report = DriftReport(threshold=threshold)
    for feature in features:
        p = psi(reference[feature].to_numpy(), current[feature].to_numpy())
        k = ks_statistic(reference[feature].to_numpy(), current[feature].to_numpy())
        report.features.append(FeatureDrift(feature=feature, psi=p, ks=k, drifted=p > threshold))
    return report
