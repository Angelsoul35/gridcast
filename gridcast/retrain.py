"""When to retrain the model.

Two signals trigger a retrain: the backtest error is too high, or the input data
has drifted. Either one means the current model may no longer fit the data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .drift import DriftReport


@dataclass
class RetrainDecision:
    should_retrain: bool
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"should_retrain": self.should_retrain, "reasons": self.reasons}


def decide(
    backtest_mape: float,
    drift: DriftReport,
    mape_threshold: float = 0.12,
) -> RetrainDecision:
    reasons: list[str] = []
    if backtest_mape > mape_threshold:
        reasons.append(f"backtest MAPE {backtest_mape:.3f} > threshold {mape_threshold:.3f}")
    if drift.drifted:
        drifted = [f.feature for f in drift.features if f.drifted]
        reasons.append(f"input drift in: {', '.join(drifted)}")
    return RetrainDecision(should_retrain=bool(reasons), reasons=reasons)
