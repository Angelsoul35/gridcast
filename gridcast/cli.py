"""GridCast command-line interface."""

from __future__ import annotations

import argparse
import json

from .backtest import rolling_backtest
from .config import get_settings
from .data import make_demand_series
from .drift import detect_drift
from .model import DemandForecaster
from .registry import ModelRegistry
from .retrain import decide
from .service import GridCastService


def cmd_train(args) -> None:
    settings = get_settings()
    df = make_demand_series(days=args.days, seed=args.seed)
    model = DemandForecaster()
    report = model.train(df)
    path = model.save(settings.model_dir)
    print(f"version {report.version}  metrics {report.metrics}  -> {path}")
    try:
        registry = ModelRegistry(settings.mlflow_tracking_uri, settings.mlflow_experiment)
        run_id = registry.log_training(report, path)
        print(f"logged to MLflow run {run_id}")
    except Exception as exc:  # pragma: no cover - MLflow optional at runtime
        print(f"MLflow logging skipped: {exc}")


def cmd_backtest(args) -> None:
    df = make_demand_series(days=args.days, seed=args.seed)
    result = rolling_backtest(
        df, n_folds=args.folds, horizon=args.horizon, params={"n_estimators": 200}
    )
    print(json.dumps(result.as_dict(), indent=2))


def cmd_drift(args) -> None:
    df = GridCastService.drifted_series(seed=args.seed)
    half = len(df) // 2
    report = detect_drift(df.iloc[:half], df.iloc[half:], ["temperature", "demand"])
    print(json.dumps(report.as_dict(), indent=2))


def cmd_demo(args) -> None:
    print("1) Training on synthetic demand ...")
    df = make_demand_series(days=args.days, seed=42)
    model = DemandForecaster(params={"n_estimators": 200})
    report = model.train(df)
    print(f"   version {report.version}  holdout metrics {report.metrics}")

    print("2) Backtesting (rolling origin) ...")
    bt = rolling_backtest(df, n_folds=3, horizon=168, params={"n_estimators": 200})
    print(f"   backtest MAPE {bt.mape:.4f} over {len(bt.folds)} folds")

    print("3) Drift check (hotter second half) ...")
    drifted = GridCastService.drifted_series()
    half = len(drifted) // 2
    drift_report = detect_drift(drifted.iloc[:half], drifted.iloc[half:], ["temperature", "demand"])
    print(f"   drifted={drift_report.drifted}  temperature PSI={drift_report.features[0].psi:.3f}")

    print("4) Retrain decision ...")
    decision = decide(bt.mape, drift_report, get_settings().retrain_mape_threshold)
    print(f"   should_retrain={decision.should_retrain}  reasons={decision.reasons}")


def main() -> None:
    parser = argparse.ArgumentParser(description="GridCast CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="train and register a model")
    p.add_argument("--days", type=int, default=730)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("backtest", help="rolling-origin backtest")
    p.add_argument("--days", type=int, default=730)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--horizon", type=int, default=168)
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("drift", help="drift report on a drifted series")
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(func=cmd_drift)

    p = sub.add_parser("demo", help="train, backtest, drift, retrain decision")
    p.add_argument("--days", type=int, default=400)
    p.set_defaults(func=cmd_demo)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
