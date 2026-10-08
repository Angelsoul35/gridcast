"""A thin MLflow wrapper for experiment tracking and the model registry.

Every training run logs its parameters, its metrics, and the saved model as an
artifact to MLflow. With a local file store (the default) this works offline and
needs no server; in production you point the tracking URI at a real MLflow
server. Tracking runs is what lets you compare models over time and roll back.
"""

from __future__ import annotations

import os
from pathlib import Path

import mlflow

from .model import TrainingReport

# The local file store is in maintenance mode in recent MLflow but is perfect for
# dev and CI. Opt in here; production would point the tracking URI at a server.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")


class ModelRegistry:
    def __init__(self, tracking_uri: str = "./mlruns", experiment: str = "gridcast") -> None:
        self.tracking_uri = tracking_uri
        self.experiment = experiment
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)

    def log_training(self, report: TrainingReport, model_dir: str | Path) -> str:
        """Log one training run and return its run id."""
        with mlflow.start_run() as run:
            flat_params = {
                k: v for k, v in report.params.items() if isinstance(v, (int, float, str))
            }
            mlflow.log_params(flat_params)
            mlflow.log_metrics(report.metrics)
            mlflow.set_tag("model_version", report.version)
            mlflow.set_tag("n_rows", report.n_rows)
            mlflow.log_artifacts(str(model_dir), artifact_path="model")
            return run.info.run_id

    def history(self) -> list[dict]:
        """Return past runs with their version and metrics, newest first."""
        runs = mlflow.search_runs(experiment_names=[self.experiment])
        if runs is None or len(runs) == 0:
            return []
        out: list[dict] = []
        for _, row in runs.iterrows():
            out.append(
                {
                    "run_id": row.get("run_id"),
                    "version": row.get("tags.model_version"),
                    "mae": row.get("metrics.mae"),
                    "rmse": row.get("metrics.rmse"),
                    "mape": row.get("metrics.mape"),
                }
            )
        return out
