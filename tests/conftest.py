from __future__ import annotations

import pytest

from gridcast.config import Settings
from gridcast.data import make_demand_series


@pytest.fixture(scope="session")
def series():
    return make_demand_series(days=120, seed=1)


@pytest.fixture
def tmp_settings(tmp_path):
    return Settings(
        model_dir=str(tmp_path / "artifacts"),
        mlflow_tracking_uri=str(tmp_path / "mlruns"),
    )
