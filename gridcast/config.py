"""Application configuration, loaded from environment variables (prefix ``GC_``)."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GC_", env_file=".env", extra="ignore")

    environment: str = "local"
    model_dir: str = "artifacts"
    mlflow_tracking_uri: str = "./mlruns"
    mlflow_experiment: str = "gridcast"
    drift_psi_threshold: float = 0.20
    retrain_mape_threshold: float = 0.12


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
