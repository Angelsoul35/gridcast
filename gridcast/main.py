"""GridCast HTTP API."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .service import GridCastService

app = FastAPI(
    title="GridCast",
    version="1.0.0",
    summary="Electricity demand forecasting with backtesting, drift detection, and MLOps.",
)

# Lazily built on first request so importing the app does not train a model.
_service: GridCastService | None = None


def get_service() -> GridCastService:
    global _service
    if _service is None:
        _service = GridCastService()
    return _service


class ForecastRequest(BaseModel):
    horizon: int = Field(default=48, ge=1, le=168, description="hours ahead to forecast")


@app.get("/health", tags=["ops"])
def health() -> dict:
    return {"status": "ok"}


@app.get("/model", tags=["model"])
def model_info() -> dict:
    return get_service().model_info()


@app.post("/forecast", tags=["forecast"])
def forecast(request: ForecastRequest) -> dict:
    try:
        return get_service().forecast(request.horizon)
    except RuntimeError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/drift", tags=["monitoring"])
def drift() -> dict:
    return get_service().check_drift()
