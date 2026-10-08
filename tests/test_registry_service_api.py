from __future__ import annotations

from fastapi.testclient import TestClient

from gridcast.model import DemandForecaster
from gridcast.registry import ModelRegistry
from gridcast.service import GridCastService


def _service(series, tmp_settings):
    return GridCastService(
        settings=tmp_settings, series=series, train_params={"n_estimators": 60}
    )


def test_registry_logs_and_lists(series, tmp_path):
    model = DemandForecaster(params={"n_estimators": 60})
    report = model.train(series)
    path = model.save(tmp_path / "art")
    registry = ModelRegistry(tracking_uri=str(tmp_path / "mlruns"), experiment="test-exp")
    run_id = registry.log_training(report, path)
    assert run_id
    history = registry.history()
    assert any(h["version"] == report.version for h in history)


def test_service_forecast_and_drift(series, tmp_settings):
    service = _service(series, tmp_settings)
    result = service.forecast(24)
    assert len(result["forecast"]) == 24
    assert service.check_drift()["features"]


def test_service_evaluate(series, tmp_settings):
    service = _service(series, tmp_settings)
    evaluation = service.evaluate(n_folds=2, horizon=48)
    assert "retrain" in evaluation
    assert "backtest" in evaluation


def test_api(series, tmp_settings):
    import gridcast.main as main

    main._service = _service(series, tmp_settings)
    client = TestClient(main.app)

    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/model").json()["version"].startswith("m_")

    response = client.post("/forecast", json={"horizon": 12})
    assert response.status_code == 200
    assert len(response.json()["forecast"]) == 12

    assert "features" in client.get("/drift").json()
    assert client.post("/forecast", json={"horizon": 999}).status_code == 422
