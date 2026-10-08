# Architecture

## Goal

Forecast hourly electricity demand, and keep the model good over time with backtesting,
drift detection, experiment tracking, and a retrain decision.

## Components

| Layer | Module | What it does |
|---|---|---|
| Data | `data.py` | Synthetic hourly demand + temperature |
| Features | `features.py` | Calendar, lag, and rolling features |
| Model | `model.py` | LightGBM forecaster + recursive multi-step forecast |
| Backtest | `backtest.py` | Rolling-origin backtest |
| Drift | `drift.py` | PSI and KS drift detection |
| Registry | `registry.py` | MLflow experiment tracking and registry |
| Retrain | `retrain.py` | The retrain decision |
| Flow | `service.py` | Ties it together |
| API | `main.py` | FastAPI endpoints |

## Features and leakage

Forecasting demand is tabular regression once the features are built. The important rule
is **no leakage**: a feature for hour *t* may only use data from before *t*. The lag
features (`lag_24`, `lag_48`, `lag_168`) use past demand. The rolling means use
`shift(1)` before the window, so the current hour's own demand is never in its features.

## Recursive forecasting

To forecast one hour we just build the features and predict. To forecast many hours we go
step by step: predict hour *t+1*, append it to the working series, then build the lags for
*t+2* from that series, and so on. This is the recursive strategy. It needs the future
temperature; in the demo and API we use a simple "same hour last week" value, and in a
real system it would come from a weather forecast.

## Backtesting

A single train/test split can be lucky. `backtest.py` does rolling-origin backtesting:
train on data up to a cut point, forecast the next window, measure the error, then move
the cut point forward and repeat. The average error over the folds is a fairer estimate.

## Drift detection

`drift.py` compares the distribution of each watched feature (temperature, demand) in a
reference window against the current window, using the **Population Stability Index (PSI)**
and the Kolmogorov-Smirnov distance. A PSI above the threshold flags drift. This catches
the case where the live data has moved away from what the model was trained on.

## The retrain decision

`retrain.py` turns two signals into one yes/no: retrain if the backtest error is above a
threshold, or if drift is detected. This is the policy that closes the MLOps loop.

## Experiment tracking

`registry.py` logs every training run to MLflow: parameters, metrics, and the saved model
as an artifact, tagged with the content-based model version. This is what lets you compare
models over time and roll back to a known-good one. The default is a local file store; in
production the tracking URI points at a server.

## Configuration

Settings use the `GC_` prefix (see `.env.example`): the model directory, the MLflow
tracking URI and experiment, the drift PSI threshold, and the retrain MAPE threshold.
