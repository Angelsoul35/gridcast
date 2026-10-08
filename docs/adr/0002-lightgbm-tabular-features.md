# ADR-0002: Forecast with LightGBM on tabular features

* Status: accepted
* Date: 2026-02-10

## Context

Demand forecasting can be done with classic time-series models (ARIMA), deep models
(LSTM), or gradient boosting on engineered features. We need something accurate, fast to
train, easy to explain, and simple to put in production.

## Decision

We turn the series into tabular features (calendar parts, temperature, lags, rolling
means) and train a LightGBM regressor. We forecast multiple hours recursively.

## Consequences

* **Good:** strong accuracy on this kind of data, fast training, no GPU, and features that
  a person can read. It handles the daily and weekly shape through the calendar features.
* **Cost:** recursive forecasting can drift over a long horizon because each step uses the
  previous prediction. Backtesting measures this, and a direct multi-output model is a
  future option.
