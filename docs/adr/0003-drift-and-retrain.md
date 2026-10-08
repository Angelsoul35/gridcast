# ADR-0003: Watch for drift and decide when to retrain

* Status: accepted
* Date: 2026-02-11

## Context

A model trained on past data gets worse when the live data moves away from it. Retraining
on every new batch is wasteful; never retraining is dangerous. We need a signal.

## Decision

We watch the input features with the Population Stability Index (PSI) and back the whole
thing with rolling-origin backtesting. The retrain policy says "retrain" when the backtest
error is above a threshold or drift is detected.

## Consequences

* **Good:** retraining happens for a reason, not on a blind schedule. Both signals are
  simple, cheap, and explainable.
* **Cost:** the thresholds need tuning per system. PSI is a heuristic, so it is a signal to
  act on, not a proof. That is why we combine it with a backtest.
