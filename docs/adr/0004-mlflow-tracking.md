# ADR-0004: Track every run in MLflow

* Status: accepted
* Date: 2026-02-11

## Context

Without a record of training runs, you cannot compare models, reproduce a result, or roll
back when a new model is worse. "Which model is in production and how good was it?" must
have an answer.

## Decision

Every training run is logged to MLflow: parameters, metrics, and the saved model as an
artifact, tagged with a content-based version. The default is a local file store; the
tracking URI points at a server in production.

## Consequences

* **Good:** runs are comparable and reproducible; rollback is possible; the model version
  in a decision can be traced to its metrics.
* **Cost:** a dependency and a store to manage. The local file store keeps the dev setup
  trivial, and a server is a configuration change, not a code change.
