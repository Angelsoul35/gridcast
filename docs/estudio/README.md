# Guía de estudio — GridCast

Esta guía está en español y explica el proyecto para que lo estudies y lo sustentes en una
entrevista. El código y la documentación técnica están en inglés; aquí va el *por qué*.

## Orden de lectura

1. [Pronóstico y features](01-pronostico-y-features.md) — cómo se convierte una serie de
   tiempo en un problema tabular y cómo pronostica el modelo.
2. [El ciclo MLOps](02-mlops.md) — backtesting, detección de drift, registro en MLflow y la
   decisión de reentrenar.
3. [Preguntas de entrevista](03-preguntas-de-entrevista.md) — preguntas típicas con
   respuestas.

## Cómo correrlo mientras estudias

```bash
make install
make demo     # entrena, backtest, drift, decisión de reentrenar
make test     # 21 pruebas
```

El concepto central: **pronosticar la demanda eléctrica y mantener el modelo bueno en el
tiempo** — con backtesting de origen móvil, detección de drift, seguimiento en MLflow y una regla
clara para reentrenar.
