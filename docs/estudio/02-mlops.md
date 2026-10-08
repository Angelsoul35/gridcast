# 2. El ciclo MLOps

Entrenar un modelo una vez es lo fácil. Lo difícil es **mantenerlo bueno en el tiempo**: el
clima cambia, los patrones de consumo cambian, y un modelo que era preciso el año pasado
empeora sin avisar. GridCast muestra el ciclo completo.

## 1. Backtesting (rolling origin)

Una sola división train/test puede salir con suerte. En `backtest.py` hacemos
**rolling-origin**: entrenamos con datos hasta un punto, pronosticamos la siguiente
ventana, medimos el error, movemos el punto hacia adelante y repetimos. El promedio sobre
los folds es una estimación más justa. (Ver
[ADR-0002](../adr/0002-lightgbm-tabular-features.md).)

## 2. Detección de drift

En `drift.py`. Comparamos la distribución de cada feature (temperatura, demanda) entre una
ventana de **referencia** y la ventana **actual**, con el **PSI** (Population Stability
Index) y la distancia de **Kolmogorov-Smirnov**.

> **Concepto — PSI.** Mide cuánto se movió una distribución. Regla práctica: < 0.1 sin
> cambio real, 0.1–0.25 algo de cambio, > 0.25 cambio grande. Si el PSI pasa el umbral,
> hay drift: la realidad se alejó de lo que el modelo aprendió.

## 3. Registro en MLflow

En `registry.py`. Cada entrenamiento registra en MLflow sus parámetros, sus métricas y el
modelo guardado como artefacto, con la versión del modelo como etiqueta.

> **Por qué importa.** Sin un registro de corridas no puedes comparar modelos, reproducir
> un resultado ni hacer rollback cuando un modelo nuevo sale peor. "¿Qué modelo está en
> producción y qué tan bueno era?" tiene que tener respuesta. (Ver
> [ADR-0004](../adr/0004-mlflow-tracking.md).)

Por defecto usa un file store local (perfecto para desarrollo y CI); en producción la URI
apunta a un servidor MLflow con base de datos.

## 4. La decisión de reentrenar

En `retrain.py`. Convierte dos señales en un sí/no: reentrena si el error del backtest pasa
el umbral **o** si hay drift. Reentrenar en cada lote es un desperdicio; no reentrenar
nunca es peligroso. La regla cierra el ciclo. (Ver
[ADR-0003](../adr/0003-drift-and-retrain.md).)
