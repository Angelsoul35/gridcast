# 3. Preguntas de entrevista

## Nivel junior / media

**¿Qué hace GridCast en una frase?**
Pronostica la demanda eléctrica por hora con LightGBM y cierra el ciclo MLOps: backtesting,
detección de drift, registro en MLflow y una decisión de reentrenar.

**¿Por qué conviertes la serie en una tabla?**
Porque así puedo usar gradient boosting (LightGBM), que es rápido y preciso en datos
tabulares. Las features de calendario capturan la forma diaria y semanal, y los lags
capturan el nivel reciente.

**¿Qué es fuga de datos y cómo la evitas?**
Usar información del futuro en las features. La evito con `shift(1)` en las medias móviles y
usando solo lags del pasado, así una feature de la hora *t* nunca ve el valor de *t*.

## Nivel media / senior

**¿Qué es el pronóstico recursivo y qué riesgo tiene?**
Predecir una hora, agregarla a la serie y usarla para la siguiente. El riesgo es que el
error se acumule en horizontes largos, porque cada paso usa la predicción anterior. El
backtesting mide ese efecto.

**¿Por qué backtesting rolling-origin y no un solo split?**
Un solo split puede salir con suerte o caer en un periodo atípico. Rolling-origin entrena y
evalúa en varias ventanas a lo largo del tiempo, y el promedio es más confiable.

**¿Cómo detectas drift y por qué importa?**
Con PSI y KS entre una ventana de referencia y la actual. Importa porque un modelo empeora
en silencio cuando la realidad se aleja de sus datos de entrenamiento; el drift me avisa
antes de que la calidad caiga.

**¿Cuándo reentrenas?**
Cuando el error del backtest pasa un umbral o cuando hay drift. No reentreno a ciegas en
cada lote ni espero a que falle.

## Nivel senior / diseño

**¿Qué registras en MLflow y para qué?**
Parámetros, métricas y el modelo como artefacto, con la versión como etiqueta. Sirve para
comparar modelos, reproducir resultados y hacer rollback. Permite responder qué modelo está
en producción y qué tan bueno es.

**¿Qué le falta para producción?**
Temperatura futura real (de un pronóstico del clima), un servidor MLflow con base de datos,
reentrenamiento automático disparado por el drift, intervalos de confianza en el
pronóstico, y un panel (por ejemplo Grafana) para seguir métricas y drift en el tiempo.

## Reto para ti

* Agrega intervalos de predicción (cuantiles) con LightGBM en modo quantile.
* Cambia el pronóstico recursivo por uno directo (un modelo por horizonte) y compara el
  error a 168 horas.
* Dispara un reentrenamiento automático cuando `/drift` marque drift, y registra el nuevo
  modelo en MLflow.
