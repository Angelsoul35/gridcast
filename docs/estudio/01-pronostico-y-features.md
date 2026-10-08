# 1. Pronóstico y features

## De serie de tiempo a problema tabular

Pronosticar demanda se puede hacer con modelos clásicos (ARIMA), redes (LSTM) o
**gradient boosting sobre features**. Aquí usamos el tercero: convertimos la serie en una
tabla de features y entrenamos un `LightGBM`.

## Las features (sin fuga de datos)

En `features.py` construimos:

* **Calendario:** hora, día de la semana, mes, fin de semana. Capturan la forma diaria
  (picos de mañana y tarde, valle en la madrugada) y semanal (fines de semana más bajos).
* **Exógena:** temperatura (la demanda sube con frío o calor extremos).
* **Lags:** demanda hace 24, 48 y 168 horas (ayer, antier, hace una semana).
* **Medias móviles:** promedio de la última semana y del último día.

> **Concepto — fuga de datos (leakage).** Una feature para la hora *t* solo puede usar
> datos **anteriores** a *t*. Si usara el valor de *t*, el modelo haría trampa y en
> producción fallaría. Por eso las medias móviles usan `shift(1)`: nunca incluyen la hora
> actual.

## Pronóstico recursivo (multi-paso)

Para una hora, basta construir las features y predecir. Para muchas horas, vamos paso a
paso: predecimos *t+1*, lo agregamos a la serie, y con esa serie construimos los lags de
*t+2*, y así. Esto es el **pronóstico recursivo**.

Necesita la temperatura futura. En el demo y la API usamos un valor simple ("misma hora la
semana pasada"); en un sistema real vendría de un pronóstico del clima.

> **Concepto — gradient boosting (LightGBM).** Muchos árboles pequeños en secuencia, cada
> uno corrige el error del conjunto anterior. LightGBM es una implementación rápida y
> eficiente. Funciona muy bien en datos tabulares como estos.

## Métricas

* **MAE:** error absoluto medio (en MW).
* **RMSE:** penaliza más los errores grandes.
* **MAPE:** error porcentual (fácil de comunicar: "fallamos ~5%").
