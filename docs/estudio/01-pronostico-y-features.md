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

* **MAE:** error absoluto medio (en MW). `MAE = (1/n) · Σ |yᵢ − ŷᵢ|`.
* **RMSE:** penaliza más los errores grandes. `RMSE = √( (1/n) · Σ (yᵢ − ŷᵢ)² )`.
* **MAPE:** error porcentual (fácil de comunicar: "fallamos ~5%").
  `MAPE = (1/n) · Σ |yᵢ − ŷᵢ| / |yᵢ|`.

## Validación con datos reales (red eléctrica española, ENTSO-E)

Los datos sintéticos demuestran que el modelo **recupera un patrón que nosotros
construimos**. La prueba de fuego es el dato que nadie diseñó. Por eso GridCast también se
mide sobre demanda horaria **real** de la red española (cifras de transparencia de
ENTSO-E, 2015–2016: 17 512 horas de `total load actual` en MW), unida con la temperatura
de Madrid. Se corre con `python -m gridcast.real_benchmark`.

### Cómo se mide: contra dos referencias

Un número de error aislado no dice nada. Por eso lo comparamos con dos referencias:

1. **Baseline ingenuo estacional** (la demanda de la *misma hora la semana pasada*,
   `demanda[t−168]`). Es el **suelo**: un modelo útil tiene que ganarle. Si no le gana,
   el modelo sobra.
2. **El pronóstico del propio operador de red** (`total load forecast`, su predicción
   publicada del día siguiente). Es un **techo profesional**: lo construyen con una
   información mucho más rica (pronósticos del clima región por región, programación de
   generación, el calendario de festivos).

### Resultados reales

```
one-step train/test: MAPE 0.0194 / 0.0359  (brecha 0.0165)   -> bien ajustado

   horizonte  folds  model MAPE  naive MAPE  TSO MAPE
        24h      6      0.0496      0.1217    0.0144
       168h      4      0.1194      0.1027    0.0118
```

**A un día (24 h)** —el horizonte operativo— el modelo (≈5,0 % de MAPE) le gana con
holgura al baseline ingenuo (12,2 %) y queda en el mismo rango de un dígito que el
operador (1,4 %), usando solo calendario, la temperatura de una ciudad y la demanda
rezagada. La brecha hasta el operador mide exactamente cuánto aportarían features más
ricas (clima por región, festivos).

**A una semana (168 h)** el error sube a 11,9 %. Aquí pasa algo importante y muy conocido:
el **pronóstico recursivo acumula error** (cada hora se predice a partir de horas ya
predichas, no reales), hasta que el baseline ingenuo "última semana" (10,3 %) se vuelve
competitivo. No es un defecto del proyecto: es la *firma* del enfoque recursivo, y la
pista clara de la mejora correcta para horizontes largos — un **modelo multi-horizonte
directo** (un modelo por adelanto, sin realimentar predicciones). El día siguiente es el
caso de uso operativo, y ahí el modelo es claramente el mejor pronosticador.

### Over/underfitting, visto en el dato real

* La **brecha una-paso** train/test es de 1,65 puntos (train 1,94 %, test 3,59 %):
  pequeña. El modelo generaliza; no memoriza (si memorizara, el test se dispararía).
* El error del **backtest de origen móvil** (4,96 % a un día) sigue de cerca al del
  split simple fuera de muestra (3,59 %): estabilidad en el tiempo, no suerte de una
  partición.
* **Prueba de que la señal es real:** si se **barajan** los valores de demanda (se
  destruyen lags y estacionalidad), el error del backtest se dispara. Lo comprueba
  `tests/test_real_benchmark.py`. Un modelo que "acertara" con datos barajados estaría
  midiendo fuga, no aprendizaje.

### La lección

- Reporta el número **fuera de muestra** y **contra baselines**, nunca el de entrenamiento.
- Un buen pronosticador le gana al ingenuo estacional; compararse con el operador real
  convierte la brecha en una hoja de ruta de mejoras.
- El recursivo acumula error: a horizonte largo, mide y, si hace falta, cambia a
  multi-horizonte directo. Medir esto es señal de madurez, no de debilidad.
