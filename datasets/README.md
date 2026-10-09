# Datasets

## `spain_demand_real.csv` — real Spanish grid demand (ENTSO-E, 2015–2016)

Real hourly electricity demand for the Spanish grid, used to validate the forecaster
on data it did not shape (see *Validated on real grid demand* in the top-level README).

**Source.** The ENTSO-E transparency figures for Spain, via the Hugging Face mirrors
[`vitaliy-sharandin/energy-consumption-hourly-spain`](https://huggingface.co/datasets/vitaliy-sharandin/energy-consumption-hourly-spain)
(demand) and
[`vitaliy-sharandin/energy-consumption-weather-hourly-spain`](https://huggingface.co/datasets/vitaliy-sharandin/energy-consumption-weather-hourly-spain)
(weather). The two are joined on the hourly timestamp.

**Columns.**

| Column         | Unit | Meaning                                                        |
| -------------- | ---- | -------------------------------------------------------------- |
| `timestamp`    | —    | local hour (the stored UTC offset flips with daylight saving)  |
| `temperature`  | °C   | Madrid temperature (converted from Kelvin)                     |
| `demand`       | MW   | `total load actual` — the real realised grid demand (target)   |
| `tso_forecast` | MW   | `total load forecast` — the operator's published day-ahead forecast |

17,512 hourly rows span 2015-01-01 to 2016-12-31. Rows with no realised demand reading
are dropped on fetch. `demand` ranges ~18,000–40,000 MW.

`tso_forecast` is kept as a **real professional baseline**: the grid operator's own
day-ahead forecast, which `gridcast.real_benchmark` compares the model against.

### Regenerating

```bash
python -c "from gridcast.real_data import fetch_spain_demand; print(fetch_spain_demand())"
```

Needs outbound access to `datasets-server.huggingface.co`. The licence and terms of the
original ENTSO-E data and the Hugging Face mirrors apply; this slice is vendored for
reproducible benchmarking only.
