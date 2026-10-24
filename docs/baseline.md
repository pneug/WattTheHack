# OLS baseline

`baseline.py` reconstructs the wind speed of park T from the park H and O averages, then maps it through the power curve to lost energy. Three numbers per turbine, all explainable.

```
T_i(t) = a_i + b_i * H(t) + c_i * O(t)        H, O = mean wind speed of parks H and O (10 min)
power  = power_curve(T_i)      energy [kWh] = power [kW] * 1/6 h
```

## What it does

| Step | Detail |
|---|---|
| Inputs | Cleaned wind speed from `preprocessing.py`; H = mean of H1–H5, O = mean of O1–O2 (a missing turbine is skipped) |
| Fit | Least squares per turbine on every step where T, H and O are known, **excluding the outage window** |
| Predict | `a + b*H + c*O`, clipped at 0 m/s; NaN if H or O is missing |
| Energy | Predicted speed to kW via the power curve (0 below 3 m/s, above 26 m/s), times 1/6 h |
| Static | One fixed (a, b, c) per turbine for the whole outage; it does not adapt hour by hour |

Fitted values (validation 4–9 Sep held out): T1 `0.80 + 0.39·H + 0.37·O`, T2 `0.50 + 0.41·H + 0.44·O`, T3 `0.72 + 0.39·H + 0.38·O`.
Validation 4–9 Sep: wind RMSE T1 1.60 / T2 1.25 / T3 1.34 m/s (see notebook sections 11a–11d and 11c+).

**Known limits:** T1 energy is about 19% too high on 4–9 Sep (T2 −5%, T3 −8%); the park total is close only because errors cancel. The 95% band (`sigma`) is too narrow (covers ~89%). The fit drifts slightly by month.

## API

| Call | Returns |
|---|---|
| `OLSBaseline().fit(ws, exclude=(start, end))` | fitted model |
| `.predict(ws)` | DataFrame, T1–T3 wind speed [m/s] |
| `.power(ws, curve)` | DataFrame, T1–T3 power [kW] |
| `.energy_loss(ws, curve)` | DataFrame, T1–T3 energy per step [kWh] |
| `.coefficients()` | a, b, c and residual `sigma` per turbine |
| `run_outage(ws, start, end)` | one table for the window: `T1_pred`, `T1_actual`, `T1_loss_kwh`, ... |

`ws` = `to_wide(load_clean())` from `preprocessing.py`. Try it: `python baseline.py`.

## Plug it into the Streamlit app

`app.py` has three placeholder functions. Replace them; the rest of the UI stays as is.

**1. Load the data once** (top of `app.py`):

```python
from baseline import OLSBaseline, run_outage
from preprocessing import load_clean, load_power_curve, to_wide

@st.cache_data
def get_wind_table() -> pd.DataFrame:
    return to_wide(load_clean())          # timestamp x turbine, ~10 s the first time
```

**2. Replace `calculate_predictions`** (same columns the charts already expect):

```python
def calculate_predictions(df_reference, outage_period, model_params):
    start, end = outage_period                              # dates, end inclusive
    res = run_outage(get_wind_table(), start, pd.Timestamp(end) + pd.Timedelta(days=1))
    t = model_params.get("turbine", "T1")                   # pick one turbine for the chart
    return pd.DataFrame({
        "timestamp": res.index,
        "reference_wind_speed": get_wind_table().loc[res.index, ["H1", "H2", "H3", "H4", "H5"]].mean(axis=1).values,
        "predicted_wind_speed": res[f"{t}_pred"].values,
        "actual_wind_speed": res[f"{t}_actual"].values,     # NaN in the hidden window: the chart should skip it
    })
```

**3. Replace `calculate_energy_loss`:**

```python
def calculate_energy_loss(predictions, power_curve):
    kwh = wind_to_power(predictions["predicted_wind_speed"], load_power_curve()) / 6
    return pd.DataFrame({"timestamp": predictions["timestamp"], "loss_kwh": kwh})
```

(add `from preprocessing import wind_to_power` to the import above). For all three turbines, sum `res[[c for c in res if c.endswith("_loss_kwh")]]` instead.

**4. Check:** `streamlit run app.py`, choose 4–9 Sep (actual T known): predicted and actual lines should follow each other; for 1–3 Sep the actual line is empty.

**Gotchas**
- The app's dates are 2025 placeholders; the data covers **1 Jul–30 Sep 2026**. Set `DEFAULT_OUTAGE_PERIOD` to `(date(2026, 9, 1), date(2026, 9, 3))`.
- The app's timestamps are tz-aware (`Europe/Berlin`); ours are naive local time. Use `.tz_localize("Europe/Berlin")` only for display.
- The app's example power curve is not the real one; use `load_power_curve()`.
- Leave the outage window out of the fit (`run_outage` does this) so the validation numbers stay honest.
