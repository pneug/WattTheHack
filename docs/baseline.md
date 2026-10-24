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

## Streamlit app integration

`app.py` now uses the best pipeline model (`pipeline.fit_predict(..., "best")`, see `docs/pipeline.md`) with uploaded data; this baseline is the reference it is compared against:

- Upload one raw file for each reference turbine H1–H5 and O1–O2, plus any nonempty subset of target files T1, T2, and T3. The files must include the original turbine name, timestamp, sensor, and `test_set` columns; upload at most one file per turbine.
- The app applies `clean_turbine()` to each uploaded turbine and pivots every sensor with `to_wide()` (the best model also uses wind direction).
- The automatic period action selects the longest contiguous run of missing wind speed across all uploaded target turbines. The user can also edit the outage dates manually; the selected window is then applied to every uploaded target.
- The model is fitted separately for each uploaded target outside the selected date window (inclusive dates in the UI); energy is the expected power over the predicted wind distribution (not the power of the single best guess), using that turbine's power curve. The dashboard shows the 95 % wind band and a 95 % energy range (`pipeline.evaluate.energy_interval`). Observed target readings are retained for validation where present; hidden or missing observations remain NaN.
- The dashboard summarizes each turbine and a combined total. Combined wind-speed and power RMSE/MAE pool all paired turbine/time observations before calculating the errors; energy totals sum across turbines. Accuracy is shown only when actual target readings are available. The default period is 1–3 Sep 2026, matching the supplied hidden test window.

Run `streamlit run app.py` from the repository root. For the supplied data, automatic detection should select 1–3 Sep 2026; this produces 432 predicted steps per uploaded target and no actual target readings in the window.
