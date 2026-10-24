# Data preprocessing

`preprocessing.py` turns the 10 raw Excel files into one clean, gap-aware table. Run it once, or import it.

```bash
python preprocessing.py          # writes data/processed/ (gitignored, ~12 s)
```

```python
from preprocessing import load_clean, to_wide, load_power_curve, wind_to_power, energy_kwh

long = load_clean()                       # 132,480 rows = 13,248 steps x 10 turbines
ws = to_wide(long)                        # timestamp x turbine wind speed [m/s]
kw = wind_to_power(ws["T1"], load_power_curve())
kwh = energy_kwh(kw)
```

## What gets cleaned (in this order)

| # | Problem in the raw data | Rule | Rows affected |
|---|---|---|---|
| 1 | **3–4 Aug rows are absent** (timestamp jump, not NaN) | Reindex every turbine to the full 10-min grid, 1 Jul 00:00 to 30 Sep 23:50 | 288 per turbine |
| 2 | **Sensor dropout in park T**: speed, direction, temperature and stddev all exactly 0 | Set all sensors to NaN (it is not calm wind) | T1: 3, T2: 2, T3: 4 |
| 3 | Values outside physical ranges | Set to NaN; direction 360° becomes 0° | 0 today (guard) |
| 4 | Frozen sensor (same non-zero speed for ≥ 2 h) | Set wind speed to NaN | 0 today (guard) |
| 5 | **Park O placeholders**: air pressure is a constant 990.27, stddev is all NaN | Constant columns become NaN, so nobody models on them | O1, O2 |
| 6 | `test_set` is NaN on re-added rows | Replace with a `split` column derived from the timestamp | all |

**Never done:** filling the target with zeros or interpolating long gaps. NaN means "unknown", not "no wind" or "no power".

**Kept on purpose:** park O wind speeds of 0.0–0.3 m/s at night. They are real calm periods (direction keeps changing, other sensors are fine), and park O rounds to 0.1 m/s.

## Splits

| `split` | Period | Target T | Use |
|---|---|---|---|
| `train` | 1 Jul – 31 Aug, 10–30 Sep | known | fit |
| `val` | 4–9 Sep | known | validate, then refit on train + val |
| `hidden` | 1–3 Sep (432 steps) | **all NaN** | the judges' test: predict this |

## Result per turbine

| Turbine | NaN wind speed in files | Sensor dropouts | Longest sensor gap | Pressure / stddev usable | Valid wind speed |
|---|---|---|---|---|---|
| T1 | 156 | 3 | **5.0 h** (7 Aug) | yes | 96.5% |
| T2 | 109 | 2 | 0.3 h | yes | 96.9% |
| T3 | 111 | 4 | 0.3 h | yes | 96.9% |
| H1 | 286 | 0 | **15.7 h** (14–15 Jul, right after a 10 h gap) | yes | 95.5% |
| H2–H5 | 116–145 | 0 | ≤ 0.3 h | yes | 96.6–96.8% |
| O1, O2 | 108–131 | 0 | 0.3 h | **no** | 96.7–96.9% |

"Valid" counts all non-hidden grid steps, so it includes the 288 absent rows. The full table is in `data/processed/quality_report.csv`.

## Output columns (`clean_long.csv`)

| Column | Meaning |
|---|---|
| `timestamp` | 10-min step, local time (CEST, no DST switch in range) |
| `turbine_name`, `park` | e.g. `H3`, `H` |
| `wind_speed` … `wind_speed_stddev` | Cleaned sensors (NaN where unknown) |
| `split` | `train` / `val` / `hidden` |
| `flag_row_absent` | Row was missing from the file (3–4 Aug) |
| `flag_dropout`, `flag_out_of_range`, `flag_stuck` | Which rule set the value to NaN |
| `flag_missing_ws` | Wind speed is NaN after cleaning, for any reason |

Also written: `wind_speed_wide.csv` (timestamp × turbine) for quick looks.

## Helpers

| Function | Does |
|---|---|
| `load_clean()` | All turbines, cleaned, long format |
| `to_wide(long, var)` | Pivot one variable to timestamp × turbine |
| `fill_short_gaps(s, max_steps=2)` | Interpolate gaps ≤ 20 min. **Reference features only**, never the target |
| `load_power_curve()` | Power curve (identical for T1–T3) |
| `wind_to_power(ws, curve)` | Linear interpolation; 0 kW below cut-in (3 m/s) and above 26 m/s (assumed cut-out); NaN stays NaN |
| `energy_kwh(kw)` | Σ P[kW] × 1/6 h |
| `quality_report(long)` | The table above |

## Things to keep in mind for modelling

- **Reindex before lags or rolling windows.** Otherwise a shift jumps silently across the absent 3–4 Aug rows. `load_clean()` already does this.
- **Use references from parks H and O only.** T1–T3 must not predict each other: the README forbids it, and all of park T is NaN in the hidden window.
- **Do not use park O pressure or stddev**; they are NaN after cleaning. Pressure and stddev features have to come from park H.
- **Validate chronologically** (4–9 Sep). A random split on 10-min data leaks.
