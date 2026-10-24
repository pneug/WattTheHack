# Pipeline: swappable stages, one config per experiment

`pipeline/` rebuilds park T wind speed for any outage window and turns it into lost energy. Each experiment is a config dict that picks one entry per stage. Swapping the model, the features or the power conversion leaves everything else unchanged.

```bash
python -m pipeline.run E12_dir8_smooth60_power  # backtest one config -> experiments/results.csv + LOG.md leaderboard
python -m pipeline.run --all                 # every config in pipeline/experiments.py
python -m pipeline.dense <id> [...]          # 83 overlapping outages (one per day) -> experiments/dense/
python -m pipeline.dense --analyze <id> ...  # paired tests, hidden-like regime, kWh interval coverage
python -m pipeline.submit                    # BEST config on 1-3 Sep -> submission/hidden_predictions.csv
python -m pipeline.figures                   # reports/experiments.html
```

## Stages (`pipeline/stages.py`)

| Stage | Picks | Entries |
|---|---|---|
| `features` | Inputs, from parks H and O only | `park_means`, `all_refs`, `park_means_smooth`, `lags`, `direction`, `hour`, `hinges`, `park_means_tf`, `h_stddev`, `temperature`, `pressure` |
| `weights` | How much each training step counts | `none`, `exp_decay`, `window` |
| `model` | Fit one per turbine | `ols`, `variance_ratio`, `ridge`, `hgb` |
| `interval` | The 95% band per step | `gaussian` (optional `bins`), `conformal`; `"calib": "oos"` uses backtest residuals |
| `power` | Wind speed to kW | `point` = P(v̂), `expected_gauss` / `expected_empirical` = E[P(v)] |

To add an entry, write a function or class in the registry. Then add a config with `add(id, parent, desc, <stage>=...)` in `pipeline/experiments.py`.

## Rules enforced in code

- The outage window is never in the fit (`core.fit_predict` asserts this).
- No feature column may come from park T (`build_X` asserts this).
- If an extra feature is missing at a step, that step falls back to the park-means OLS, so a prediction is never NaN while H and O are known.
- Evaluation uses 28 simulated 3-day outages (anchored at 1 Sep, each fitted without itself) plus 4–9 Sep. Decision metric: mean |park energy deviation| per outage (the judges score the park total), together with power RMSE. See `experiments/LOG.md`.

## Current BEST: `E12_dir8_smooth60_power`

`T = a + b·H + c·O` on the 60-min average of the neighbours, plus a separate offset and H factor for each of 8 wind-direction sectors. The spread is estimated per wind-speed bin, and power is the power curve averaged over that spread.

| | E00 baseline | BEST |
|---|---|---|
| Wind RMSE (backtest) | 1.261 m/s | 1.200 m/s |
| Power RMSE (backtest) | 737 kW | 682 kW |
| Mean \|park energy dev\| per outage (judged) | 11.0% | 8.5% |
| Park energy bias (mean signed) | -8.5% | +1.4% |
| Mean \|energy dev\| per turbine (not judged) | 13.8% | 10.0% |
| Energy bias T1/T2/T3 | -8.6/-5.4/-7.2% | +2.3/+1.5/+2.2% |
| 95% band coverage | 94.6% | 94.8% |
| Lost energy 1–3 Sep (hidden) | | 322.4 MWh (95%: 247.1–377.6) |

## Using it in the Streamlit app

This replaces step 2 of `docs/baseline.md`. The output has the same columns as `baseline.run_outage`, plus `<T>_lo` and `<T>_hi`.

```python
from pipeline import load_data, run_outage
from pipeline.evaluate import energy_interval

@st.cache_data
def get_data():
    return load_data()                      # dict of wide tables; ["wind_speed"] is the old `ws`

res = run_outage(get_data(), start, end)    # config="best"; naive timestamps, end exclusive
total_kwh = sum(res[f"{t}_loss_kwh"].sum() for t in ["T1", "T2", "T3"])
lo_kwh, hi_kwh = energy_interval(total_kwh)  # 95% range from the backtest (needs experiments/runs/<BEST>.json)
```

- **Band:** `<T>_lo` / `<T>_hi` give the band for the wind-speed chart.
- **Methodology tab:** the figures in `pipeline/figures.py` return Plotly figures you can drop straight into the tab.
- **After changing BEST:** run `python -m pipeline.run <BEST>` once, so that `energy_interval` has its backtest file.

## Optional side experiments

See [NN and TabPFN experiments](side-experiments.md) for results, API setup, reproducible commands and execution status. `S01_mlp_matched`, `S02_mlp_refs` and `S03_tabpfn35_refs` use the standard runner/logging. `--all` excludes these unless `--include-side-experiments` is supplied. TabPFN uses an isolated optional environment and resumes complete windows from local checkpoints.
