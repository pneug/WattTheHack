# Wind Farm Outage Estimator

BayWa r.e. "Watt The Hack 2026", challenge 1. Park T (turbines T1–T3) had an outage, so its own wind data is missing. We reconstruct T's wind speed from the neighbouring parks H (H1–H5) and O (O1–O2), turn it into power with the turbine power curve (rated 4,500 kW), and add up the lost energy.

**Answer for the judged outage (1–3 Sep 2026): about 322 MWh lost, 95% range 262–387 MWh.**

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
python preprocessing.py           # cleans data/dataset/*.xlsx into data/processed/ (~12 s)
streamlit run app.py              # opens the dashboard
```

- `openpyxl` is listed in `requirements.txt` (pandas needs it to read the `.xlsx` files). If an import error mentions it anyway, run `pip install openpyxl`.
- The raw Excel files must be in `data/dataset/`.

## How it works

- **Data cleaning:** the 10 raw turbine files become one gap-aware table of 10-minute wind speeds (`preprocessing.py`).
- **Model:** per turbine, `T = a + b·H + c·O`, where H and O are the park-mean wind speeds, averaged over 1 hour. The offset and the H factor are fitted separately for each of 8 wind directions.
- **Power conversion:** wind speed goes through the power curve. We average the power over the uncertainty of the wind estimate, E[P(v)]. This removes the systematic under-estimate of energy.
- **Energy:** lost energy in kWh = power in kW x 1/6 h per 10-minute step, summed over the outage and the three turbines.
- **Uncertainty:** a 95% band per time step, and a 95% range for the total energy built from 83 simulated outages.
- **Evaluation:** 28 simulated 3-day outages (each predicted without seeing itself), a dense check with 83 overlapping outages, and the validation week 4–9 Sep. The outage window is never used for fitting.

Final configuration: `E12_dir8_smooth60_power`.

## Results

Average over the 28 simulated outages.

| Metric | Baseline | Final |
|---|---|---|
| Wind speed RMSE | 1.261 m/s | 1.200 m/s |
| Power RMSE | 737 kW | 682 kW |
| Park energy error (mean absolute, the judged metric) | 11.0% | 8.5% |
| Energy bias (mean signed) | -8.5% | +1.4% |
| 95% band coverage | 94.6% | 94.8% |

The 83-outage dense check agrees: park energy error 10.8% (baseline) vs 8.3% (final). The 95% energy range is built from these 83 outages.

## Reproduce

Run from the repo root (after `python preprocessing.py`).

```bash
python -m pipeline.run E12_dir8_smooth60_power   # backtest one config (--all for every config)
python -m pipeline.dense E12_dir8_smooth60_power # 83 overlapping outages -> experiments/dense/
python -m pipeline.submit                        # predict 1–3 Sep -> submission/hidden_predictions.csv
python -m pipeline.figures                       # rebuild reports/experiments.html and app figures
```

- `python baseline.py` runs the simple OLS baseline on its own.
- To add an experiment, add a config in `pipeline/experiments.py` (see `docs/pipeline.md`).

## Repo layout

| Path | Purpose |
|---|---|
| `app.py` | Streamlit dashboard: upload data, pick an outage, see wind, power and lost energy |
| `methods_tabs.py` | Content of the "Methodology & Validation" and "Things We Tried" tabs |
| `preprocessing.py` | Loads and cleans the raw data, power curve, wind-to-power helpers |
| `baseline.py` | Simple OLS baseline (`T = a + b·H + c·O`) |
| `turbine_availability.py` | Detects which turbines are available in uploaded data |
| `pipeline/` | Swappable stages (features, model, interval, power), backtests, dense check, submission, figures |
| `experiments/` | `LOG.md` leaderboard, `results.csv`, per-run and dense results |
| `docs/` | Written documentation (see below) |
| `reports/` | `experiments.html` report and figures used by the app |
| `submission/` | `hidden_predictions.csv`: predictions for 1–3 Sep |
| `notebooks/` | `01_key_visuals.ipynb`, the key figures |
| `data/` | `dataset/` raw Excel files, `processed/` cleaned output (generated) |

## Docs

- [`docs/preprocessing.md`](docs/preprocessing.md): data cleaning
- [`docs/baseline.md`](docs/baseline.md): the OLS baseline
- [`docs/pipeline.md`](docs/pipeline.md): pipeline stages, commands, current best model
- [`docs/insights.md`](docs/insights.md): key findings and dead ends, written for non-experts
- [`experiments/LOG.md`](experiments/LOG.md): every experiment with its numbers
- [`reports/experiments.html`](reports/experiments.html): interactive experiment report
