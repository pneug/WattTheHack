# Neural-network and TabPFN side experiments

These are curiosity experiments using the existing outage-reconstruction pipeline. They use the same 28 simulated three-day outages and 4–9 September validation window, one model per target turbine, five-bin Gaussian residual bands, and expected-power conversion. `BEST` remains `E12_dir8_smooth60_power`.

## Completed model results

| Experiment | Backtest wind RMSE [m/s] | Backtest power RMSE [kW] | Mean absolute park energy deviation [%] | Validation park deviation [%] | Backtest band coverage [%] |
|---|---:|---:|---:|---:|---:|
| E12 existing model | 1.1995 | 682.3263 | 8.5282 | +2.5278 | 94.76 |
| S01_mlp_matched | 1.2012 | 681.6251 | 9.0791 | -0.7667 | 94.61 |
| S02_mlp_refs | 1.2087 | 688.7175 | 9.1310 | +0.2481 | 93.42 |
| S03_tabpfn35_refs | 1.2368 | 707.1839 | 9.2147 | +1.2716 | 73.08 |
| S04_tabpfn35_oos (calibration diagnostic) | 1.2368 | 702.4159 | 11.2721 | +2.2409 | 94.95 |

S01 changes only the model relative to E12. S02 supplies seven individually smoothed reference wind speeds, two smoothed park means, and the reference park's wind-direction sine/cosine (11 inputs). Both use a small ReLU MLP with hidden layers 32 and 16, Adam, weighted feature/target standardization, a fixed seed, and an 80-epoch ceiling. Epoch selection uses a chronological 10% tail of the supplied training rows; the selected epoch count is then refitted on all supplied training rows. No outage targets enter scaling, stopping or fitting.

Each run contains 28 backtests plus validation, three targets, and 38,880 prediction rows. A second complete run reproduced every rounded summary metric exactly. The verified runs took about 72 and 78 seconds. Selected epochs ranged from 3–50 for S01 and 2–61 for S02. Genuine sample weighting requires scikit-learn >=1.7.

The NN adds no clear gain here: S01 has essentially the same pointwise accuracy as E12, while both NNs have slightly worse average backtest park-energy error. Their near-zero energy error on the single validation window does not establish general superiority. There was no hyperparameter search or multi-seed comparison.

## TabPFN configuration and execution status

`S03_tabpfn35_refs` completed all 28 backtests plus validation after the user explicitly approved uploading this project's turbine features/targets to Prior Labs and consuming potentially billable API quota. The run took 779.92 seconds, with no retries or missing prediction/band/power outputs across 38,880 rows. Its backtest wind and power RMSE and average absolute park-energy error are worse than E12; it has lower absolute park-energy error on 12 of 28 outages. Its better park-energy deviation on the single validation window is not enough to prefer it.

The selected model is hosted **TabPFN-3.5-Plus**, pinned to `model_path="v3.5_default"` with eight estimators and `fit_mode="fit_with_cache"`. The official [model-selection guide](https://docs.priorlabs.ai/models/selecting-model-version) describes this selector; Fast is alpha, while Thinking uses extra optimization and has a lower fit-rate allowance. Stable Plus is a practical first comparison across 87 independent fold/turbine fits. The task is offline multivariate regression with contemporaneous reference measurements, so a regressor using the pipeline's existing features fits the task directly.

S03 uses S02's 11 inputs without additional scaling. A seeded split reserves 512 training-only rows per fit for residual calibration **before** choosing context. The 16,384-row context ceiling retains all other eligible training rows in this dataset: 10,837–11,465 rows per fit. Calibration rows never enter the model's context. This avoids calibrating uncertainty with predictions on the context's own labels. Calibration is pointwise rather than a separate three-day block. The nominal 95% bands cover only 73.08% of backtest actuals and 72.72% on validation; their mean width is 2.4864 m/s versus E12's 4.5409 m/s. A plausible explanation is that nearby context observations make random held-out calibration too optimistic for a contiguous outage. This is an inference from the coverage results, not a causal test.

The completed run used 87 fits and 174 prediction calls, querying 82,944 rows in total. Both cached and fallback inference had been quoted at **1,740,000 tokens** using conservative quotes at the largest context size. Actual account usage rose from 0 to **1,740,000 of 20,000,000 monthly tokens**, exactly matching the quote; the usage snapshots and delta are saved in `experiments/runs/S03_tabpfn35_refs_usage.json`. These are API computation units, not a monetary price estimate; see [metering](https://docs.priorlabs.ai/api/metering) and the [cost-estimate API](https://docs.priorlabs.ai/api-reference/python/tabpfn-client/usage-and-costs). The preflight sent dimensions and configuration only; the approved inference run uploaded the training and query data.

## Cached calibration diagnostic

`S04_tabpfn35_oos` reuses all 29 completed S03 checkpoints and applies the existing E07 out-of-sample residual-calibration protocol. It made **zero new fits or API calls**. The source checkpoints and S03 result artifacts remained unchanged. Wind predictions and actuals match S03 within JSON floating-point precision (maximum difference below 3e-15 m/s); all rounded wind metrics are identical.

Calibration pools contain residuals from other backtest windows that do not overlap the scored window; validation rows never supply calibration residuals. This excludes the scored window's residuals directly. However, donor models may have used labels from the scored window during their own training. That dependence limits this protocol to a diagnostic here; a strictly nested blocked evaluation would need additional model fits. S04 remains separate from the primary model comparison and does not change BEST.

Backtest coverage rises from 73.08% to 94.95%, with mean band width 4.7620 m/s; validation coverage rises from 72.72% to 93.79%. Because the pipeline computes expected power using residual spread, recalibration also changes energy estimates: power RMSE improves from 707.2 to 702.4 kW, while average absolute park-energy error worsens from 9.2147% to 11.2721%. Better interval coverage alone does not ensure a better energy total.

## Running and logging

From the repository root in PowerShell:

```powershell
# NN: existing main environment
.\.venv\Scripts\python.exe -m pipeline.run S01_mlp_matched S02_mlp_refs

# Isolated optional client environment (already prepared in this workspace)
uv venv .venv\tabpfn --python .venv\Scripts\python.exe
uv pip install --python .venv\tabpfn\Scripts\python.exe -r requirements-side-experiments.txt

# Shape-only quota/cost check, followed by the explicitly authorized inference run
.\.venv\tabpfn\Scripts\python.exe -m pipeline.side_preflight S03_tabpfn35_refs
.\.venv\tabpfn\Scripts\python.exe -u -m pipeline.run S03_tabpfn35_refs

# Offline diagnostic: requires the completed S03 checkpoints; no extra API calls
.\.venv\tabpfn\Scripts\python.exe -m pipeline.run S04_tabpfn35_oos
```

The official client 0.6.1 constrains pandas, scikit-learn and PyArrow to versions older than the main environment. Keep it isolated. The original main versions (pandas 3.1.0, scikit-learn 1.9.1, PyArrow 26.0.0) were restored, and a complete E12 backtest matched its saved summary exactly both in the main and isolated environments.

The API adapter reads `tabpfn_API_key` from the root `.env`; it also accepts `TABPFN_TOKEN`, `TABPFN_API_KEY` and `PRIORLABS_API_KEY`, with process-environment values taking precedence. It sets the token programmatically in memory and does not log or save it. Nonuniform training weights fail explicitly because the API does not support them.

All successful runs use `pipeline.run`: append to `experiments/results.csv`, write config/summary/per-window metrics plus runtime and model metadata to `experiments/runs/<id>.json`, save `<id>_steps.csv`, and refresh the generated `experiments/LOG.md` leaderboard. TabPFN's separate `<id>_preflight.json` records the plan and cost quotes without pretending to be an evaluated result.

Completed TabPFN windows checkpoint atomically as JSON under ignored `.cache/side_backtests/`. Re-running the same experiment reuses completed windows. Changes to config, sensor values, timestamps, relevant source or power curve invalidate checkpoints. Only complete three-turbine windows are published; work in an interrupted partial window can still need repeating. No SDK token or remote model identifier is stored in checkpoints.

S04 requires a direct source with matching features, weights and model. A missing or invalid source checkpoint raises before any fit. The original S03 run remains reproducible independently of the diagnostic.

`python -m pipeline.run --all` runs the ordinary challenge experiments. Side experiments require their explicit IDs or `--all --include-side-experiments`; the latter includes remote API usage and requires the optional client environment.

## Verification

All **35 tests passed in both the main and isolated environments**. `python -m unittest discover -s tests -v` covers train-only scaling/stopping, real NN weighting and determinism, API context/calibration separation, batching and cache behavior, token/error handling, outage-target invariance, opt-in CLI routing, shape-only cost accounting, checkpoint invalidation/atomicity, and cache-only recalibration with donor exclusions and a fail-before-fit guard. API behavior is mocked in the offline tests; actual TabPFN quality is measured by the completed S03 evaluation. A fresh complete E12 backtest still matches every saved summary metric after integration.
