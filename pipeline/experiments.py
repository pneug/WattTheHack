"""Experiment configs. One dict per experiment; each changes one thing versus its parent.

Add a new experiment with `derive(parent, **changes)`, then run `python -m pipeline.run <id>`.
Set BEST to the id that the app and the submission should use.
"""

from __future__ import annotations

import copy

E00 = {
    "desc": "OLS T = a + b*H + c*O, static per turbine",
    "parent": None,
    "features": ["park_means"],
    "weights": {"name": "none"},
    "model": {"name": "ols"},
    "power": {"mode": "point"},
    "interval": {"name": "gaussian"},
}


def derive(parent_id: str, desc: str, **changes) -> dict:
    """Copy CONFIGS[parent_id] and override whole stages, e.g. derive("E00_baseline", "...", weights={...})."""
    cfg = copy.deepcopy(CONFIGS[parent_id])
    cfg.update(copy.deepcopy(changes), desc=desc, parent=parent_id)
    return cfg


CONFIGS: dict[str, dict] = {
    "E00_baseline": E00,
}


def add(eid: str, parent_id: str, desc: str, **changes) -> None:
    CONFIGS[eid] = derive(parent_id, desc, **changes)


# Round 1: one change each versus the baseline (ablation), then stack the winners.
B = "E00_baseline"
PM = "park_means"

# E01 time weighting: recent (both sides of the outage) data counts more
for hl in (3, 7, 14, 30):
    add(f"E01_decay{hl}", B, f"time weights, half-life {hl} d", weights={"name": "exp_decay", "half_life_days": hl})
for days in (7, 14, 30):
    add(f"E01_window{days}", B, f"fit only on +-{days} d around the outage", weights={"name": "window", "days": days})

# E02 smoothed references: centred rolling mean of H and O instead of the raw 10-min values
for steps in (3, 6, 12):
    add(f"E02_smooth{steps * 10}min", B, f"references smoothed over {steps * 10} min",
        features=[{"name": "park_means_smooth", "steps": steps}])

# E03 power conversion: E[P(v)] over the predictive distribution instead of P(v_hat)
add("E03_power_gauss", B, "E[P(v)], Gaussian residuals", power={"mode": "expected_gauss"})
add("E03_power_empirical", B, "E[P(v)], empirical residuals", power={"mode": "expected_empirical"})

# E04 shrinkage fix: variance-ratio MCP keeps the spread of the real wind
add("E04_variance_ratio", B, "variance-ratio MCP instead of OLS", model={"name": "variance_ratio"})

# E05 nonlinearity and heteroscedasticity
add("E05_hinges", B, "piecewise linear (knots 3/6/9 m/s)", features=[PM, {"name": "hinges"}])
add("E05_log", B, "fit on log1p(H), log1p(O)", features=[{"name": "park_means_tf", "fn": "log1p"}])
add("E05_sqrt", B, "fit on sqrt(H), sqrt(O)", features=[{"name": "park_means_tf", "fn": "sqrt"}])
add("E05_sigma_bins", B, "residual spread per wind-speed bin (5)", interval={"name": "gaussian", "bins": 5})

# E06 secondary corrections, one at a time
add("E06_hour", B, "+ hour of day (and x H)", features=[PM, {"name": "hour"}])
add("E06_dir", B, "+ wind direction sin/cos", features=[PM, {"name": "direction"}])
add("E06_dir4", B, "+ direction quadrant (x H)", features=[PM, {"name": "direction", "sectors": 4}])
add("E06_dir8", B, "+ direction octant (x H)", features=[PM, {"name": "direction", "sectors": 8}])
add("E06_lags", B, "+ references -20..+20 min", features=[PM, {"name": "lags", "steps": [-2, -1, 1, 2]}])
add("E06_stddev", B, "+ H turbulence (stddev)", features=[PM, "h_stddev"])
add("E06_temp", B, "+ temperature H, O", features=[PM, "temperature"])
add("E06_pressure", B, "+ air pressure H", features=[PM, "pressure"])

# E07 calibrated uncertainty: out-of-sample (backtest) residuals instead of training residuals
add("E07_gauss_oos", B, "95% band from out-of-sample sigma", interval={"name": "gaussian", "calib": "oos"})
add("E07_conformal", B, "conformal band (empirical quantiles, OOS)", interval={"name": "conformal", "calib": "oos"})
add("E07_conformal_bins", B, "conformal band per wind-speed bin (5, OOS)",
    interval={"name": "conformal", "bins": 5, "calib": "oos"})

# E08 challengers: how much does the explainable model leave on the table?
add("E08_ridge_refs", B, "ridge on all 7 references", features=["all_refs"], model={"name": "ridge", "alpha": 1.0})
add("E08_hgb", B, "gradient boosting, refs + hour + direction + stddev",
    features=["all_refs", PM, {"name": "hour"}, {"name": "direction"}, "h_stddev"], model={"name": "hgb"})

# Round 2: stack the round-1 winners (direction, lags/smoothing, binned sigma, E[P(v)])
SB = {"name": "gaussian", "bins": 5}
add("E09_power_gauss_bins", B, "E[P(v)] with sigma per wind-speed bin", interval=SB, power={"mode": "expected_gauss"})
for nb in (3, 10):
    add(f"E09_power_gauss_bins{nb}", B, f"E[P(v)] with sigma per wind-speed bin ({nb})",
        interval={"name": "gaussian", "bins": nb}, power={"mode": "expected_gauss"})
DIR8 = {"name": "direction", "sectors": 8}
LAGS = {"name": "lags", "steps": [-2, -1, 1, 2]}
add("E10_dir8_lags", B, "dir octant + lags", features=[PM, DIR8, LAGS])
add("E10_dir_lags", B, "dir sin/cos + lags", features=[PM, {"name": "direction"}, LAGS])
add("E10_dir8_smooth60", B, "dir octant on 60-min smoothed refs",
    features=[{"name": "park_means_smooth", "steps": 6}, DIR8])
add("E11_dir8_lags_bins", "E10_dir8_lags", "dir octant + lags, sigma per bin", interval=SB)
add("E11_dir8_lags_power", "E10_dir8_lags", "dir octant + lags, sigma per bin, E[P(v)]",
    interval=SB, power={"mode": "expected_gauss"})
add("E11_dir_lags_power", "E10_dir_lags", "dir sin/cos + lags, sigma per bin, E[P(v)]",
    interval=SB, power={"mode": "expected_gauss"})

BEST = "E12_dir8_smooth60_power"


def get_config(config: str | dict) -> dict:
    """Accept an id, "best", or a config dict."""
    if isinstance(config, dict):
        return config
    return CONFIGS[BEST if config == "best" else config]

# Explainability check: smoothing instead of lags keeps one coefficient per park
add("E12_dir8_smooth60_power", "E10_dir8_smooth60", "dir octant on 60-min smoothed refs, sigma per bin, E[P(v)]",
    interval=SB, power={"mode": "expected_gauss"})
add("E12_dir8_smooth30_power", "E12_dir8_smooth60_power", "dir octant on 30-min smoothed refs, sigma per bin, E[P(v)]",
    features=[{"name": "park_means_smooth", "steps": 3}, DIR8])

# Round 3: direction completeness, weather-similarity weights, automatic reference selection
BST = "E12_dir8_smooth60_power"
SM6 = {"name": "park_means_smooth", "steps": 6}
add("E13_dir8_smoothint", BST, "dir octant: H interaction terms use the 60-min smoothed H",
    features=[SM6, {"name": "direction", "sectors": 8, "smooth_steps": 6}])
add("E13_dir8_O", "E13_dir8_smoothint", "dir octant + per-sector O slope (smoothed)",
    features=[SM6, {"name": "direction", "sectors": 8, "smooth_steps": 6, "o_factor": True}])
add("E13_dir4_O", "E13_dir8_O", "dir quadrant + per-sector O slope (smoothed)",
    features=[SM6, {"name": "direction", "sectors": 4, "smooth_steps": 6, "o_factor": True}])
for _f in (0.2, 0.5):
    add(f"E14_dirmatch_f0{int(_f * 10)}", BST, f"train weights match the outage's wind-direction mix (floor {_f})",
        weights={"name": "direction_match", "sectors": 8, "floor": _f})
add("E14_dirspeed_match", BST, "train weights match the outage's direction x H-speed tercile mix (floor 0.2)",
    weights={"name": "direction_match", "sectors": 8, "floor": 0.2, "speed_bins": 3})
_RS = {"name": "all_refs_smooth", "steps": 6}
add("E15_refsel", BST, "automatic reference selection (forward CV) on 7 smoothed refs",
    features=[_RS], model={"name": "ols_forward"})
add("E15_refsel_dir8", BST, "automatic reference selection + direction octants",
    features=[_RS, DIR8], model={"name": "ols_forward"})
