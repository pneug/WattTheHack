"""Pipeline core: one config in, predictions for an outage window out.

    from pipeline import load_data, run_outage
    res = run_outage(load_data(), "2026-09-01", "2026-09-04", config="best")

This file only wires the stages (pipeline/stages.py) together and enforces the leakage rules:
the outage window is never in the fit, and park T is never a feature.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from pipeline import stages
from pipeline.experiments import get_config
from preprocessing import SENSORS, STEP_HOURS, TARGETS, load_clean, load_power_curve, to_wide


@lru_cache(maxsize=1)
def _load() -> tuple:
    long = load_clean()
    return tuple((var, to_wide(long, var)) for var in SENSORS)


def load_data() -> stages.Data:
    """All sensors as wide tables: {"wind_speed": timestamp x turbine, "wind_direction": ..., ...}."""
    return dict(_load())


def in_window(index: pd.DatetimeIndex, window) -> np.ndarray:
    start, end = (pd.Timestamp(x) for x in window)
    return np.asarray((index >= start) & (index < end))


def fit_predict(data: stages.Data, config, window, targets=TARGETS,
                curve: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Fit one model per target on all known steps outside `window` [start, end), predict inside it.

    Returns {target: DataFrame} on the window's steps with columns
    pred, lo, hi, sigma [m/s], kw [kW], actual [m/s, NaN if unknown].
    df.attrs holds "model" (describe()), "resid" and "fitted" (training residuals and predictions).
    """
    cfg = get_config(config)
    curve = load_power_curve() if curve is None else curve
    X = stages.build_X(data, cfg["features"])
    test = in_window(X.index, window)
    w = stages.sample_weights(X.index, window, cfg["weights"], data)
    x_ok = X.notna().all(axis=1).values
    out = {}
    for t in targets:
        y = data["wind_speed"][t].reindex(X.index)
        train = ~test & x_ok & y.notna().values & (w > 0)
        assert not (train & test).any(), "outage window leaked into the fit"
        model = stages.make_model(cfg["model"]).fit(X[train], y[train], w[train])
        # Context-based models reserve training-only rows for honest residuals:
        # predicting their own context can severely underestimate uncertainty.
        calib = getattr(model, "calibration_mask_", None)
        if calib is None:
            calib_X, calib_y = X[train], y[train]
        else:
            calib = np.asarray(calib)
            if calib.dtype != bool or calib.shape != (int(train.sum()),) or not calib.any():
                raise ValueError("model calibration_mask_ must select passed training rows")
            calib_X, calib_y = X[train].iloc[calib], y[train].iloc[calib]
        fitted = model.predict(calib_X)
        resid = calib_y.values - fitted

        pred = pd.Series(np.nan, index=X.index[test])
        ok = x_ok[test]
        pred[ok] = np.clip(model.predict(X[test][ok]), 0, None)
        if not ok.all():  # an extra feature is missing: fall back to the plain park-means OLS
            pred = _fallback(data, X.index, test, train, y, w, pred)
        # training residuals calibrate here; "calib": "oos" is applied later by evaluate.backtest
        spec = {k: v for k, v in cfg["interval"].items() if k != "calib"}
        lo, hi, sigma = stages.band(pred, resid, fitted, spec)
        kw = stages.to_power(pred, sigma, resid, curve, cfg["power"])

        df = pd.DataFrame({"pred": pred, "lo": lo, "hi": hi, "sigma": sigma, "kw": kw, "actual": y[test]})
        # lists, not arrays: pandas compares attrs with == when concatenating frames
        df.attrs = {"model": model.describe(), "resid": resid.tolist(), "fitted": fitted.tolist()}
        out[t] = df
    return out


def _fallback(data, index, test, train, y, w, pred: pd.Series) -> pd.Series:
    """Fill steps the main model cannot predict with OLS on the park means (fitted on the same rows)."""
    Xb = stages.build_X(data, ["park_means"]).reindex(index)
    xb_ok = Xb.notna().all(axis=1).values
    fit_rows = train & xb_ok
    base = stages.OLS().fit(Xb[fit_rows], y[fit_rows], w[fit_rows])
    fill = pred.isna().values & xb_ok[test]
    pred = pred.copy()
    pred[fill] = np.clip(base.predict(Xb[test][fill]), 0, None)
    return pred


def run_outage(data: stages.Data, start, end, config="best", curve: pd.DataFrame | None = None) -> pd.DataFrame:
    """App contract (same columns as baseline.run_outage, plus the band):
    <T>_pred, <T>_lo, <T>_hi [m/s], <T>_actual [m/s], <T>_loss_kwh [kWh per step]."""
    res = fit_predict(data, config, (start, end), curve=curve)
    cols = {}
    for t, df in res.items():
        cols[f"{t}_pred"], cols[f"{t}_lo"], cols[f"{t}_hi"] = df["pred"], df["lo"], df["hi"]
        cols[f"{t}_actual"] = df["actual"]
        cols[f"{t}_loss_kwh"] = df["kw"] * STEP_HOURS
    return pd.DataFrame(cols)
