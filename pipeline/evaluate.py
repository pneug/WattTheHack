"""Evaluation harness: judge-style metrics and a 3-day block backtest anchored at the hidden window."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline import core, stages
from preprocessing import HIDDEN, STEP_HOURS, TARGETS, VAL, load_power_curve, wind_to_power

BLOCK = pd.Timedelta(days=3)
MIN_KNOWN = 0.5  # min share of steps with a known T actual (all targets combined)


def judge_metrics(pred, kw, actual, curve, lo=None, hi=None) -> dict:
    """Metrics on steps where pred and actual are both known. ws in m/s, power in kW, energy in kWh."""
    pred, kw, actual = (np.asarray(x, dtype=float) for x in (pred, kw, actual))
    ok = ~np.isnan(pred) & ~np.isnan(actual)
    n = int(ok.sum())
    if n == 0:
        return dict(n=0)
    err = pred[ok] - actual[ok]
    p_act = wind_to_power(actual[ok], curve)
    perr = kw[ok] - p_act
    e_pred = float(np.nansum(kw[ok]) * STEP_HOURS)
    e_act = float(p_act.sum() * STEP_HOURS)
    m = dict(n=n, ws_rmse=float(np.sqrt(np.mean(err ** 2))), ws_mae=float(np.mean(np.abs(err))),
             p_rmse=float(np.sqrt(np.nanmean(perr ** 2))), p_mae=float(np.nanmean(np.abs(perr))),
             e_pred_kwh=e_pred, e_act_kwh=e_act,
             e_dev_pct=100 * (e_pred / e_act - 1) if e_act > 0 else np.nan)
    if lo is not None and hi is not None:
        l, h = np.asarray(lo, dtype=float)[ok], np.asarray(hi, dtype=float)[ok]
        m["cover"] = float(np.mean((actual[ok] >= l) & (actual[ok] <= h)))
        m["width"] = float(np.mean(h - l))
    return m


def windows(data=None, step_days: int = 3) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    """3-day blocks anchored at HIDDEN[0] (excluding the hidden block), plus "val". End is exclusive."""
    data = core.load_data() if data is None else data
    idx = data["wind_speed"].index
    first, last = idx.min().normalize(), idx.max() + pd.Timedelta(minutes=10)
    anchor = HIDDEN[0]
    step = pd.Timedelta(days=step_days)
    out = []
    s = anchor - ((anchor - first) // step) * step
    while s + BLOCK <= last:
        if not _overlaps(s, s + BLOCK, *HIDDEN):
            out.append((f"bt_{s:%m%d}", s, s + BLOCK))
        s += step
    out.append(("val", *VAL))
    return out


def _overlaps(a0, a1, b0, b1) -> bool:
    return a0 < b1 and b0 < a1


def _oos_recalibrate(res: dict, config: dict, curve) -> dict:
    """Leave-one-window-out calibration. res: {window: (start, end, {target: df})}.
    Recompute lo, hi, sigma, kw of every window from the residuals of the bt windows that do not
    overlap it (for "val": the bt windows not overlapping 4-9 Sep)."""
    spec = {k: v for k, v in config["interval"].items() if k != "calib"}
    for name, (s, e, per_t) in res.items():
        others = [d for n2, (s2, e2, d) in res.items()
                  if n2.startswith("bt_") and not _overlaps(s, e, s2, e2)]
        for t, df in per_t.items():
            pool = pd.concat([d[t][["pred", "actual"]].dropna() for d in others])
            resid, fitted = (pool["actual"] - pool["pred"]).values, pool["pred"].values
            lo, hi, sigma = stages.band(df["pred"], resid, fitted, spec)
            df["lo"], df["hi"], df["sigma"] = lo, hi, sigma
            df["kw"] = stages.to_power(df["pred"], sigma, resid, curve, config["power"])
    return res


def backtest(config, data=None, curve=None, step_days: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = core.get_config(config)
    data = core.load_data() if data is None else data
    curve = load_power_curve() if curve is None else curve
    res = {}
    for name, s, e in windows(data, step_days):
        assert not _overlaps(s, e, *HIDDEN), f"{name} overlaps the hidden window"
        known = data["wind_speed"][TARGETS].reindex(pd.date_range(s, e, freq="10min", inclusive="left"))
        if name != "val" and known.notna().to_numpy().mean() < MIN_KNOWN:
            continue
        res[name] = (s, e, core.fit_predict(data, cfg, (s, e), curve=curve))
    if cfg["interval"].get("calib") == "oos":
        res = _oos_recalibrate(res, cfg, curve)

    rows, steps = [], []
    for name, (s, e, per_t) in res.items():
        kind = "val" if name == "val" else "bt"
        park = dict(e_pred_kwh=0.0, e_act_kwh=0.0)
        for t, df in per_t.items():
            m = judge_metrics(df["pred"], df["kw"], df["actual"], curve, df["lo"], df["hi"])
            rows.append(dict(window=name, kind=kind, target=t, start=s, **m))
            park["e_pred_kwh"] += m.get("e_pred_kwh", 0.0)
            park["e_act_kwh"] += m.get("e_act_kwh", 0.0)
            st = df[["pred", "lo", "hi", "sigma", "kw", "actual"]].copy()
            st.insert(0, "target", t)
            st.insert(0, "window", name)
            steps.append(st)
        park["e_dev_pct"] = 100 * (park["e_pred_kwh"] / park["e_act_kwh"] - 1) if park["e_act_kwh"] > 0 else np.nan
        rows.append(dict(window=name, kind=kind, target="park", start=s, **park))
    steps = pd.concat(steps)
    steps.index.name = "timestamp"
    return pd.DataFrame(rows), steps.reset_index()


def _pooled_rmse(d: pd.DataFrame, col: str) -> float:
    return float(np.sqrt((d["n"] * d[col] ** 2).sum() / d["n"].sum()))


def _pooled_mean(d: pd.DataFrame, col: str) -> float:
    return float((d["n"] * d[col]).sum() / d["n"].sum())


def summarize(per_window: pd.DataFrame) -> dict:
    tw = per_window[per_window["target"] != "park"]
    bt, val = tw[tw["kind"] == "bt"], tw[tw["kind"] == "val"]
    park = per_window[(per_window["target"] == "park") & (per_window["kind"] == "bt")]
    s = dict(
        bt_ws_rmse=_pooled_rmse(bt, "ws_rmse"), bt_ws_mae=_pooled_mean(bt, "ws_mae"),
        bt_p_rmse=_pooled_rmse(bt, "p_rmse"), bt_p_mae=_pooled_mean(bt, "p_mae"),
        bt_abs_edev=float(bt["e_dev_pct"].abs().mean()),
        bt_abs_edev_park=float(park["e_dev_pct"].abs().mean()),
    )
    for t in TARGETS:
        s[f"bt_edev_{t}"] = float(bt.loc[bt["target"] == t, "e_dev_pct"].mean())
    s["bt_edev_p90"] = float(np.percentile(bt["e_dev_pct"].abs().dropna(), 90))
    s["bt_cover"] = _pooled_mean(bt, "cover") if "cover" in bt else np.nan
    s["bt_width"] = _pooled_mean(bt, "width") if "width" in bt else np.nan
    if len(val):
        s["val_ws_rmse"], s["val_p_rmse"] = _pooled_rmse(val, "ws_rmse"), _pooled_rmse(val, "p_rmse")
        for t in TARGETS:
            s[f"val_edev_{t}"] = float(val.loc[val["target"] == t, "e_dev_pct"].mean())
        vp = per_window[(per_window["target"] == "park") & (per_window["kind"] == "val")]
        s["val_edev_park"] = float(vp["e_dev_pct"].iloc[0])
        s["val_cover"] = _pooled_mean(val, "cover") if "cover" in val else np.nan
    s["n_windows"] = int(bt["window"].nunique())
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}


def energy_interval(e_pred_kwh: float, config="best", target: str = "park", level: float = 0.95) -> tuple[float, float]:
    """kWh interval for an outage's predicted energy, from the errors in the backtest windows.

    Reads experiments/runs/<id>.json (run `python -m pipeline.run <id>` first). In those simulated
    3-day outages the prediction was off by e_dev_pct; the interval inverts the central `level` of
    those errors: actual = predicted / (1 + dev). Skewed on purpose: calm windows give large relative errors.
    """
    import json
    from pathlib import Path

    from pipeline.experiments import BEST
    exp_id = BEST if config == "best" else config
    path = Path(__file__).resolve().parents[1] / "experiments" / "runs" / f"{exp_id}.json"
    pw = pd.DataFrame(json.loads(path.read_text())["per_window"])
    dev = pw.loc[(pw["kind"] == "bt") & (pw["target"] == target), "e_dev_pct"].dropna() / 100
    lo_q, hi_q = np.quantile(dev, [(1 - level) / 2, (1 + level) / 2])
    return e_pred_kwh / (1 + hi_q), e_pred_kwh / (1 + lo_q)
