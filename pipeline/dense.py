"""Dense backtest (3-day windows starting every day) plus paired / regime-matched analysis.

CLI:  python -m pipeline.dense <id> [<id> ...]            run and store experiments/dense/<id>.json
      python -m pipeline.dense --analyze <id> [<id> ...]  paired, regime and interval report
Writes only inside experiments/dense/ (never experiments/results.csv, runs/ or LOG.md).
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import evaluate, stages

ROOT = Path(__file__).resolve().parents[1]
DENSE = ROOT / "experiments" / "dense"
RUNS = ROOT / "experiments" / "runs"
RESULTS = DENSE / "results.csv"
E_PRED_HIDDEN_KWH = 322400.0  # BEST park prediction for the hidden window
STEP = pd.Timedelta(minutes=10)


def run_one(exp_id: str, data=None) -> dict:
    from pipeline.experiments import CONFIGS
    cfg = CONFIGS[exp_id]
    t0 = time.time()
    per_window, _ = evaluate.backtest(cfg, data, step_days=1)
    summary = evaluate.summarize(per_window)
    DENSE.mkdir(parents=True, exist_ok=True)
    recs = json.loads(per_window.to_json(orient="records", date_format="iso"))
    (DENSE / f"{exp_id}.json").write_text(json.dumps({"config": cfg, "summary": summary, "per_window": recs},
                                                     indent=1), encoding="utf-8")
    row = {"run_at": datetime.now().isoformat(timespec="seconds"), "id": exp_id, "step_days": 1, **summary}
    new = pd.DataFrame([row])
    if RESULTS.exists():
        new = pd.concat([pd.read_csv(RESULTS), new], ignore_index=True)
    new.to_csv(RESULTS, index=False)
    print(f"{exp_id:<26} {time.time() - t0:5.0f}s  n={summary['n_windows']}  park |Edev| {summary['bt_abs_edev_park']:.2f}%  "
          f"wsRMSE {summary['bt_ws_rmse']:.3f}  PRMSE {summary['bt_p_rmse']:.1f}", flush=True)
    return summary


def load(exp_id: str, dense: bool = True) -> pd.DataFrame:
    path = (DENSE if dense else RUNS) / f"{exp_id}.json"
    pw = pd.DataFrame(json.loads(path.read_text())["per_window"])
    pw["start"] = pd.to_datetime(pw["start"])
    return pw


def park(pw: pd.DataFrame) -> pd.DataFrame:
    return pw[(pw["kind"] == "bt") & (pw["target"] == "park")].set_index("window")


def paired(a_id: str, b_id: str, verbose: bool = True) -> dict:
    """Paired comparison of |park dev| over the shared dense bt windows. diff = mean(|b|) - mean(|a|), so
    negative means b is better. Block-robust SE: windows split into 3 non-overlapping offset classes
    (start day mod 3); the paired mean difference per class; report mean and spread across classes."""
    pa, pb = park(load(a_id)), park(load(b_id))
    shared = pa.index.intersection(pb.index)
    aa, bb = pa.loc[shared, "e_dev_pct"].abs(), pb.loc[shared, "e_dev_pct"].abs()
    d = (bb - aa).dropna()
    start = pa.loc[d.index, "start"]
    cls = (start - start.min()).dt.days % 3
    cm = np.array([d[cls == k].mean() for k in range(3)])
    out = dict(a=a_id, b=b_id, n=len(d), mean_abs_a=aa.mean(), mean_abs_b=bb.mean(), diff=d.mean(),
               naive_se=d.std(ddof=1) / np.sqrt(len(d)), class_means=cm, class_mean=cm.mean(),
               class_sd=cm.std(ddof=1), block_se=cm.std(ddof=1) / np.sqrt(3),
               b_win_share=float((d < 0).mean()))
    if verbose:
        print(f"{b_id} vs {a_id}: n={out['n']} |a|={out['mean_abs_a']:.2f} |b|={out['mean_abs_b']:.2f} "
              f"diff={out['diff']:+.2f} naiveSE={out['naive_se']:.2f} classes={np.round(cm, 2)} "
              f"classSD={out['class_sd']:.2f} blockSE={out['block_se']:.2f} b_wins={out['b_win_share']:.2f}")
    return out


def regime_tags(data, starts, length=evaluate.BLOCK) -> pd.DataFrame:
    """Per window (index = start): sw_share (circular-mean H direction in 180-270 deg) and H mean wind speed."""
    wd = np.deg2rad(data["wind_direction"][stages.H_COLS])
    s, c = np.sin(wd).mean(axis=1, skipna=True), np.cos(wd).mean(axis=1, skipna=True)
    deg = (np.rad2deg(np.arctan2(s, c)) % 360).where(s.notna())
    sw = ((deg >= 180) & (deg <= 270)).astype(float).where(deg.notna())
    H = stages.park_means(data["wind_speed"])["H"]
    rows = {st: dict(sw_share=sw[st:st + length - STEP].mean(), h_mean=H[st:st + length - STEP].mean())
            for st in starts}
    return pd.DataFrame(rows).T


def metrics(pw: pd.DataFrame, names) -> dict:
    names = list(names)
    p = park(pw).loc[names]
    t = pw[(pw["kind"] == "bt") & (pw["target"] != "park") & pw["window"].isin(names)]
    w = t["n"]
    return dict(n=len(p), abs_edev=p["e_dev_pct"].abs().mean(), bias=p["e_dev_pct"].mean(),
                ws_rmse=float(np.sqrt((w * t["ws_rmse"] ** 2).sum() / w.sum())),
                p_rmse=float(np.sqrt((w * t["p_rmse"] ** 2).sum() / w.sum())))


def hidden_like(tags: pd.DataFrame, sw=0.6, h=5.5) -> pd.Series:
    return (tags["sw_share"] >= sw) & (tags["h_mean"] >= h)


def _quant_interval(dev_pct, e_pred, level=0.95):
    lo_q, hi_q = np.quantile(np.asarray(dev_pct) / 100, [(1 - level) / 2, (1 + level) / 2])
    return e_pred / (1 + hi_q), e_pred / (1 + lo_q)


def loo_coverage(exp_id: str, level=0.95) -> dict:
    """For each dense bt window, quantiles of the park dev of windows not overlapping it; is its actual inside?"""
    p = park(load(exp_id)).dropna(subset=["e_dev_pct"])
    inside = []
    for _, r in p.iterrows():
        oth = p[(p["start"] <= r["start"] - evaluate.BLOCK) | (p["start"] >= r["start"] + evaluate.BLOCK)]
        lo, hi = _quant_interval(oth["e_dev_pct"], r["e_pred_kwh"], level)
        inside.append(lo <= r["e_act_kwh"] <= hi)
    return dict(n=len(inside), coverage=float(np.mean(inside)))


def analyze(ids: list[str]) -> None:
    from pipeline.core import load_data
    data = load_data()
    dw = {i: load(i) for i in ids}
    sw_ = {i: load(i, dense=False) for i in ids}
    ref = ids[0]
    pd_, ps_ = park(dw[ref]).sort_values("start"), park(sw_[ref]).sort_values("start")
    tags_d = regime_tags(data, list(pd_["start"]))
    tags_s = regime_tags(data, list(ps_["start"]))
    tags_d.index, tags_s.index = pd_.index, ps_.index  # index by window name
    sw_th = 0.6
    hl_d = hidden_like(tags_d, sw_th)
    if hl_d.sum() < 8:
        sw_th = 0.5
        hl_d = hidden_like(tags_d, sw_th)
        print(f"NOTE: <8 dense windows at SW>=0.6, relaxed to SW>={sw_th}")
    hl_s = hidden_like(tags_s, sw_th)
    print(f"hidden-like (SW>={sw_th}, H>=5.5): dense {int(hl_d.sum())}/{len(tags_d)}, "
          f"3-day-step {int(hl_s.sum())}/{len(tags_s)} (also at 0.6: dense {int(hidden_like(tags_d).sum())}, "
          f"step3 {int(hidden_like(tags_s).sum())})")
    sets = {"dense_all": (dw, list(tags_d.index)), "dense_hidlike": (dw, list(tags_d.index[hl_d])),
            "step3_all": (sw_, list(tags_s.index)), "step3_hidlike": (sw_, list(tags_s.index[hl_s]))}
    print("\nconfig | set | n | park|Edev| | park bias | wsRMSE | PRMSE")
    for i in ids:
        for k, (pws, names) in sets.items():
            m = metrics(pws[i], names)
            print(f"{i} | {k} | {m['n']} | {m['abs_edev']:.2f} | {m['bias']:+.2f} | {m['ws_rmse']:.3f} | {m['p_rmse']:.1f}")
    print("\nPaired, all dense (diff = |b|-|a|, negative = b better)")
    for a in ("E00_baseline", "E12_dir8_smooth60_power"):
        for b in ids:
            if b != a and a in ids:
                paired(a, b)
    print("\nPaired on hidden-like dense windows (naive SE only; windows few and overlapping)")
    hn = sets["dense_hidlike"][1]
    for a in ("E00_baseline", "E12_dir8_smooth60_power"):
        for b in ids:
            if b != a and a in ids:
                d = (park(dw[b]).loc[hn, "e_dev_pct"].abs() - park(dw[a]).loc[hn, "e_dev_pct"].abs()).dropna()
                print(f"{b} vs {a}: n={len(d)} diff={d.mean():+.2f} naiveSE={d.std(ddof=1) / np.sqrt(len(d)):.2f} "
                      f"b_wins={(d < 0).mean():.2f}")
    best = "E12_dir8_smooth60_power"
    if best in ids:
        p = park(dw[best]).dropna(subset=["e_dev_pct"])
        hl_names = [n for n in sets["dense_hidlike"][1] if n in p.index]
        print(f"\nBEST kWh 95% interval for e_pred={E_PRED_HIDDEN_KWH:.0f}")
        print("  all dense bt windows:", np.round(_quant_interval(p["e_dev_pct"], E_PRED_HIDDEN_KWH)))
        print("  hidden-like dense   :", np.round(_quant_interval(p.loc[hl_names, "e_dev_pct"], E_PRED_HIDDEN_KWH)),
              f"(n={len(hl_names)})")
        ps = park(sw_[best]).dropna(subset=["e_dev_pct"])
        hs = [n for n in sets["step3_hidlike"][1] if n in ps.index]
        print("  all 3-day-step      :", np.round(_quant_interval(ps["e_dev_pct"], E_PRED_HIDDEN_KWH)))
        print("  hidden-like 3d-step :", np.round(_quant_interval(ps.loc[hs, "e_dev_pct"], E_PRED_HIDDEN_KWH)),
              f"(n={len(hs)})")
        print("  LOO coverage (dense, all-windows interval):", loo_coverage(best))
    print("\nWind terciles (dense, mean |park dev| per tercile of window mean H wind)")
    for i in ids:
        p = park(dw[i])
        h = tags_d["h_mean"].reindex(p.index)
        q = pd.qcut(h, 3, labels=["low", "mid", "high"])
        edges = np.round(np.nanquantile(h, [0, 1 / 3, 2 / 3, 1]), 2)
        g = p["e_dev_pct"].abs().groupby(q, observed=True).agg(["mean", "count"])
        print(i, "edges", edges, {k: (round(v["mean"], 2), int(v["count"])) for k, v in g.iterrows()})


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    if args[0] == "--analyze":
        analyze(args[1:])
    else:
        from pipeline.core import load_data
        d = load_data()
        for i in args:
            run_one(i, d)
