"""Predict the hidden outage (1-3 Sep) with the BEST config, fitted on every known step (train + val).

    python -m pipeline.submit            # writes submission/hidden_predictions.csv
"""

from __future__ import annotations

from pathlib import Path

from pipeline import evaluate, load_data, run_outage
from pipeline.experiments import BEST
from preprocessing import HIDDEN, TARGETS

OUT = Path(__file__).resolve().parents[1] / "submission" / "hidden_predictions.csv"

if __name__ == "__main__":
    res = run_outage(load_data(), *HIDDEN, config="best")  # fit excludes only the hidden window
    res = res.drop(columns=[f"{t}_actual" for t in TARGETS])
    assert len(res) == 432 and res.filter(like="_pred").notna().all().all(), "missing hidden predictions"
    OUT.parent.mkdir(exist_ok=True)
    res.rename_axis("timestamp").to_csv(OUT, float_format="%.4f")

    total = sum(res[f"{t}_loss_kwh"].sum() for t in TARGETS)
    lo, hi = evaluate.energy_interval(total)
    print(f"{BEST}: wrote {len(res)} rows to {OUT}")
    for t in TARGETS:
        e = res[f"{t}_loss_kwh"].sum()
        tl, th = evaluate.energy_interval(e, target=t)
        print(f"  {t}: lost energy {e / 1000:,.1f} MWh (95%: {tl / 1000:,.1f}-{th / 1000:,.1f})")
    print(f"  park: {total / 1000:,.1f} MWh (95%: {lo / 1000:,.1f}-{hi / 1000:,.1f})")
