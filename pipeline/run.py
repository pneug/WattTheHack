"""CLI: python -m pipeline.run E00_baseline [E01_x ...]  |  python -m pipeline.run --all"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from pipeline import evaluate, report
from pipeline.experiments import CONFIGS

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments" / "runs"
RESULTS = report.RESULTS


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return ""


def run_one(exp_id: str, data=None) -> dict:
    cfg = CONFIGS[exp_id]
    t0 = time.time()
    per_window, steps = evaluate.backtest(cfg, data)
    summary = evaluate.summarize(per_window)
    clean = {k: v for k, v in cfg.items() if k not in ("desc", "parent")}
    row = {"run_at": datetime.now().isoformat(timespec="seconds"), "id": exp_id, "parent": cfg.get("parent") or "",
           "desc": cfg.get("desc", ""), "git_sha": _git_sha(), "config_json": json.dumps(clean), **summary}
    new = pd.DataFrame([row])
    if RESULTS.exists():
        new = pd.concat([pd.read_csv(RESULTS), new], ignore_index=True)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(RESULTS, index=False)
    RUNS.mkdir(parents=True, exist_ok=True)
    recs = json.loads(per_window.to_json(orient="records", date_format="iso"))
    (RUNS / f"{exp_id}.json").write_text(json.dumps({"config": cfg, "summary": summary, "per_window": recs},
                                                    indent=1), encoding="utf-8")
    steps.to_csv(RUNS / f"{exp_id}_steps.csv", index=False)
    s = summary
    print(f"{exp_id:<24} {time.time() - t0:4.0f}s  bt wsRMSE {s['bt_ws_rmse']:.3f}  PRMSE {s['bt_p_rmse']:.1f}  "
          f"|Edev| {s['bt_abs_edev']:.2f}%  park {s['bt_abs_edev_park']:.2f}%  "
          f"bias {s['bt_edev_T1']:+.1f}/{s['bt_edev_T2']:+.1f}/{s['bt_edev_T3']:+.1f}  "
          f"val wsRMSE {s['val_ws_rmse']:.3f}  val park {s['val_edev_park']:+.2f}%  cover {s['bt_cover']:.2f}")
    return summary


def main(argv: list[str]) -> None:
    ids = list(CONFIGS) if "--all" in argv else [a for a in argv if not a.startswith("-")]
    if not ids:
        sys.exit("usage: python -m pipeline.run <id> [<id> ...] | --all")
    unknown = [i for i in ids if i not in CONFIGS]
    if unknown:
        sys.exit(f"unknown ids {unknown}; known: {list(CONFIGS)}")
    from pipeline.core import load_data
    data = load_data()
    for i in ids:
        run_one(i, data)
    report.update_log()


if __name__ == "__main__":
    main(sys.argv[1:])
