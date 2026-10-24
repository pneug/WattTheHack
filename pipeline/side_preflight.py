"""Shape-only TabPFN cost/quota check; never fit or predict a model.

Run with the isolated client environment:
    .venv/tabpfn/Scripts/python.exe -m pipeline.side_preflight S03_tabpfn35_refs
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import core, evaluate, stages
from pipeline.experiments import CONFIGS, get_config
from pipeline.tabpfn_model import ROOT, _load_api_token
from preprocessing import TARGETS


def build_plan(config, data=None) -> dict:
    """Match backtest completeness, weighting, calibration, and batch sizes locally."""
    cfg = get_config(config)
    spec = cfg["model"]
    if spec["name"] != "tabpfn_api" or spec.get("model_path", "v3.5_default") != "v3.5_default":
        raise ValueError("Preflight currently supports the v3.5_default TabPFN API model")
    data = core.load_data() if data is None else data
    X = stages.build_X(data, cfg["features"])
    valid = X.notna().all(axis=1).to_numpy()
    cap, reserve = spec.get("max_train_rows", 4096), spec.get("calibration_rows", 512)
    batch = spec.get("batch_size", 4096)
    if min(cap, reserve, batch) < 1:
        raise ValueError("Context, calibration and batch sizes must be positive")
    shapes, folds = Counter(), []
    for name, start, end in evaluate.windows(data):
        grid = pd.date_range(start, end, freq="10min", inclusive="left")
        known = data["wind_speed"][TARGETS].reindex(grid)
        if name != "val" and known.notna().to_numpy().mean() < evaluate.MIN_KNOWN:
            continue
        test = core.in_window(X.index, (start, end))
        weights = stages.sample_weights(X.index, (start, end), cfg["weights"], data)
        for target in TARGETS:
            y = data["wind_speed"][target].reindex(X.index)
            train = ~test & valid & y.notna().to_numpy() & (weights > 0)
            n = int(train.sum())
            if n < 3:
                raise ValueError("Not enough training rows for reserved calibration")
            if not np.all(weights[train] == weights[train][0]):
                raise ValueError("TabPFN API requires uniform positive training weights")
            calibration = min(reserve, max(1, n // 5), n - 2)
            context = min(cap, n - calibration)
            outage = int((test & valid).sum())
            for count in (calibration, outage):
                for offset in range(0, count, batch):
                    shapes[(context, min(batch, count - offset), X.shape[1])] += 1
            folds.append(dict(window=name, target=target, training_rows=n,
                              context_rows=context, calibration_rows=calibration,
                              outage_query_rows=outage))
    return {
        "model_version": "v3.5", "model_path": "v3.5_default",
        "n_estimators": spec.get("n_estimators", 8),
        "n_features": X.shape[1], "feature_columns": list(X.columns),
        "windows": len({f["window"] for f in folds}),
        "backtest_windows": len({f["window"] for f in folds if f["window"] != "val"}),
        "fit_calls": len(folds), "predict_calls": sum(shapes.values()),
        "query_rows": sum(f["calibration_rows"] + f["outage_query_rows"] for f in folds),
        "calibration_query_rows": sum(f["calibration_rows"] for f in folds),
        "outage_query_rows": sum(f["outage_query_rows"] for f in folds),
        "shape_pairs": [dict(context_rows=a, query_rows=b, features=c, calls=count)
                        for (a, b, c), count in sorted(shapes.items())],
        "folds": folds,
    }


def quote_plan(plan: dict, client=None) -> dict:
    """Only zero-filled shape surrogates reach estimate_cost; no turbine data."""
    client = importlib.import_module("tabpfn_client") if client is None else client
    token = _load_api_token()
    quotes, totals = [], {"cache_predict": 0.0, "predict": 0.0}
    try:
        client.set_access_token(token)
        usage = str(client.get_api_usage()).replace(token, "[redacted]")
        # Quote the largest context for each query shape: a conservative bound
        # avoids one network quote for every slightly different training count.
        bound = max(s["context_rows"] for s in plan["shape_pairs"])
        counts = Counter()
        for shape in plan["shape_pairs"]:
            counts[(shape["query_rows"], shape["features"])] += shape["calls"]
        for (query_rows, features), calls in sorted(counts.items()):
            shape = dict(context_rows=bound, query_rows=query_rows, features=features, calls=calls)
            train = np.zeros((shape["context_rows"], shape["features"]), dtype=np.float32)
            query = np.zeros((shape["query_rows"], shape["features"]), dtype=np.float32)
            for operation in totals:
                response = client.estimate_cost(
                    train, query, model_version=plan["model_version"],
                    operation=operation, n_estimators=plan["n_estimators"],
                )
                record = response.model_dump(mode="json") if hasattr(response, "model_dump") else dict(response)
                cost = float(record["estimated_cost"])
                if not np.isfinite(cost) or cost < 0:
                    raise ValueError("Invalid estimate")
                totals[operation] += cost * shape["calls"]
                quotes.append({**shape, "operation": operation, "estimated_cost": cost,
                               "pricing_version": record.get("pricing_version"),
                               "inputs": record.get("inputs"),
                               "total_estimated_cost": cost * shape["calls"]})
    except Exception as exc:
        raise RuntimeError(
            f"TabPFN shape-only preflight failed ({type(exc).__name__}); client details withheld"
        ) from None
    return {"usage": usage, "client_version": getattr(client, "__version__", "unknown"),
            "quotes": quotes, "total_estimated_cost": totals,
            "pricing_versions": sorted({str(q["pricing_version"]) for q in quotes}),
            "estimate_calls": len(quotes),
            "note": "Conservative estimates use the largest context for every query shape. "
                    "Cache hits and account/model eligibility are not guaranteed. "
                    "Standard fits/uploads have no separate token charge. No inference was requested."}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", nargs="?", default="S03_tabpfn35_refs", choices=CONFIGS)
    args = parser.parse_args(argv)
    try:
        plan = build_plan(args.experiment)
        result = quote_plan(plan)
        report = {"experiment": args.experiment,
                  "created_at": datetime.now(timezone.utc).isoformat(),
                  "plan": plan, **result}
        path = ROOT / "experiments" / "runs" / f"{args.experiment}_preflight.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"{args.experiment}: {plan['backtest_windows']} backtests + validation; "
              f"{plan['fit_calls']} fits, {plan['predict_calls']} predicts, "
              f"{plan['query_rows']} query rows, {plan['n_features']} features")
        print(f"Cost estimates: cached={result['total_estimated_cost']['cache_predict']:,.0f}; "
              f"standard/fallback={result['total_estimated_cost']['predict']:,.0f}; "
              f"pricing={','.join(result['pricing_versions'])}")
        print(f"Quota: {result['usage']}")
        print(f"Saved {path.relative_to(ROOT)}")
        return 0
    except Exception as exc:
        print(f"Preflight stopped ({type(exc).__name__}); no inference requested. "
              "Check client installation, token, model settings and API availability.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
