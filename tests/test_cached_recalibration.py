"""A cached OOS diagnostic must never request another remote prediction."""

import copy
import unittest
from unittest.mock import call, patch

import numpy as np
import pandas as pd

from pipeline import core, evaluate, stages
from pipeline.experiments import CONFIGS
from preprocessing import TARGETS


class CachedRecalibrationTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range("2026-01-01", periods=12, freq="10min")
        self.windows = [
            ("bt_a", self.index[0], self.index[4]),
            ("bt_b", self.index[4], self.index[8]),
            ("bt_c", self.index[8], self.index[-1] + pd.Timedelta(minutes=10)),
            ("val", self.index[6], self.index[10]),
        ]
        self.data = {"wind_speed": pd.DataFrame(7., index=self.index, columns=TARGETS)}
        self.curve = pd.DataFrame({"wind_speed": [0., 5., 10., 20.],
                                   "power": [0., 100., 900., 0.]})
        self.source = {
            "features": ["park_means"], "weights": {"name": "none"},
            "model": {"name": "tabpfn_api", "random_state": 0},
            "interval": {"name": "gaussian", "bins": 1},
            "power": {"mode": "expected_gauss"},
            "checkpoint": True, "side_experiment": True,
        }
        self.config = copy.deepcopy(self.source)
        self.config.update(prediction_cache_from="source",
                           interval={"name": "gaussian", "bins": 1, "calib": "oos"})
        residuals = {"bt_a": [1., 2., -1., -2.], "bt_b": [3., -3., 1., -1.],
                     "bt_c": [5., -5., 2., -2.], "val": [-50., -50., -50., -50.]}
        self.results = {}
        for name, start, end in self.windows:
            index = self.index[(self.index >= start) & (self.index < end)]
            per_target = {}
            for i, target in enumerate(TARGETS):
                pred = self.data["wind_speed"].loc[index, target] - residuals[name]
                lo, hi, sigma = stages.band(pred, [.1, -.1], [5., 6.], self.source["interval"])
                kw = stages.to_power(pred, sigma, [.1, -.1], self.curve, self.source["power"])
                frame = pd.DataFrame({"pred": pred, "lo": lo, "hi": hi,
                                      "sigma": sigma, "kw": kw,
                                      "actual": pred + residuals[name]})
                frame.attrs = {"model": {"type": "dummy"}, "resid": [.1, -.1], "fitted": [5., 6.]}
                per_target[target] = frame
            self.results[name] = per_target

    def backtest(self, config=None, source=None, missing=None):
        source = self.source if source is None else source
        config = self.config if config is None else config

        def load(name, start, end):
            return None if name == missing else copy.deepcopy(self.results[name])

        with patch.dict(CONFIGS, {"source": source}), \
                patch.object(evaluate, "windows", return_value=self.windows), \
                patch("pipeline.side_cache.SideBacktestCache") as cache_type, \
                patch.object(core, "fit_predict", side_effect=AssertionError("remote fit forbidden")) as fit:
            cache = cache_type.return_value
            cache.load.side_effect = load
            output = evaluate.backtest(config, self.data, self.curve)
            fit.assert_not_called()
            cache.save.assert_not_called()
            cache_type.assert_called_once_with(source, self.data, curve=self.curve)
            self.assertEqual(cache.load.call_args_list,
                             [call(*window) for window in self.windows])
        return output

    def test_reuses_source_predictions_and_only_nonoverlapping_bt_residuals(self):
        before = copy.deepcopy(self.results)
        per_window, steps = self.backtest()
        self.assertEqual(len(per_window), len(self.windows) * (len(TARGETS) + 1))
        power_changed = False
        for name, start, end in self.windows:
            donors = [n for n, s, e in self.windows
                      if n.startswith("bt_") and not evaluate._overlaps(start, end, s, e)]
            self.assertNotIn(name, donors)
            self.assertNotIn("val", donors)
            if name == "val":
                self.assertEqual(donors, ["bt_a"])
            for target in TARGETS:
                source = self.results[name][target]
                actual = steps[(steps.window == name) & (steps.target == target)].set_index("timestamp")
                self.assertTrue(actual.index.equals(source.index))
                np.testing.assert_array_equal(actual.pred, source.pred)
                np.testing.assert_array_equal(actual.actual, source.actual)
                pool = pd.concat([self.results[n][target][["pred", "actual"]] for n in donors])
                resid = (pool.actual - pool.pred).to_numpy()
                fitted = pool.pred.to_numpy()
                spec = {key: value for key, value in self.config["interval"].items() if key != "calib"}
                lo, hi, sigma = stages.band(source.pred, resid, fitted, spec)
                kw = stages.to_power(source.pred, sigma, resid, self.curve, self.config["power"])
                for column, expected in (("lo", lo), ("hi", hi), ("sigma", sigma), ("kw", kw)):
                    np.testing.assert_allclose(actual[column], expected)
                power_changed |= not np.allclose(actual.kw, source.kw)
                pd.testing.assert_frame_equal(self.results[name][target], before[name][target])
                self.assertEqual(self.results[name][target].attrs, before[name][target].attrs)
        self.assertTrue(power_changed, "expected Gaussian power must use recalibrated sigma")
        _, start, end = self.windows[-1]
        validation = (start, end, copy.deepcopy(self.results["val"]))
        with self.assertRaisesRegex(ValueError, "no nonoverlapping backtest windows"):
            evaluate._oos_recalibrate({"val": validation}, self.config, self.curve)
        empty = copy.deepcopy(self.results["bt_a"])
        for frame in empty.values():
            frame["actual"] = np.nan
        with self.assertRaisesRegex(ValueError, "no finite OOS calibration residuals"):
            evaluate._oos_recalibrate({"val": validation, "bt_a": (*self.windows[0][1:], empty)},
                                      self.config, self.curve)

    def test_missing_or_corrupt_source_checkpoint_fails_before_any_fit(self):
        # SideBacktestCache.load already maps corruption to the same None as a miss.
        with patch.dict(CONFIGS, {"source": self.source}), \
                patch.object(evaluate, "windows", return_value=self.windows), \
                patch("pipeline.side_cache.SideBacktestCache") as cache_type, \
                patch.object(core, "fit_predict", side_effect=AssertionError("remote fit forbidden")) as fit:
            cache_type.return_value.load.return_value = None
            with self.assertRaises((ValueError, RuntimeError, FileNotFoundError)):
                evaluate.backtest(self.config, self.data, self.curve)
            fit.assert_not_called()
            cache_type.return_value.save.assert_not_called()

    def test_rejects_incompatible_stages_before_any_fit(self):
        for stage, changed in (("features", ["all_refs"]),
                               ("weights", {"name": "window", "days": 7}),
                               ("model", {"name": "tabpfn_api", "random_state": 99})):
            with self.subTest(stage=stage):
                cfg = copy.deepcopy(self.config)
                cfg[stage] = changed
                with patch.dict(CONFIGS, {"source": self.source}), \
                        patch.object(evaluate, "windows", return_value=self.windows), \
                        patch("pipeline.side_cache.SideBacktestCache") as cache_type, \
                        patch.object(core, "fit_predict", side_effect=AssertionError("remote fit forbidden")) as fit:
                    cache_type.return_value.load.side_effect = lambda name, s, e: copy.deepcopy(self.results[name])
                    with self.assertRaises(ValueError):
                        evaluate.backtest(cfg, self.data, self.curve)
                    fit.assert_not_called()

    def test_rejects_alias_chains_source_oos_or_target_without_oos(self):
        scenarios = []
        source = copy.deepcopy(self.source)
        source["prediction_cache_from"] = "deeper"
        scenarios.append((self.config, source))
        source = copy.deepcopy(self.source)
        source["interval"]["calib"] = "oos"
        scenarios.append((self.config, source))
        config = copy.deepcopy(self.config)
        config["interval"].pop("calib")
        scenarios.append((config, self.source))
        for config, source in scenarios:
            with self.subTest(config=config, source=source), \
                    patch.dict(CONFIGS, {"source": source}), \
                    patch.object(evaluate, "windows", return_value=self.windows), \
                    patch("pipeline.side_cache.SideBacktestCache") as cache_type, \
                    patch.object(core, "fit_predict", side_effect=AssertionError("remote fit forbidden")) as fit:
                cache_type.return_value.load.side_effect = lambda name, s, e: copy.deepcopy(self.results[name])
                with self.assertRaises(ValueError):
                    evaluate.backtest(config, self.data, self.curve)
                fit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
