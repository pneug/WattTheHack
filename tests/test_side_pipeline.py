"""Network-free integration checks for optional models and calibration rows."""

import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from pipeline import core, stages
from pipeline import run as runner
from pipeline.experiments import CONFIGS


class ReservedCalibrationModel:
    """API-like model: fit context rows, reserve other rows for calibration."""

    def fit(self, X, y, w):
        self.train_X, self.train_y, self.weights = X.copy(), y.copy(), w.copy()
        self.calibration_mask_ = np.arange(len(X)) % 3 == 0
        self.context = X.index[~self.calibration_mask_]
        self.value = float(np.average(y.iloc[~self.calibration_mask_],
                                      weights=w[~self.calibration_mask_]))
        self.prediction_indices = []
        return self

    def predict(self, X):
        if X.index.isin(self.context).any():
            raise AssertionError("context rows must not calibrate the interval")
        self.prediction_indices.append(X.index.copy())
        return np.full(len(X), self.value)

    def describe(self):
        return {"type": "reserved-test-model", "value": self.value}


class SidePipelineTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range("2026-01-01", periods=60, freq="10min")
        x = np.linspace(2, 12, len(self.index))
        ws = pd.DataFrame(index=self.index)
        for i, name in enumerate(["H1", "H2", "H3", "H4", "H5", "O1", "O2"]):
            ws[name] = x + 0.1 * i + np.sin(np.arange(len(x)) / (3 + i))
        ws["T1"] = 1 + 0.8 * x + 0.3 * np.cos(np.arange(len(x)))
        self.data = {"wind_speed": ws}
        self.window = (self.index[24], self.index[30])
        self.test_rows = core.in_window(self.index, self.window)
        self.curve = pd.DataFrame({"wind_speed": [0., 10., 26.],
                                   "power": [0., 1000., 0.]})

    def predict(self, data, config="E00_baseline"):
        return core.fit_predict(data, config, self.window, targets=["T1"],
                                curve=self.curve)["T1"]

    def test_outage_targets_cannot_change_baseline_predictions_or_calibration(self):
        before = self.predict(self.data)
        changed = {"wind_speed": self.data["wind_speed"].copy()}
        changed["wind_speed"].loc[self.test_rows, "T1"] = 1e8
        after = self.predict(changed)
        pd.testing.assert_frame_equal(before.drop(columns="actual"),
                                      after.drop(columns="actual"))
        self.assertEqual(before.attrs, after.attrs)
        self.assertFalse(before.actual.equals(after.actual))

    def test_reserved_rows_only_calibrate_and_outage_is_never_context(self):
        model = ReservedCalibrationModel()
        with patch.object(stages, "make_model", return_value=model):
            result = self.predict(self.data)
        train_index = self.index[~self.test_rows]
        self.assertTrue(model.train_X.index.equals(train_index))
        self.assertTrue(model.train_y.index.equals(train_index))
        np.testing.assert_array_equal(model.weights, np.ones(len(train_index)))
        calibration = train_index[model.calibration_mask_]
        self.assertTrue(model.prediction_indices[0].equals(calibration))
        self.assertTrue(model.prediction_indices[1].equals(self.index[self.test_rows]))
        expected_resid = self.data["wind_speed"].loc[calibration, "T1"] - model.value
        np.testing.assert_allclose(result.attrs["resid"], expected_resid)
        np.testing.assert_allclose(result.attrs["fitted"], model.value)
        np.testing.assert_allclose(result.sigma, expected_resid.std(ddof=0))

        changed = {"wind_speed": self.data["wind_speed"].copy()}
        changed["wind_speed"].loc[self.test_rows, "T1"] = np.nan
        second = ReservedCalibrationModel()
        with patch.object(stages, "make_model", return_value=second):
            rerun = self.predict(changed)
        pd.testing.assert_series_equal(model.train_y, second.train_y)
        pd.testing.assert_frame_equal(result.drop(columns="actual"),
                                      rerun.drop(columns="actual"))
        self.assertEqual(result.attrs, rerun.attrs)

    def test_invalid_calibration_masks_fail_before_predicting(self):
        n = int((~self.test_rows).sum())
        invalid = [np.ones(n, dtype=int), np.ones(n - 1, dtype=bool),
                   np.ones((n, 1), dtype=bool), np.zeros(n, dtype=bool),
                   np.array(True)]
        for mask in invalid:
            with self.subTest(shape=mask.shape, dtype=mask.dtype):
                model = ReservedCalibrationModel()
                original_fit = model.fit

                def fit_with_invalid_mask(X, y, w):
                    original_fit(X, y, w)
                    model.calibration_mask_ = mask
                    return model

                with patch.object(model, "fit", side_effect=fit_with_invalid_mask), \
                        patch.object(stages, "make_model", return_value=model):
                    with self.assertRaisesRegex(ValueError, "calibration_mask_"):
                        self.predict(self.data)
                self.assertEqual(model.prediction_indices, [])


class SideExperimentCLITests(unittest.TestCase):
    def run_cli(self, argv):
        data = object()
        with patch.object(core, "load_data", return_value=data) as load, \
                patch.object(runner, "run_one") as run_one, \
                patch.object(runner.report, "update_log") as report:
            runner.main(argv)
        load.assert_called_once_with()
        report.assert_called_once_with()
        self.assertTrue(all(call.args[1] is data for call in run_one.call_args_list))
        return [call.args[0] for call in run_one.call_args_list]

    def test_all_excludes_every_side_experiment_by_default(self):
        expected = [key for key, cfg in CONFIGS.items() if not cfg.get("side_experiment")]
        self.assertEqual(self.run_cli(["--all"]), expected)
        self.assertTrue(all(CONFIGS[key].get("side_experiment") for key in
                            ("S01_mlp_matched", "S02_mlp_refs", "S03_tabpfn35_refs")))

    def test_all_opt_in_includes_side_experiments(self):
        self.assertEqual(self.run_cli(["--all", "--include-side-experiments"]), list(CONFIGS))

    def test_explicit_side_ids_are_dispatched_without_all_flag(self):
        ids = ["S03_tabpfn35_refs", "S01_mlp_matched", "S02_mlp_refs"]
        self.assertEqual(self.run_cli(ids), ids)

    def test_baseline_import_and_factory_do_not_import_optional_packages(self):
        script = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'tabpfn', 'tabpfn_client', 'torch', 'sklearn'}:
        raise AssertionError('optional import during baseline use: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from pipeline import stages
from pipeline import run
assert isinstance(stages.make_model('ols'), stages.OLS)
"""
        proc = subprocess.run([sys.executable, "-c", script], capture_output=True,
                              text=True, cwd=Path(__file__).resolve().parents[1],
                              timeout=30)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)


if __name__ == "__main__":
    unittest.main()
