"""Focused checks for the optional weighted neural pipeline model."""

import json
import unittest

import numpy as np
import pandas as pd

from pipeline.nn_model import SmallMLP


class SmallMLPTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(11)
        self.X = pd.DataFrame(rng.normal(size=(160, 2)), columns=["H", "O"],
                              index=pd.date_range("2026-01-01", periods=160, freq="10min"))
        self.y = pd.Series(4 + 2 * self.X.H - 0.5 * self.X.O, index=self.X.index)

    def make_model(self, **kwargs):
        return SmallMLP(hidden_layer_sizes=(8, 4), max_epochs=15, patience=4,
                        batch_size=32, **kwargs)

    def test_finite_deterministic_predictions_and_json_description(self):
        first = self.make_model().fit(self.X, self.y)
        second = self.make_model().fit(self.X, self.y)
        actual = first.predict(self.X)
        np.testing.assert_array_equal(actual, second.predict(self.X))
        self.assertEqual(actual.shape, (len(self.X),))
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_array_equal(actual, first.predict(self.X[["O", "H"]]))
        self.assertEqual(first.predict(self.X.iloc[:0]).shape, (0,))
        self.assertLessEqual(first.selection_epochs_, 15)
        self.assertLessEqual(first.selected_epochs_, first.selection_epochs_)
        json.dumps(first.describe(), allow_nan=False)

    def test_inner_scalers_exclude_chronological_tail_and_final_fit_uses_all_training(self):
        # A large shift in the last 10% reveals accidental use of the inner tail
        # when fitting selection scalers. Shuffling input tests temporal sorting.
        X, y = self.X.copy(), self.y.copy()
        X.iloc[144:] += 1000
        y.iloc[144:] += 1000
        order = np.random.default_rng(4).permutation(len(X))
        model = self.make_model().fit(X.iloc[order], y.iloc[order])
        np.testing.assert_allclose(model.selection_x_mean_, X.iloc[:144].mean())
        self.assertAlmostEqual(model.selection_y_mean_, y.iloc[:144].mean())
        np.testing.assert_allclose(model.x_scaler_.mean_, X.mean())
        self.assertAlmostEqual(model.y_scaler_.mean_[0], y.mean())
        old_x_mean, old_y_mean = model.x_scaler_.mean_.copy(), model.y_scaler_.mean_.copy()
        model.predict(X.iloc[:3] * 100)
        np.testing.assert_array_equal(model.x_scaler_.mean_, old_x_mean)
        np.testing.assert_array_equal(model.y_scaler_.mean_, old_y_mean)

    def test_zero_weight_rows_have_no_effect_and_weights_really_change_fit(self):
        weights = np.ones(len(self.X))
        weights[-30:] = 0
        base = self.make_model(validation_fraction=0).fit(self.X.iloc[:-30], self.y.iloc[:-30])
        modified = self.y.copy()
        modified.iloc[-30:] = 100000
        weighted = self.make_model(validation_fraction=0).fit(self.X, modified, weights)
        np.testing.assert_allclose(base.predict(self.X), weighted.predict(self.X), atol=1e-12)
        self.assertEqual(weighted.n_train_, 130)

        # Conflicting targets at identical feature values isolate weighting from
        # feature selection. Constant y within each half gives a clear signal.
        X = pd.DataFrame({"H": np.zeros(100)})
        y = pd.Series(np.r_[np.zeros(50), np.full(50, 10.)])
        uniform = self.make_model(validation_fraction=0).fit(X, y)
        w = np.r_[np.ones(50), np.full(50, 20.)]
        heavy = self.make_model(validation_fraction=0).fit(X, y, w)
        self.assertGreater(heavy.predict(X)[0], uniform.predict(X)[0] + 2)
        self.assertAlmostEqual(heavy.y_scaler_.mean_[0], np.average(y, weights=w))

    def test_positive_weight_scale_does_not_change_results(self):
        weights = np.linspace(0.1, 3, len(self.X))
        first = self.make_model().fit(self.X, self.y, weights)
        second = self.make_model().fit(self.X, self.y, weights * 40)
        np.testing.assert_allclose(first.predict(self.X), second.predict(self.X), atol=1e-12)

    def test_weights_affect_training_loss_even_with_identical_scaling(self):
        # Both weightings have the same X/y marginal means and variances, but
        # favour opposite relationships. This catches weights used in scaling
        # alone and silently dropped when training the network.
        X = pd.DataFrame({"H": np.tile([-1., -1., 1., 1.], 50)})
        y = pd.Series(np.tile([-1., 1., -1., 1.], 50))
        kwargs = dict(hidden_layer_sizes=(8, 4), max_epochs=60, validation_fraction=0,
                      learning_rate_init=0.01, batch_size=32)
        a = SmallMLP(**kwargs).fit(X, y, np.tile([20., 1., 1., 20.], 50))
        b = SmallMLP(**kwargs).fit(X, y, np.tile([1., 20., 20., 1.], 50))
        np.testing.assert_allclose(a.x_scaler_.mean_, b.x_scaler_.mean_, atol=1e-12)
        np.testing.assert_allclose(a.y_scaler_.mean_, b.y_scaler_.mean_, atol=1e-12)
        np.testing.assert_allclose(a.y_scaler_.scale_, b.y_scaler_.scale_, atol=1e-12)
        self.assertGreater(a.predict(X.iloc[[2]])[0], 0.7)
        self.assertLess(b.predict(X.iloc[[2]])[0], -0.7)

    def test_patience_selects_and_refits_best_epoch(self):
        model = SmallMLP(hidden_layer_sizes=(8, 4), max_epochs=20, patience=3,
                         min_delta=1e6).fit(self.X, self.y)
        self.assertEqual(model.selected_epochs_, 1)
        self.assertEqual(model.selection_epochs_, 4)
        self.assertEqual(model.m_.n_iter_, 1)
        self.assertEqual(model.n_train_, len(self.X))

    def test_invalid_inputs_fail_explicitly(self):
        for kwargs in ({"max_epochs": 0}, {"patience": -1}, {"batch_size": 0},
                       {"validation_fraction": 0.5}, {"alpha": -1},
                       {"learning_rate_init": 0}, {"random_state": None},
                       {"hidden_layer_sizes": (0,)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SmallMLP(**kwargs)
        for weights in (np.zeros(160), np.full(160, -1.), np.full(160, np.nan), np.ones(2)):
            with self.subTest(weights=weights[:2]), self.assertRaises(ValueError):
                self.make_model().fit(self.X, self.y, weights)
        with self.assertRaises(ValueError):
            self.make_model().fit(self.X, self.y.iloc[::-1])
        with self.assertRaises(ValueError):
            self.make_model().fit(self.X.assign(H=np.nan), self.y)
        with self.assertRaises(RuntimeError):
            self.make_model().predict(self.X)


if __name__ == "__main__":
    unittest.main()
