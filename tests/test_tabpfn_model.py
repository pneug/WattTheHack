"""Offline tests: no API credentials, optional client, or network required."""

import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from pipeline.tabpfn_model import TabPFNAPI, _load_api_token


class FakeRegressor:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []

    def fit(self, X, y):
        self.fit_X, self.fit_y = X.copy(), np.asarray(y).copy()
        return self

    def predict(self, X, **kwargs):
        self.calls.append(X.copy())
        return X.iloc[:, 0].to_numpy() + 0.25


class TabPFNTests(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame({"H1": np.arange(100.), "O1": np.arange(100.) / 2},
                              index=pd.date_range("2026-07-01", periods=100, freq="10min"))
        self.y = pd.Series(np.arange(100.) + 1, index=self.X.index)

    def fitted(self, **kwargs):
        model = TabPFNAPI(max_train_rows=30, calibration_rows=10,
                          request_interval=0, **kwargs)
        fake = FakeRegressor()
        with patch.object(model, "_client", return_value=fake):
            model.fit(self.X, self.y, np.ones(len(self.X)))
        return model, fake

    def test_calibration_is_reserved_before_context_and_selection_is_repeatable(self):
        a, fake = self.fitted()
        b, _ = self.fitted()
        self.assertEqual(len(fake.fit_X), 30)
        self.assertEqual(int(a.calibration_mask_.sum()), 10)
        self.assertFalse(a.calibration_mask_[a.context_indices_].any())
        np.testing.assert_array_equal(a.context_indices_, b.context_indices_)
        np.testing.assert_array_equal(a.calibration_mask_, b.calibration_mask_)
        np.testing.assert_array_equal(fake.fit_y, self.y.iloc[a.context_indices_])
        changed_y = self.y * 4
        with patch.object(b, "_client", return_value=FakeRegressor()):
            b.fit(self.X, changed_y)
        np.testing.assert_array_equal(a.context_indices_, b.context_indices_)

    def test_batches_cache_column_order_and_return_copy(self):
        model, fake = self.fitted(batch_size=4)
        query = self.X.iloc[:10][["O1", "H1"]]
        p = model.predict(query)
        np.testing.assert_allclose(p, np.arange(10.) + .25)
        self.assertEqual([len(c) for c in fake.calls], [4, 4, 2])
        p[:] = -999
        np.testing.assert_allclose(model.predict(query), np.arange(10.) + .25)
        self.assertEqual(len(fake.calls), 3)
        self.assertEqual(len(model.predict(query.iloc[:0])), 0)

    def test_refit_clears_prediction_cache(self):
        model, fake = self.fitted()
        model.predict(self.X.iloc[:5])
        second = FakeRegressor()
        with patch.object(model, "_client", return_value=second):
            model.fit(self.X, self.y)
        model.predict(self.X.iloc[:5])
        self.assertEqual(len(second.calls), 1)

    def test_nonuniform_weights_fail_before_client_or_network(self):
        for w in (np.arange(100.), np.ones(99), np.full(100, np.nan), np.zeros(100)):
            with self.subTest(weights=w.shape):
                model = TabPFNAPI()
                with patch.object(model, "_client") as client:
                    with self.assertRaisesRegex(ValueError, "uniform positive"):
                        model.fit(self.X, self.y, w)
                    client.assert_not_called()

    def test_tiny_fit_keeps_at_least_two_context_rows(self):
        model = TabPFNAPI(request_interval=0)
        with patch.object(model, "_client", return_value=FakeRegressor()):
            model.fit(self.X.iloc[:3], self.y.iloc[:3])
        self.assertEqual(model.calibration_mask_.sum(), 1)
        self.assertEqual(len(model.context_indices_), 2)

    def test_bad_data_and_config_fail_early(self):
        for kwargs in ({"batch_size": 0}, {"max_train_rows": 1}, {"n_estimators": 9},
                       {"calibration_rows": False}, {"request_interval": np.nan}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TabPFNAPI(**kwargs)
        model = TabPFNAPI()
        for X, y in ((self.X.iloc[:2], self.y.iloc[:2]),
                     (self.X, self.y.reset_index(drop=True)),
                     (self.X.assign(H1=np.nan), self.y)):
            with self.assertRaises(ValueError):
                model.fit(X, y)
        with self.assertRaisesRegex(RuntimeError, "Fit TabPFN"):
            model.predict(self.X)

    def test_invalid_api_prediction_is_rejected(self):
        model, fake = self.fitted()
        fake.predict = lambda *args, **kwargs: np.array([np.nan])
        with self.assertRaisesRegex(RuntimeError, "invalid predictions"):
            model.predict(self.X.iloc[:3])

    def test_api_error_text_is_suppressed(self):
        model, fake = self.fitted()
        def fail(*args, **kwargs):
            raise RuntimeError("secret-bearer-token-in-client-error")
        fake.predict = fail
        with self.assertRaises(RuntimeError) as ctx:
            model.predict(self.X.iloc[:2])
        self.assertNotIn("secret-bearer", str(ctx.exception))
        self.assertTrue(ctx.exception.__suppress_context__)

    def test_token_environment_precedence_and_dotenv_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / ".env"
            p.write_text('UNRELATED=$(do_not_execute)\nexport tabpfn_API_key="fake-file-token" # note\n')
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(_load_api_token(p), "fake-file-token")
            with patch.dict(os.environ, {"TABPFN_TOKEN": "fake-env-token"}, clear=True):
                self.assertEqual(_load_api_token(p), "fake-env-token")
            p.write_text('UNRELATED=x\n')
            with patch.dict(os.environ, {}, clear=True), self.assertRaises(RuntimeError):
                _load_api_token(p)

    def test_explicit_stable_model_and_programmatic_auth(self):
        tokens = []
        fake_module = types.SimpleNamespace(set_access_token=tokens.append,
                                            TabPFNRegressor=FakeRegressor)
        with patch("pipeline.tabpfn_model._load_api_token", return_value="fake-test-token"), \
             patch("pipeline.tabpfn_model.importlib.import_module", return_value=fake_module):
            model = TabPFNAPI(request_interval=0).fit(self.X, self.y)
        self.assertEqual(tokens, ["fake-test-token"])
        self.assertEqual(model.m_.kwargs["model_path"], "v3.5_default")
        self.assertEqual(model.m_.kwargs["fit_mode"], "fit_with_cache")
        self.assertFalse(model.m_.kwargs["thinking_mode"])
        self.assertNotIn("fake-test-token", json.dumps(model.describe()))


if __name__ == "__main__":
    unittest.main()
