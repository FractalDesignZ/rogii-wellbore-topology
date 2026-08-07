"""Unit tests for CUDA residual packaging (device-agnostic)."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from rwt.neural_residual import _device, _well_tensors


class TestNeuralResidualUnit(unittest.TestCase):
    def test_device_is_str(self):
        d = _device()
        self.assertIn(d, ("cpu", "cuda"))

    def test_well_tensors_row_i_truncation(self):
        n = 100
        df = pd.DataFrame(
            {
                "well_id": ["W1"] * n,
                "station": np.arange(n),
                "TVT": np.linspace(100, 120, n),
                "prior_stack": np.linspace(100, 119, n),
                "f0": np.random.randn(n),
                "f1": np.random.randn(n),
            }
        )
        packs = _well_tensors(df, ["f0", "f1"], "prior_stack", max_len=40)
        self.assertEqual(len(packs), 1)
        p = packs[0]
        self.assertEqual(p["n"], 40)
        self.assertEqual(len(p["row_i"]), 40)
        # toe-heavy: last 40 rows of original frame
        np.testing.assert_array_equal(p["row_i"], np.arange(60, 100))
        np.testing.assert_allclose(p["y"], df["TVT"].to_numpy()[-40:])

    def test_multi_well_pack(self):
        rows = []
        for w, n in (("A", 10), ("B", 12)):
            for i in range(n):
                rows.append(
                    {
                        "well_id": w,
                        "station": i,
                        "TVT": float(i),
                        "prior_stack": float(i) - 0.1,
                        "f0": 1.0,
                    }
                )
        df = pd.DataFrame(rows)
        packs = _well_tensors(df, ["f0"], "prior_stack", max_len=64)
        self.assertEqual(len(packs), 2)
        self.assertEqual({p["well_id"] for p in packs}, {"A", "B"})


if __name__ == "__main__":
    unittest.main()
