from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rwt.gates import (  # noqa: E402
    gate_ip_fence,
    gate_residual_acceptance,
    run_entry_gates,
)


class TestGates(unittest.TestCase):
    def test_ip_fence(self) -> None:
        g = gate_ip_fence()
        self.assertTrue(g.passed)
        self.assertIsInstance(g.detail["kernel_private"], bool)
        self.assertIn("kernel_id", g.detail)

    def test_residual_accept(self) -> None:
        y = np.array([1.0, 2.0, 3.0, 4.0])
        prior = np.array([1.1, 2.1, 3.1, 4.1])
        better = np.array([1.0, 2.0, 3.0, 4.0])
        g = gate_residual_acceptance(y, better, prior)
        self.assertTrue(g.passed)
        worse = np.array([10.0, 10.0, 10.0, 10.0])
        g2 = gate_residual_acceptance(y, worse, prior)
        self.assertFalse(g2.passed)
        self.assertEqual(g2.detail["decision"], "FALLBACK_PURE_PRIOR")
        # residual beats prior (rmse~0.05) but loses a stronger floor (0.01) → REJECT
        almost = np.array([1.05, 2.05, 3.05, 4.05])
        g3 = gate_residual_acceptance(
            y,
            almost,
            prior,
            floor_rmses={"strong_floor": 0.01},
        )
        self.assertFalse(g3.passed)
        self.assertTrue(g3.detail["beat_prior"])
        self.assertFalse(g3.detail["beat_all_floors"])
        self.assertEqual(g3.detail["best_floor_name"], "strong_floor")

    def test_entry_gates(self) -> None:
        self.assertGreaterEqual(len(run_entry_gates()), 1)


if __name__ == "__main__":
    unittest.main()
