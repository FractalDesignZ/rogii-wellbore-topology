from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rwt.gates import gate_residual_acceptance, gate_residual_acceptance_per_well  # noqa: E402


class TestAdaptiveGate(unittest.TestCase):
    def test_global_reject_blocks_fire(self) -> None:
        y = np.array([10.0, 12.0, 11.0, 13.0])
        prior = np.array([10.0, 10.0, 10.0, 10.0])
        # residual worse than prior
        res = np.array([20.0, 22.0, 21.0, 23.0])
        g3 = gate_residual_acceptance(y, res, prior, prior_name="prior_stack")
        self.assertFalse(g3.passed)
        pw = gate_residual_acceptance_per_well(
            np.array(["w1", "w1", "w2", "w2"]),
            y, res, prior, global_gate=g3,
        )
        self.assertEqual(pw.detail["decision"], "GLOBAL_REJECT")
        self.assertEqual(pw.detail["n_fire"], 0)

    def test_per_well_fire_when_local_beat(self) -> None:
        # two wells; residual better on w1, worse on w2
        y = np.array([10.0, 12.0, 20.0, 22.0])
        prior = np.array([10.5, 12.5, 20.0, 22.0])
        res = np.array([10.1, 12.1, 25.0, 27.0])  # beats on w1, loses on w2
        g3 = gate_residual_acceptance(y, res, prior, prior_name="prior_stack")
        # may or may not pass global depending on overall; force via None global for local test
        pw = gate_residual_acceptance_per_well(
            np.array(["w1", "w1", "w2", "w2"]),
            y, res, prior, global_gate=None, min_stations=2,
        )
        self.assertTrue(pw.detail["well_fire"].get("w1", False))
        self.assertFalse(pw.detail["well_fire"].get("w2", True))


if __name__ == "__main__":
    unittest.main()
