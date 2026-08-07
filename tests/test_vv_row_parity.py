"""V&V regressions: Gate3 row-set parity and honest floor selection.

Both cover ways a *reported* number can flatter the model without any single
line being obviously wrong — the failure mode this repo's headline OOF claims
are most exposed to.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rwt.gates import gate_residual_acceptance  # noqa: E402
from rwt.train import nested_floor_selection  # noqa: E402


class TestGate3RowParity(unittest.TestCase):
    def test_residual_cannot_win_by_dropping_its_hard_rows(self) -> None:
        """A residual defined only where it does well must not beat the prior.

        rmse() masks each predictor to its own finite pairs, so scoring the
        residual on 6 rows against a prior scored on 10 once let a head that is
        10x worse on every shared row pass the gate.
        """
        y = np.arange(10.0)
        prior = y.copy()
        prior[:6] += 0.1     # excellent on the rows the residual covers
        prior[6:] += 10.0    # awful on the rows the residual abstains from
        res = y + 1.0
        res[6:] = np.nan

        g = gate_residual_acceptance(y, res, prior)
        self.assertFalse(g.passed, "residual won on an unequal row set")
        self.assertEqual(g.detail["decision"], "FALLBACK_PURE_PRIOR")
        self.assertEqual(g.detail["n_scored_rows"], 6)
        self.assertEqual(g.detail["n_dropped_nonfinite_rows"], 4)
        # both scored on the shared rows only
        self.assertAlmostEqual(g.detail["oof_rmse_residual"], 1.0, places=6)
        self.assertAlmostEqual(g.detail["oof_rmse_prior"], 0.1, places=6)

    def test_last_known_shares_the_mask(self) -> None:
        y = np.arange(10.0)
        prior = y + 1.0
        lk = y + 0.05
        lk[7:] = np.nan
        res = y + 0.5
        g = gate_residual_acceptance(y, res, prior, last_known_pred=lk)
        self.assertEqual(g.detail["n_scored_rows"], 7)
        self.assertFalse(g.passed)  # loses last_known on the shared rows

    def test_all_nonfinite_fails_closed(self) -> None:
        y = np.arange(5.0)
        g = gate_residual_acceptance(y, np.full(5, np.nan), y + 1.0)
        self.assertFalse(g.passed)
        self.assertEqual(g.detail["n_scored_rows"], 0)


class TestNestedFloorSelection(unittest.TestCase):
    def test_prices_in_the_winners_curse(self) -> None:
        """Argmin over pure noise should look good in-sample and not out."""
        rng = np.random.default_rng(0)
        n, k = 400, 12
        y = rng.normal(0, 1, n)
        # every candidate is equally worthless; the best-looking one is luck
        cands = {f"c{i}": rng.normal(0, 1, n) for i in range(k)}
        fold_id = np.repeat(np.arange(5), n // 5)

        out = nested_floor_selection(y, cands, fold_id)
        self.assertTrue(out["ok"])
        self.assertEqual(out["n_candidates"], k)
        self.assertGreater(
            out["selection_optimism"], 0.0,
            "argmin on the scored rows must look better than an honest choice",
        )
        self.assertFalse(out["stable_pick"], "noise picks should not agree across folds")

    def test_needs_two_folds_and_two_candidates(self) -> None:
        y = np.arange(10.0)
        one = {"a": y + 1.0}
        self.assertFalse(nested_floor_selection(y, one, np.repeat([0, 1], 5))["ok"])
        two = {"a": y + 1.0, "b": y + 2.0}
        self.assertFalse(nested_floor_selection(y, two, np.zeros(10, int))["ok"])

    def test_skips_candidates_with_holes(self) -> None:
        y = np.arange(10.0)
        holed = y + 1.0
        holed[3] = np.nan
        cands = {"good": y + 0.5, "also": y + 2.0, "holed": holed}
        out = nested_floor_selection(y, cands, np.repeat([0, 1], 5))
        self.assertTrue(out["ok"])
        self.assertEqual(out["n_candidates"], 2)

    def test_stable_pick_when_one_candidate_truly_dominates(self) -> None:
        rng = np.random.default_rng(1)
        n = 400
        y = rng.normal(0, 1, n)
        cands = {"dominant": y + rng.normal(0, 0.05, n)}
        for i in range(5):
            cands[f"weak{i}"] = y + rng.normal(0, 2.0, n)
        out = nested_floor_selection(y, cands, np.repeat(np.arange(5), n // 5))
        self.assertTrue(out["ok"])
        self.assertTrue(out["stable_pick"])
        self.assertEqual(set(out["fold_picks"].values()), {"dominant"})


if __name__ == "__main__":
    unittest.main()
