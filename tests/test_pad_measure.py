"""Independent synthetic metric/gate falsifiers over retained station arrays."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rwt.pad_measure import evaluate_station_metrics  # noqa: E402


def fixture() -> tuple[dict[str, np.ndarray], dict[str, str]]:
    arrays = {}
    roles = {"A": "train", "B": "test"}
    for well_id in roles:
        values = {
            "truth": np.array([0., 1., 2., 3.]), "last": np.zeros(4),
            "neighbor": np.array([0., 1., 2., 3.]), "is_predict": np.ones(4, dtype=bool),
            "gr_error_true": np.ones(4), "gr_error_last": np.full(4, 3.),
            "gr_error_neighbor": np.full(4, 2.),
        }
        arrays.update({f"{well_id}/{key}": value for key, value in values.items()})
    return arrays, roles


class TestPadMeasure(unittest.TestCase):
    def test_exact_common_scores_and_joint_gate(self) -> None:
        arrays, roles = fixture()
        table, summary = evaluate_station_metrics(arrays, roles)
        self.assertEqual(len(table), 2)
        self.assertAlmostEqual(summary["mean_rmse_last_known_test"], np.sqrt(3.5))
        self.assertEqual(summary["mean_rmse_neighbor_idw_test"], 0.)
        self.assertEqual(summary["mean_gr_mae_at_true_tvt"], 1.)
        self.assertTrue(summary["ok"])
        self.assertEqual(summary["status"], "synthetic_construct_passed")
        self.assertFalse(summary["predictive_admitted"])
        json.dumps(summary, allow_nan=False)

    def test_book_success_cannot_hide_failed_placement(self) -> None:
        arrays, roles = fixture()
        arrays["B/neighbor"] = np.full(4, 10.)
        _, summary = evaluate_station_metrics(arrays, roles)
        self.assertTrue(summary["falsifier_book_ok"])
        self.assertFalse(summary["falsifier_pad_transfer_ok"])
        self.assertFalse(summary["ok"])
        self.assertEqual(summary["status"], "rejected")

    def test_candidate_cannot_win_by_omitting_difficult_stations(self) -> None:
        arrays, roles = fixture()
        arrays["B/neighbor"][3] = np.nan
        table, summary = evaluate_station_metrics(arrays, roles)
        target = table[table.well_id.eq("B")].iloc[0]
        self.assertEqual(target.n_toe, 3)
        self.assertEqual(target.n_toe_expected, 4)
        self.assertAlmostEqual(target.rmse_last_known, np.sqrt(5 / 3))
        self.assertEqual(target.rmse_neighbor_idw, 0.)
        self.assertFalse(target.placement_coverage_complete)
        self.assertFalse(summary["ok"])
        self.assertEqual(summary["unscored_or_incomplete_wells"], ["B"])

    def test_gr_candidate_cannot_drop_reference_support(self) -> None:
        arrays, roles = fixture()
        arrays["B/gr_error_neighbor"][2] = np.nan
        table, summary = evaluate_station_metrics(arrays, roles)
        target = table[table.well_id.eq("B")].iloc[0]
        self.assertEqual(target.gr_n, 3)
        self.assertEqual(target.gr_n_expected, 4)
        self.assertFalse(target.gr_coverage_complete)
        self.assertFalse(summary["ok"])

    def test_missing_well_support_is_retained_and_never_nanmean_dropped(self) -> None:
        arrays, roles = fixture()
        for field in ("truth", "gr_error_true"):
            arrays[f"B/{field}"][:] = np.nan
        table, summary = evaluate_station_metrics(arrays, roles)
        self.assertEqual(set(table.well_id), {"A", "B"})
        self.assertEqual(summary["n_test"], 1)
        self.assertIsNone(summary["mean_rmse_neighbor_idw"])
        self.assertIsNone(summary["mean_gr_mae_at_true_tvt"])
        self.assertFalse(summary["ok"])
        json.dumps(summary, allow_nan=False)

    def test_optional_limit_is_declared_and_strict(self) -> None:
        arrays, roles = fixture()
        self.assertTrue(evaluate_station_metrics(arrays, roles)[1]["ok"])
        self.assertTrue(evaluate_station_metrics(arrays, roles, book_mae_limit=2.)[1]["ok"])
        for limit in (0., 1.):
            self.assertFalse(evaluate_station_metrics(arrays, roles, book_mae_limit=limit)[1]["ok"])
        for limit in (-1., np.nan, np.inf, True):
            with self.assertRaises(ValueError):
                evaluate_station_metrics(arrays, roles, book_mae_limit=limit)

    def test_at_least_one_test_well_is_required(self) -> None:
        arrays, roles = fixture()
        roles["B"] = "train"
        _, summary = evaluate_station_metrics(arrays, roles)
        self.assertEqual(summary["n_test"], 0)
        self.assertFalse(summary["ok"])

    def test_malformed_arrays_roles_and_keys_are_rejected(self) -> None:
        arrays, roles = fixture()
        for broken in (arrays | {"B/truth": np.zeros(3)}, arrays | {"B/truth": np.zeros((4, 1))}, arrays | {"B/is_predict": np.array([0., 1., np.nan, 0.])}):
            with self.assertRaises(ValueError):
                evaluate_station_metrics(broken, roles)
        for broken_roles in ({}, {"A": "train"}, roles | {"B": "hidden"}):
            with self.assertRaises(ValueError):
                evaluate_station_metrics(arrays, broken_roles)
        with self.assertRaises(ValueError):
            evaluate_station_metrics(arrays | {"B/secret": np.zeros(4)}, roles)


if __name__ == "__main__":
    unittest.main()
