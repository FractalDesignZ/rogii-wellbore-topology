"""WP8 pad synthetic GR@TVT construct unit tests."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rwt.pad_synth import (  # noqa: E402
    gr_mismatch_stats,
    last_known_series,
    make_pad_laterals,
    make_typewell_book,
    neighbor_idw_tvt,
)


class TestPadSynth(unittest.TestCase):
    def test_book_and_pad_roles(self) -> None:
        pad = make_pad_laterals(n_wells=4, n_stations=80, n_train=2, seed=3)
        self.assertEqual(len(pad["train_ids"]), 2)
        self.assertEqual(len(pad["test_ids"]), 2)
        tw = pad["typewell"]
        self.assertGreater(len(tw), 50)
        for w in pad["wells"]:
            pred = w["is_predict"].to_numpy()
            if w["pad_role"].iloc[0] == "test":
                self.assertFalse(np.isfinite(w.loc[pred, "TVT"]).any())
            self.assertTrue(np.isfinite(w["TVT_true"]).all())

    def test_gr_mae_true_beats_wrong_tvt(self) -> None:
        pad = make_pad_laterals(n_wells=3, n_stations=100, seed=1, gr_noise=2.0)
        w = pad["wells"][0]
        tw = pad["typewell"]
        pred = w["is_predict"].to_numpy()
        y = w["TVT_true"].to_numpy()
        gr = w["GR"].to_numpy()
        heel = ~pred
        known = np.where(heel, w["TVT_input"].to_numpy(dtype=float), np.nan)
        mm_t = gr_mismatch_stats(gr, tw, y, known_tvt=known, mask=pred)
        mm_w = gr_mismatch_stats(
            gr, tw, y + 25.0, known_tvt=known, mask=pred
        )
        self.assertLess(
            float(mm_t["mean_abs_mismatch"]), float(mm_w["mean_abs_mismatch"])
        )

    def test_neighbor_beats_last_on_test(self) -> None:
        pad = make_pad_laterals(
            n_wells=6, n_stations=120, n_train=3, seed=9, gr_noise=2.0
        )
        donors = [w for w in pad["wells"] if w["pad_role"].iloc[0] == "train"]
        from rwt.baselines import rmse

        wins = 0
        for w in pad["wells"]:
            if w["pad_role"].iloc[0] != "test":
                continue
            pred = w["is_predict"].to_numpy()
            y = w["TVT_true"].to_numpy()
            last = last_known_series(w)
            nb = neighbor_idw_tvt(w, donors, label_policy="full_tvt")
            nb_f = np.where(np.isfinite(nb), nb, last)
            m = pred & np.isfinite(y)
            if rmse(y[m], nb_f[m]) < rmse(y[m], last[m]):
                wins += 1
        self.assertGreaterEqual(wins, 1)

    def test_target_and_donor_truth_never_enter_predictions(self) -> None:
        pad = make_pad_laterals(n_wells=4, n_stations=80, n_train=2, seed=13)
        target = pad["wells"][-1].copy()
        donors = [w.copy() for w in pad["wells"][:2]]
        for policy in ("observed_only", "full_tvt"):
            expected = neighbor_idw_tvt(target, donors, label_policy=policy)
            poisoned_target = target.copy()
            poisoned_target[["TVT", "TVT_true"]] = -1e9
            poisoned = [d.copy() for d in donors]
            for d in poisoned:
                d["TVT_true"] = 1e9
                if policy == "observed_only":
                    d["TVT"] = -1e9
            actual = neighbor_idw_tvt(poisoned_target, poisoned, label_policy=policy)
            np.testing.assert_array_equal(actual, expected)

    def test_self_donors_are_excluded_by_identity(self) -> None:
        target = make_pad_laterals(n_wells=2, n_stations=60, seed=11)["wells"][0]
        for policy in ("observed_only", "full_tvt"):
            actual = neighbor_idw_tvt(target, [target.copy()], label_policy=policy)
            self.assertTrue(np.isnan(actual).all())

    def test_missing_observations_do_not_fall_back_to_full_labels(self) -> None:
        pad = make_pad_laterals(n_wells=2, n_stations=60, seed=11)
        donor, target = [w.copy() for w in pad["wells"]]
        for missing in ("absent", "nonfinite", "sentinel"):
            d = donor.drop(columns="TVT_input") if missing == "absent" else donor.assign(TVT_input=-999.25 if missing == "sentinel" else np.nan)
            self.assertTrue(np.isnan(neighbor_idw_tvt(target, [d])).all())
            full = neighbor_idw_tvt(target, [d], label_policy="full_tvt")
            self.assertTrue(np.isfinite(full).any())

    def test_full_labels_require_explicit_training_donor_role(self) -> None:
        pad = make_pad_laterals(n_wells=3, n_stations=60, n_train=1, seed=11)
        target = pad["wells"][1]
        test_donor = pad["wells"][2].copy()
        test_donor["TVT"] = test_donor["TVT_true"]
        with self.assertRaisesRegex(ValueError, "training donors"):
            neighbor_idw_tvt(target, [test_donor], label_policy="full_tvt")
        train_donor = pad["wells"][0].drop(columns="pad_role")
        with self.assertRaisesRegex(ValueError, "training donors"):
            neighbor_idw_tvt(target, [train_donor], label_policy="full_tvt")

    def test_invalid_coordinates_have_no_transfer(self) -> None:
        x = np.arange(8, dtype=float)
        donor = pd.DataFrame({"well_id": "donor", "pad_role": "train", "X": x, "Y": 1.0, "TVT_input": 100 + x, "TVT": 200 + x})
        target = pd.DataFrame({"well_id": "target", "X": x, "Y": 0.0})
        donor.loc[:2, "X"] = np.nan
        donor.loc[3, "Y"] = np.inf
        target.loc[0, "X"] = np.nan
        target.loc[1, "Y"] = np.inf
        pred = neighbor_idw_tvt(target, [donor])
        self.assertTrue(np.isnan(pred[:2]).all())
        self.assertTrue(np.isfinite(pred[2:]).all())
        donor["X"] = np.nan
        self.assertTrue(np.isnan(neighbor_idw_tvt(target, [donor])).all())

    def test_equal_distance_donors_are_order_independent(self) -> None:
        x = np.arange(8, dtype=float)
        target = pd.DataFrame({"well_id": "target", "X": x, "Y": 0.0})
        a = pd.DataFrame({"well_id": "a", "X": x, "Y": -1.0, "TVT_input": 10.0})
        b = pd.DataFrame({"well_id": "b", "X": x, "Y": 1.0, "TVT_input": 20.0})
        np.testing.assert_array_equal(neighbor_idw_tvt(target, [a, b], k=1), 10.0)
        np.testing.assert_array_equal(neighbor_idw_tvt(target, [b, a], k=1), 10.0)

    def test_ambiguous_identity_is_rejected_before_transfer(self) -> None:
        pad = make_pad_laterals(n_wells=2, n_stations=60, seed=11)
        donor, target = pad["wells"]
        with self.assertRaisesRegex(ValueError, "well_id"):
            neighbor_idw_tvt(target.drop(columns="well_id"), [donor])
        mixed = donor.copy()
        mixed.loc[0, "well_id"] = "other"
        with self.assertRaisesRegex(ValueError, "well_id"):
            neighbor_idw_tvt(target, [mixed])


if __name__ == "__main__":
    unittest.main()
