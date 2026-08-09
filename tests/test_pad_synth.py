"""WP8 pad synthetic GR@TVT construct unit tests."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

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
            nb = neighbor_idw_tvt(w, donors)
            nb_f = np.where(np.isfinite(nb), nb, last)
            m = pred & np.isfinite(y)
            if rmse(y[m], nb_f[m]) < rmse(y[m], last[m]):
                wins += 1
        self.assertGreaterEqual(wins, 1)


if __name__ == "__main__":
    unittest.main()
