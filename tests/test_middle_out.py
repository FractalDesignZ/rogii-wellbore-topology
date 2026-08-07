from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rwt.load import synthetic_well  # noqa: E402
from rwt.middle_out import known_prior_tvt, resolve_middle_out, split_by_mask  # noqa: E402


class TestMiddleOut(unittest.TestCase):
    def test_split_counts(self) -> None:
        lat, _tw = synthetic_well(n=100, ps_frac=0.7, seed=1)
        sp = split_by_mask(lat, well_id="SYN01")
        self.assertEqual(sp.n_known + sp.n_predict, 100)
        self.assertGreater(sp.n_predict, 0)
        self.assertGreater(sp.n_known, 0)
        self.assertAlmostEqual(sp.ps_md or 0, float(lat.loc[sp.predict_idx[0], "MD"]))

    def test_prior_on_known(self) -> None:
        lat, _ = synthetic_well(n=80, seed=2)
        lat = lat.assign(TVT=lat["TVT_true"])
        sp = split_by_mask(lat, well_id="SYN01")
        p = known_prior_tvt(lat, sp)
        self.assertTrue(all(k in p for k in ("prior_last_tvt", "prior_tvt_plus_z")))

    def test_tvt_input_heel_mask(self) -> None:
        """ROGII train: full TVT labels + TVT_input only on known heel."""
        n = 20
        df = pd.DataFrame(
            {
                "MD": np.arange(n, dtype=float),
                "Z": -np.arange(n, dtype=float),
                "TVT": 100 + np.arange(n, dtype=float),
                "TVT_input": [100.0 + i if i < 8 else np.nan for i in range(n)],
            }
        )
        sp = resolve_middle_out(df, well_id="w1")
        self.assertEqual(sp.n_known, 8)
        self.assertEqual(sp.n_predict, 12)

    def test_residual_zone_is_predict_not_heel(self) -> None:
        """Labeled residual stations should sit in predict zone (toe), prior from heel."""
        from rwt.features import build_station_frame

        n = 20
        df = pd.DataFrame(
            {
                "MD": np.arange(n, dtype=float),
                "X": np.arange(n, dtype=float),
                "Y": np.zeros(n),
                "Z": -np.arange(n, dtype=float),
                "GR": np.linspace(40, 80, n),
                "TVT": 100 + np.arange(n, dtype=float),
                "TVT_input": [100.0 + i if i < 8 else np.nan for i in range(n)],
            }
        )
        sp = resolve_middle_out(df, well_id="w1")
        st = build_station_frame(df, None, sp, station_idx=sp.predict_idx)
        self.assertTrue(bool(st["is_predict"].all()))
        self.assertFalse(bool(st["is_known"].any()))
        # prior uses heel last TVT, not toe station TVT
        self.assertAlmostEqual(float(st["prior_last_tvt"].iloc[0]), 107.0)


if __name__ == "__main__":
    unittest.main()
