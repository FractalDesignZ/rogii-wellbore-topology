"""Blueprint traps: form-surface leak, local-dip prior, relative features."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rwt.features import FEATURE_COLS_DEFAULT, build_station_frame, feature_matrix  # noqa: E402
from rwt.middle_out import resolve_middle_out  # noqa: E402
from rwt.priors import FORM_SURFACE_LEAK_COLS, compute_local_dip_prior  # noqa: E402


class TestBlueprintEngines(unittest.TestCase):
    def _synthetic_well(self, n: int = 80, heel: int = 25) -> tuple[pd.DataFrame, object]:
        # flat beds TVT = C - Z with C=1000; Z decreases along well
        md = np.arange(n, dtype=float)
        z = -9000.0 - 0.05 * md
        tvt = 1000.0 - z  # dTVT/dZ = -1
        # mild GR signal
        gr = 80 + 5 * np.sin(md / 7)
        df = pd.DataFrame(
            {
                "MD": md,
                "X": md * 10.0,
                "Y": np.zeros(n),
                "Z": z,
                "GR": gr,
                "TVT": tvt,
                "TVT_input": [float(tvt[i]) if i < heel else np.nan for i in range(n)],
                # train-only mirage surfaces — must never enter residual X
                "ANCC": np.full(n, -9100.0),
                "BUDA": np.full(n, -9500.0),
            }
        )
        sp = resolve_middle_out(df, well_id="BLUE01")
        return df, sp

    def test_form_surfaces_not_in_feature_cols(self) -> None:
        for c in FORM_SURFACE_LEAK_COLS:
            self.assertNotIn(c, FEATURE_COLS_DEFAULT)
            self.assertNotIn(f"form_{c}", FEATURE_COLS_DEFAULT)
            self.assertNotIn(f"z_minus_{c}", FEATURE_COLS_DEFAULT)
        # absolute map coords banned from residual X
        self.assertNotIn("MD", FEATURE_COLS_DEFAULT)
        self.assertNotIn("X", FEATURE_COLS_DEFAULT)
        self.assertNotIn("Y", FEATURE_COLS_DEFAULT)
        self.assertNotIn("Z", FEATURE_COLS_DEFAULT)
        # poison dual-transport / absolute lease encodings out of residual X
        self.assertNotIn("prior_mid", FEATURE_COLS_DEFAULT)
        self.assertNotIn("well_azimuth_x", FEATURE_COLS_DEFAULT)
        self.assertIn("relative_md", FEATURE_COLS_DEFAULT)
        self.assertIn("prior_local_dip", FEATURE_COLS_DEFAULT)
        self.assertIn("prior_last_tvt", FEATURE_COLS_DEFAULT)
        # private residual columns allowed (filtered out when absent from stations)
        self.assertIn("prior_beam_tvt", FEATURE_COLS_DEFAULT)
        self.assertIn("prior_beam_delta_stack", FEATURE_COLS_DEFAULT)

    def test_attach_private_residual_deltas_noop_and_deltas(self) -> None:
        from rwt.features import attach_private_residual_deltas

        pub = pd.DataFrame({"prior_stack": [1.0, 2.0], "prior_last_tvt": [1.0, 2.0]})
        out = attach_private_residual_deltas(pub)
        self.assertNotIn("prior_beam_delta_stack", out.columns)

        priv = pd.DataFrame(
            {
                "prior_beam_tvt": [3.0, 5.0],
                "prior_lattice_tvt": [4.0, 6.0],
                "prior_stack": [1.0, 2.0],
                "prior_last_tvt": [0.0, 1.0],
                "prior_best": [2.0, 4.0],
            }
        )
        out2 = attach_private_residual_deltas(priv)
        self.assertAlmostEqual(float(out2["prior_beam_delta_stack"].iloc[0]), 2.0)
        self.assertAlmostEqual(float(out2["prior_beam_delta_last"].iloc[1]), 4.0)
        self.assertAlmostEqual(float(out2["prior_beam_delta_best"].iloc[0]), 1.0)
        # lattice parity + stalk disagreement (cycle-17)
        self.assertAlmostEqual(float(out2["prior_lattice_delta_best"].iloc[0]), 2.0)
        self.assertAlmostEqual(float(out2["prior_lattice_delta_stack"].iloc[0]), 3.0)
        self.assertAlmostEqual(float(out2["prior_lat_beam_delta"].iloc[0]), 1.0)
        self.assertAlmostEqual(float(out2["prior_lat_beam_delta"].iloc[1]), 1.0)

        poison = pd.DataFrame(
            {
                "prior_beam_tvt": [3.0, np.nan, 7.0],
                "prior_lattice_tvt": [4.0, 5.0, np.nan],
                "prior_stack": [np.inf, 2.0, 3.0],
                "prior_last_tvt": [0.0, 1.0, np.inf],
                "prior_best": [2.0, np.inf, 4.0],
            }
        )
        out3 = attach_private_residual_deltas(poison)
        self.assertTrue(np.isnan(out3["prior_beam_delta_stack"].iloc[0]))
        self.assertTrue(np.isnan(out3["prior_beam_delta_stack"].iloc[1]))
        self.assertTrue(np.isnan(out3["prior_beam_delta_last"].iloc[2]))
        self.assertTrue(np.isnan(out3["prior_beam_delta_best"].iloc[1]))
        self.assertTrue(np.isnan(out3["prior_lattice_delta_best"].iloc[1]))
        self.assertTrue(np.isnan(out3["prior_lat_beam_delta"].iloc[1]))
        self.assertTrue(np.isnan(out3["prior_lat_beam_delta"].iloc[2]))

    def test_feature_matrix_rejects_form_leak(self) -> None:
        for leaked in FORM_SURFACE_LEAK_COLS:
            st = pd.DataFrame(
                {
                    "relative_md": [0.0, 1.0],
                    "GR": [80.0, 81.0],
                    f"form_{leaked}": [-9100.0, -9100.0],
                }
            )
            with self.assertRaises(ValueError):
                feature_matrix(st, cols=["relative_md", "GR", f"form_{leaked}"])

    def test_local_dip_prior_locks_heel_and_projects(self) -> None:
        df, sp = self._synthetic_well()
        out = compute_local_dip_prior(df, sp, heel_window_ft=300.0)
        prior = out["prior_local_dip"]
        k = sp.known_idx
        toe = sp.predict_idx
        # heel lock
        self.assertTrue(
            np.allclose(
                prior[k],
                df.iloc[k]["TVT_input"].to_numpy(float),
                atol=1e-6,
            )
        )
        # tiny MD slope only (<=0.008); never Z blow-up; heel locked
        self.assertLessEqual(abs(out["md_slope"]), 0.008 + 1e-9)
        self.assertEqual(out["local_dip"], 0.0)
        err = float(np.nanmean(np.abs(prior[toe] - df.iloc[toe]["TVT"].to_numpy(float))))
        # scale-independent: reject 50-150 ft poisoned projections
        self.assertLess(err, 25.0, msg=f"local dip toe err={err}")

    def test_build_station_frame_has_relative_and_no_form(self) -> None:
        df, sp = self._synthetic_well()
        tw = pd.DataFrame(
            {
                "TVT": np.linspace(float(df["TVT"].min()), float(df["TVT"].max()), 50),
                "GR": 80 + 5 * np.sin(np.linspace(0, 8, 50)),
            }
        )
        st = build_station_frame(df, tw, sp, station_idx=sp.predict_idx)
        self.assertIn("relative_md", st.columns)
        self.assertIn("relative_z", st.columns)
        self.assertIn("prior_local_dip", st.columns)
        self.assertIn("GR_mean_10", st.columns)
        self.assertIn("curvature_dls", st.columns)
        for c in FORM_SURFACE_LEAK_COLS:
            self.assertNotIn(f"form_{c}", st.columns)
            self.assertNotIn(f"z_minus_{c}", st.columns)
        X, cols = feature_matrix(st)
        self.assertEqual(X.shape[0], len(st))
        self.assertTrue(all("ANCC" not in c and "BUDA" not in c for c in cols))


if __name__ == "__main__":
    unittest.main()
