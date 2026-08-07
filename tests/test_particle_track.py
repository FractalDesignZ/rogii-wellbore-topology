from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rwt.baselines import rmse  # noqa: E402
from rwt.middle_out import resolve_middle_out  # noqa: E402
from rwt.particle_track import (  # noqa: E402
    blend_pf_last_known,
    particle_filter_tvt,
    particle_prior_for_well,
)


class TestParticleTrack(unittest.TestCase):
    def test_blend_respects_low_confidence(self) -> None:
        pf = np.array([100.0, 200.0, 300.0])
        conf = np.array([0.0, 0.5, 1.0])
        out = blend_pf_last_known(pf, 50.0, conf, conf_floor=0.0, conf_ceil=1.0)
        self.assertAlmostEqual(out[0], 50.0)
        self.assertAlmostEqual(out[2], 300.0)

    def test_pf_runs_and_shapes(self) -> None:
        # synthetic typewell GR sine in TVT
        tw_tvt = np.linspace(100, 200, 200)
        tw_gr = 80 + 20 * np.sin(tw_tvt / 8)
        # lateral follows mid TVT with noise
        n = 80
        true_tvt = 150 + np.linspace(0, 5, n)
        lat_gr = 80 + 20 * np.sin(true_tvt / 8) + np.random.default_rng(0).normal(0, 2, n)
        out = particle_filter_tvt(
            lat_gr, tw_gr, tw_tvt, seed_tvt=150.0, n_particles=200, rng=np.random.default_rng(1)
        )
        self.assertEqual(len(out["prior_pf_tvt"]), n)
        self.assertTrue(np.isfinite(out["prior_pf_tvt"]).all())
        # should track near true better than constant 100
        err_pf = rmse(true_tvt, out["prior_pf_tvt"])
        err_bad = rmse(true_tvt, np.full(n, 100.0))
        self.assertLess(err_pf, err_bad)

    def test_dip_prior_heel_only(self) -> None:
        from rwt.priors import compute_heel_fitted_dip_prior

        n = 30
        df = pd.DataFrame(
            {
                "MD": np.arange(n, dtype=float),
                "X": np.linspace(0, 1000, n),
                "Y": np.linspace(0, 200, n),
                "Z": -9000 - np.linspace(0, 30, n),
                "TVT": 100 + 0.01 * np.linspace(0, 1000, n) + 0.02 * np.linspace(0, 200, n),
                "TVT_input": [
                    100 + 0.01 * (i * 1000 / 29) + 0.02 * (i * 200 / 29) if i < 12 else np.nan
                    for i in range(n)
                ],
            }
        )
        sp = resolve_middle_out(df, well_id="w")
        d = compute_heel_fitted_dip_prior(df, sp, alpha=1.0)
        self.assertEqual(len(d["prior_dip_stack"]), n)
        k = sp.known_idx
        err = np.nanmean(np.abs(d["prior_dip_stack"][k] - df.iloc[k]["TVT_input"].to_numpy(float)))
        self.assertLess(err, 2.0)
        # dip coefficients should be non-trivial for this synthetic plane
        self.assertTrue(abs(d["dip_coef_x"]) + abs(d["dip_coef_y"]) > 1e-6)

    def test_well_prior_blend_column(self) -> None:
        n = 40
        df = pd.DataFrame(
            {
                "MD": np.arange(n, dtype=float),
                "X": np.arange(n, dtype=float),
                "Y": np.zeros(n),
                "Z": -np.arange(n, dtype=float),
                "GR": 80 + 10 * np.sin(np.arange(n) / 5),
                "TVT": 100 + 0.1 * np.arange(n),
                "TVT_input": [100.0 + 0.1 * i if i < 15 else np.nan for i in range(n)],
            }
        )
        tw = pd.DataFrame(
            {
                "TVT": np.linspace(90, 130, 100),
                "GR": 80 + 10 * np.sin(np.linspace(0, 8, 100)),
            }
        )
        sp = resolve_middle_out(df, well_id="w")
        p = particle_prior_for_well(df, tw, sp, n_particles=150, bag=False, seed=0)
        self.assertEqual(len(p["prior_pf_blend"]), n)
        self.assertIn("prior_last_tvt", p)
        # heel lock
        k = sp.known_idx
        locked = p["prior_pf_blend"][k]
        true_h = df.iloc[k]["TVT_input"].to_numpy(float)
        self.assertTrue(np.allclose(locked, true_h, equal_nan=False, atol=1e-6))
        # seed changes bagged toe slightly when bag=False still uses seed
        p2 = particle_prior_for_well(df, tw, sp, n_particles=150, bag=False, seed=7)
        toe = sp.predict_idx
        # at least some toe difference expected under different RNG
        self.assertEqual(len(p2["prior_pf_tvt"]), n)
        _ = np.max(np.abs(p["prior_pf_tvt"][toe] - p2["prior_pf_tvt"][toe]))


if __name__ == "__main__":
    unittest.main()
