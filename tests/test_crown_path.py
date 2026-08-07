"""Crown-path prior / PF upgrades: bias walk, ensemble best, neighbor offset."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rwt.baselines import rmse  # noqa: E402
from rwt.middle_out import resolve_middle_out  # noqa: E402
from rwt.particle_track import particle_filter_tvt, particle_prior_for_well  # noqa: E402
from rwt.priors import blend_prior_best  # noqa: E402


class TestCrownPath(unittest.TestCase):
    def test_blend_prior_best_stays_near_last(self) -> None:
        n = 50
        last = 100.0
        runaway = np.full(n, 200.0)
        conf = np.linspace(0, 1, n)
        out = blend_prior_best(
            last,
            runaway,
            runaway,
            runaway,
            conf,
            max_dev=18.0,
        )
        self.assertTrue(np.all(np.abs(out - last) <= 18.0 + 1e-9))

    def test_blend_uses_neighbor_when_ok(self) -> None:
        n = 20
        last = 100.0
        stack = np.full(n, 100.0)
        blend = np.full(n, 100.0)
        local = np.full(n, 100.0)
        conf = np.ones(n)  # neighbor weight requires PF conf
        nb = np.full(n, 110.0)
        ok = np.ones(n)
        out = blend_prior_best(
            last, stack, blend, local, conf, neighbor_tvt=nb, neighbor_ok=ok, max_dev=18.0
        )
        # with ok neighbor + conf, should pull slightly above last but still clamped
        self.assertGreater(float(np.mean(out)), last)
        self.assertTrue(np.all(out <= last + 18.0 + 1e-9))

    def test_pf_process_bias_and_shapes(self) -> None:
        tw_tvt = np.linspace(100, 200, 200)
        tw_gr = 80 + 20 * np.sin(tw_tvt / 8)
        n = 100
        true = 150 + np.linspace(0, 4, n)
        lat_gr = 80 + 20 * np.sin(true / 8) + np.random.default_rng(0).normal(0, 1.5, n)
        known = np.full(n, np.nan)
        known[:30] = true[:30]
        out = particle_filter_tvt(
            lat_gr,
            tw_gr,
            tw_tvt,
            seed_tvt=true[29],
            known_tvt=known,
            n_particles=300,
            rng=np.random.default_rng(1),
        )
        self.assertEqual(len(out["prior_pf_tvt"]), n)
        self.assertIn("process_bias", out)
        self.assertTrue(np.isfinite(out["prior_pf_tvt"]).all())
        # toe track must beat constant last-known on this synthetic GR match
        toe = slice(30, None)
        err_pf = rmse(true[toe], out["prior_pf_tvt"][toe])
        err_lk = rmse(true[toe], np.full(n - 30, true[29]))
        self.assertLess(err_pf, err_lk)

    def test_neighbor_offset_uses_tvt_input_only(self) -> None:
        """Hidden toe TVT must not drive neighbor heel offset."""
        from rwt.spatial import SpatialIndex, attach_spatial_features, well_spatial_node_from_lateral

        n = 40
        # target: heel TVT_input ~100, toe full TVT jumps to 200 (must not leak)
        lat = pd.DataFrame(
            {
                "MD": np.arange(n, dtype=float),
                "X": np.linspace(0, 100, n),
                "Y": np.zeros(n),
                "Z": -np.arange(n, dtype=float),
                "TVT_input": [100.0 if i < 15 else np.nan for i in range(n)],
                "TVT": [100.0 if i < 15 else 200.0 for i in range(n)],
            }
        )
        # neighbor nearby with flat TVT 100
        nb = pd.DataFrame(
            {
                "MD": np.arange(n, dtype=float),
                "X": np.linspace(10, 110, n),
                "Y": np.zeros(n),
                "Z": -np.arange(n, dtype=float),
                "TVT": np.full(n, 100.0),
                "TVT_input": [100.0 if i < 20 else np.nan for i in range(n)],
            }
        )
        idx = SpatialIndex(max_dist=500.0, close_dist=200.0)
        idx.add(well_spatial_node_from_lateral("tgt", lat))
        idx.add(well_spatial_node_from_lateral("nb", nb))
        st = pd.DataFrame({"station": list(range(n)), "well_id": ["tgt"] * n})
        out = attach_spatial_features(st, lat, idx, "tgt", max_dist=500.0, close_dist=200.0)
        # offset from heel-only should be near 0, not pulled by toe 200 labels
        self.assertLess(abs(float(out["neighbor_offset"].iloc[0])), 5.0)

    def test_ncc_track_clamped_and_shaped(self) -> None:
        from rwt.gr_match import ncc_tvt_track

        tw_tvt = np.linspace(100, 200, 200)
        tw_gr = 80 + 20 * np.sin(tw_tvt / 8)
        n = 80
        true = 150 + np.linspace(0, 3, n)
        lat = 80 + 20 * np.sin(true / 8)
        known = np.full(n, np.nan)
        known[:20] = true[:20]
        out = ncc_tvt_track(
            lat, tw_gr, tw_tvt, known_tvt=known, seed_tvt=true[19], stride=4, max_dev=18.0
        )
        self.assertEqual(len(out["prior_ncc_tvt"]), n)
        self.assertTrue(np.all(np.abs(out["prior_ncc_tvt"] - true[19]) <= 18.0 + 1e-6))
        # heel locked
        self.assertTrue(np.allclose(out["prior_ncc_tvt"][:20], true[:20]))
        self.assertTrue(np.all(out["ncc_track_conf"][:20] == 1.0))

    def test_ncc_cold_track_holds_seed(self) -> None:
        """Uncorrelated GR → no stalk trust → prior collapses to seed/known."""
        from rwt.gr_match import ncc_tvt_track

        tw_tvt = np.linspace(100, 200, 200)
        tw_gr = 80 + 20 * np.sin(tw_tvt / 8)
        n = 60
        seed = 150.0
        lat = np.random.default_rng(3).normal(80, 30, n)  # noise, no shape match
        known = np.full(n, np.nan)
        known[:10] = seed
        out = ncc_tvt_track(
            lat, tw_gr, tw_tvt, known_tvt=known, seed_tvt=seed, stride=8, max_dev=18.0
        )
        toe = out["prior_ncc_tvt"][10:]
        self.assertTrue(np.all(np.abs(toe - seed) <= 18.0 + 1e-6))
        # cold conf off-heel should be low when no peak accepted
        self.assertLess(float(np.mean(out["ncc_track_conf"][10:])), 0.5)

    def test_ncc_heel_conf_does_not_bleed_to_toe(self) -> None:
        """Known heel conf=1 must not nearest-fill into cold toe (prior_best poison)."""
        from rwt.gr_match import ncc_tvt_track

        tw_tvt = np.linspace(100, 200, 200)
        tw_gr = 80 + 20 * np.sin(tw_tvt / 8)
        n = 80
        seed = 150.0
        # pure noise + very high min_peak → zero NCC peaks accepted
        lat = np.random.default_rng(7).normal(80, 40, n)
        known = np.full(n, np.nan)
        known[:20] = seed
        out = ncc_tvt_track(
            lat,
            tw_gr,
            tw_tvt,
            known_tvt=known,
            seed_tvt=seed,
            stride=4,
            max_dev=18.0,
            min_peak=0.99,
        )
        self.assertTrue(np.all(out["ncc_track_conf"][:20] == 1.0))
        # toe must be ~0 conf (no peak samples to nearest-fill from)
        self.assertLess(float(np.max(out["ncc_track_conf"][20:])), 0.05)

    def test_ncc_unsorted_typewell_still_clamps(self) -> None:
        """TVT-descending typewell must not break seed-local band / clamp law."""
        from rwt.gr_match import ncc_tvt_track

        tw_tvt = np.linspace(200, 100, 200)  # reverse order
        tw_gr = 80 + 20 * np.sin(tw_tvt / 8)
        n = 50
        true = 150 + np.linspace(0, 2, n)
        lat = 80 + 20 * np.sin(true / 8)
        known = np.full(n, np.nan)
        known[:12] = true[:12]
        out = ncc_tvt_track(
            lat, tw_gr, tw_tvt, known_tvt=known, seed_tvt=true[11], stride=4, max_dev=18.0
        )
        self.assertTrue(np.all(np.abs(out["prior_ncc_tvt"] - true[11]) <= 18.0 + 1e-6))
        self.assertTrue(np.allclose(out["prior_ncc_tvt"][:12], true[:12]))

    def test_blend_ncc_conf_gated(self) -> None:
        """Zero NCC conf must not move prior_best; high conf may pull within clamp."""
        n = 30
        last = 100.0
        stack = np.full(n, 100.0)
        blend = np.full(n, 100.0)
        local = np.full(n, 100.0)
        conf = np.ones(n) * 0.5
        ncc = np.full(n, 112.0)
        cold = blend_prior_best(
            last, stack, blend, local, conf, prior_ncc=ncc, ncc_conf=np.zeros(n), max_dev=18.0
        )
        hot = blend_prior_best(
            last, stack, blend, local, conf, prior_ncc=ncc, ncc_conf=np.ones(n), max_dev=18.0
        )
        self.assertTrue(np.allclose(cold, last, atol=1e-6))
        self.assertGreater(float(np.mean(hot)), last + 0.5)
        self.assertTrue(np.all(hot <= last + 18.0 + 1e-9))

    def test_blend_stack_pull_when_pf_cold(self) -> None:
        """Stack-dominated weights: cold PF should still follow a good stack."""
        n = 20
        last = 100.0
        stack = np.full(n, 108.0)
        blend = np.full(n, 100.0)
        local = np.full(n, 100.0)
        conf = np.zeros(n)
        out = blend_prior_best(last, stack, blend, local, conf, max_dev=18.0)
        # stack weight high when conf=0 → mean between last and stack
        self.assertGreater(float(np.mean(out)), last + 2.0)
        self.assertLess(float(np.mean(out)), 108.0)
        self.assertTrue(np.all(np.abs(out - last) <= 18.0 + 1e-9))

    def test_process_bias_near_ps_clipped(self) -> None:
        """Build-section large dTVT must not set toe process_bias outside ±0.05."""
        tw_tvt = np.linspace(100, 200, 200)
        tw_gr = np.full(200, 80.0)
        n = 100
        known = np.full(n, np.nan)
        # first half: steep build; last 40: flat near PS
        known[:50] = 100 + np.linspace(0, 40, 50)
        known[50:70] = 140 + np.linspace(0, 0.5, 20)
        out = particle_filter_tvt(
            np.full(n, 80.0),
            tw_gr,
            tw_tvt,
            seed_tvt=140.5,
            known_tvt=known,
            n_particles=80,
            rng=np.random.default_rng(0),
        )
        self.assertLessEqual(abs(float(out["process_bias"])), 0.05 + 1e-9)

    def test_well_prior_has_blend(self) -> None:
        n = 40
        df = pd.DataFrame(
            {
                "MD": np.arange(n, dtype=float),
                "X": np.arange(n, dtype=float),
                "Y": np.zeros(n),
                "Z": -np.arange(n, dtype=float),
                "GR": 80 + 10 * np.sin(np.arange(n) / 5),
                "TVT": 100 + 0.05 * np.arange(n),
                "TVT_input": [100.0 + 0.05 * i if i < 15 else np.nan for i in range(n)],
            }
        )
        tw = pd.DataFrame(
            {
                "TVT": np.linspace(90, 120, 100),
                "GR": 80 + 10 * np.sin(np.linspace(0, 8, 100)),
            }
        )
        sp = resolve_middle_out(df, well_id="w")
        p = particle_prior_for_well(df, tw, sp, n_particles=120, bag=True, seed=0)
        self.assertIn("prior_pf_blend", p)
        self.assertEqual(len(p["prior_pf_bag"]), n)
        self.assertGreaterEqual(int(p["n_pf_bag"]), 4)
        last = float(p["prior_last_tvt"])
        # bag + blend must honor clamp law
        self.assertTrue(np.all(np.abs(p["prior_pf_bag"] - last) <= 18.0 + 1e-6))
        self.assertTrue(np.all(np.abs(p["prior_pf_blend"] - last) <= 18.0 + 1e-6))


if __name__ == "__main__":
    unittest.main()
