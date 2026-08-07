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

    def test_stack_blend_ncc_soft_and_clamp(self) -> None:
        """Stack uses PF stalk + high-conf NCC soft pull, always clamp law."""
        from rwt.priors import blend_stack_priors

        n = 40
        last = 100.0
        dip = np.full(n, 105.0)
        pf = np.full(n, 110.0)
        conf = np.linspace(0.0, 1.0, n)
        ncc = np.full(n, 112.0)
        nc = np.where(np.arange(n) > 20, 0.9, 0.0)
        out = blend_stack_priors(
            last, dip, pf, conf, prior_ncc=ncc, ncc_conf=nc, max_dev=18.0
        )
        self.assertEqual(len(out), n)
        self.assertTrue(np.all(np.abs(out - last) <= 18.0 + 1e-9))
        # cold conf (near heel of this synthetic) stays near last / soft dip
        self.assertLess(float(np.mean(out[:8])), last + 4.0)
        # high conf + high ncc moves toward stalk within clamp
        self.assertGreater(float(np.mean(out[-8:])), last + 3.0)

    def test_impute_feature_frame_median_not_zero(self) -> None:
        from rwt.features import impute_feature_frame

        df = pd.DataFrame({"a": [10.0, np.nan, 20.0], "b": [np.nan, np.nan, np.nan]})
        out = impute_feature_frame(df)
        self.assertAlmostEqual(float(out["a"].iloc[1]), 15.0)
        self.assertTrue(np.allclose(out["b"].to_numpy(), 0.0))

    def test_fold_medians_do_not_use_holdout(self) -> None:
        """Train-fold medians ignore validation rows (CR P1)."""
        from rwt.features import compute_feature_medians, impute_feature_frame

        # train fold all ~10; holdout has 1000 — must not pull train median
        tr = pd.DataFrame({"x": [10.0, 10.0, np.nan]})
        va = pd.DataFrame({"x": [1000.0, np.nan]})
        med = compute_feature_medians(tr)
        self.assertAlmostEqual(med["x"], 10.0)
        va_i = impute_feature_frame(va, medians=med)
        self.assertAlmostEqual(float(va_i["x"].iloc[1]), 10.0)

    def test_serve_residual_clip(self) -> None:
        """Serve path clamps residual Δ and uses stored train medians."""
        from rwt.submit import predict_stations

        class _M:
            def predict(self, X):
                # batch values [99, nan] would fit median 99 if serve re-fit; stored is 7.0
                self.last_x = float(X[1, 0])
                return np.full(len(X), 50.0)  # runaway residual

        m = _M()
        st = pd.DataFrame(
            {
                "prior_stack": [100.0, 100.0],
                # finite 99 differs from stored median 7 so batch-fit would fail assert
                "relative_md": [99.0, np.nan],
            }
        )
        blob = {
            "models": [m],
            "feature_cols": ["relative_md"],
            "prior_column": "prior_stack",
            "use_residual": True,
            "residual_max_dev": 18.0,
            "feature_medians": {"relative_md": 7.0},
        }
        y = predict_stations(blob, st)
        self.assertAlmostEqual(m.last_x, 7.0)
        self.assertTrue(np.allclose(y, 118.0))

    def test_serve_residual_clip_vs_base_not_last(self) -> None:
        """Residual Δ clipped vs residual base prior; may leave last±dev if base does."""
        from rwt.submit import predict_stations

        class _M:
            def predict(self, X):
                return np.full(len(X), 30.0)

        st = pd.DataFrame(
            {
                "prior_stack": [110.0, 110.0],  # base already +10 from last
                "prior_last_tvt": [100.0, 100.0],
                "relative_md": [0.0, 1.0],
            }
        )
        blob = {
            "models": [_M()],
            "feature_cols": ["relative_md"],
            "prior_column": "prior_stack",
            "use_residual": True,
            "residual_max_dev": 18.0,
            "feature_medians": {"relative_md": 0.0},
        }
        y = predict_stations(blob, st)
        # 110 + clip(30→18) = 128 (beyond last±18 — intentional when residual wins)
        self.assertTrue(np.allclose(y, 128.0))

    def test_serve_blob_medians_match_refit_contract(self) -> None:
        """predict_stations reads feature_medians / residual_max_dev / prior_column."""
        from rwt.submit import predict_stations

        class _M:
            def predict(self, X):
                self.seen = float(X[0, 0])
                return np.zeros(len(X))

        m = _M()
        # keys the train.py blob writer / predict_stations consumer share
        blob = {
            "models": [m],
            "feature_cols": ["a"],
            "feature_medians": {"a": 1.5},
            "residual_max_dev": 18.0,
            "use_residual": True,
            "prior_column": "prior_stack",
        }
        for key in ("feature_medians", "residual_max_dev", "prior_column", "use_residual"):
            self.assertIn(key, blob)
        batch = pd.DataFrame(
            {
                "a": [np.nan, 99.0],
                "prior_stack": [10.0, 10.0],
            }
        )
        y = predict_stations(blob, batch)
        self.assertAlmostEqual(m.seen, 1.5)  # stored median, not batch 99
        self.assertTrue(np.allclose(y, 10.0))

    def test_serve_legacy_blob_zero_fill_not_batch_median(self) -> None:
        """Legacy blobs without feature_medians use zero-fill, not batch medians."""
        from rwt.submit import predict_stations

        class _M:
            def predict(self, X):
                self.seen = list(X[:, 0])
                return np.zeros(len(X))

        m = _M()
        blob = {
            "models": [m],
            "feature_cols": ["a"],
            "use_residual": True,
            "prior_column": "prior_stack",
            # no feature_medians key
        }
        batch = pd.DataFrame(
            {
                "a": [np.nan, 99.0],
                "prior_stack": [5.0, 5.0],
            }
        )
        predict_stations(blob, batch)
        # zero-fill nan; leave 99 as observed — never fit median=99 then fill nan→99
        self.assertAlmostEqual(m.seen[0], 0.0)
        self.assertAlmostEqual(m.seen[1], 99.0)

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
