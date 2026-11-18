from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rwt.features import build_station_frame, feature_matrix, FEATURE_COLS_DEFAULT  # noqa: E402
from rwt.middle_out import resolve_middle_out  # noqa: E402
from rwt.spatial import (  # noqa: E402
    attach_spatial_features,
    build_spatial_index,
    neighbor_tvt_transfer,
    well_spatial_node_from_lateral,
)


class TestSpatial(unittest.TestCase):
    @staticmethod
    def _well(*, y: float = 0.0, tvt: float = 100.0) -> pd.DataFrame:
        n = 20
        observed = tvt + np.arange(n, dtype=float) * 0.2
        observed[8:] = np.nan
        return pd.DataFrame(
            {
                "MD": np.arange(n, dtype=float),
                "X": np.arange(n, dtype=float) * 5.0,
                "Y": np.full(n, y),
                "Z": -np.arange(n, dtype=float),
                "TVT_input": observed,
                "TVT": tvt + np.arange(n, dtype=float),
            }
        )

    @staticmethod
    def _stations(lateral: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"station": np.arange(len(lateral), dtype=int)})

    def test_neighbor_graph_and_transfer(self) -> None:
        def make_well(wid: str, x0: float, y0: float, tvt0: float) -> pd.DataFrame:
            n = 30
            return pd.DataFrame(
                {
                    "MD": np.arange(n, dtype=float),
                    "X": x0 + np.linspace(0, 500, n),
                    "Y": y0 + np.zeros(n),
                    "Z": -9000 - np.linspace(0, 20, n),
                    "TVT": tvt0 + np.linspace(0, 10, n),
                    "TVT_input": [tvt0 + i if i < 10 else np.nan for i in range(n)],
                }
            )

        a = make_well("a", 0, 0, 100)
        b = make_well("b", 50, 10, 110)  # close neighbor
        c = make_well("c", 5000, 5000, 200)  # far
        idx = build_spatial_index([("a", a), ("b", b), ("c", c)])
        # adaptive radii from field quantiles (close pair a–b must link)
        neigh = idx.neighbors("a", k=5)
        ids = [w for w, _d in neigh]
        self.assertIn("b", ids)
        self.assertGreater(idx.max_dist, 0.0)
        node_b = well_spatial_node_from_lateral("b", b)
        tr = neighbor_tvt_transfer(a, node_b, close_dist=150.0)
        self.assertEqual(len(tr["neighbor_tvt"]), len(a))
        self.assertTrue(np.isfinite(tr["neighbor_tvt"]).all())

    def test_multi_neighbor_columns(self) -> None:
        def make_well(wid: str, x0: float, y0: float, tvt0: float) -> pd.DataFrame:
            n = 40
            return pd.DataFrame(
                {
                    "MD": np.arange(n, dtype=float),
                    "X": x0 + np.linspace(0, 400, n),
                    "Y": y0 + np.zeros(n),
                    "Z": -9000 - np.linspace(0, 15, n),
                    "TVT": tvt0 + np.linspace(0, 8, n),
                    "TVT_input": [tvt0 + i * 0.2 if i < 12 else np.nan for i in range(n)],
                    "GR": 80 + 5 * np.sin(np.arange(n) / 6),
                }
            )

        a = make_well("a", 0, 0, 100)
        b = make_well("b", 40, 5, 105)
        c = make_well("c", 80, -8, 98)
        idx = build_spatial_index([("a", a), ("b", b), ("c", c)])
        sp = resolve_middle_out(a, well_id="a")
        st = build_station_frame(a, None, sp, spatial_index=idx)
        self.assertIn("neighbor_tvt_wmean", st.columns)
        self.assertIn("neighbor_count_close", st.columns)
        self.assertIn("neighbor_weight_sum", st.columns)
        # interaction features present
        self.assertIn("pf_conf_x_rel_md", st.columns)
        self.assertIn("ncc_peak", st.columns)
        self.assertIn("nb_offset_x_ok", st.columns)
        # FEATURE_COLS includes the new ones
        for col in ("neighbor_tvt_wmean", "pf_conf_x_rel_md", "ncc_lag_consistency"):
            self.assertIn(col, FEATURE_COLS_DEFAULT)

    def test_unseen_target_uses_donors_without_mutating_index(self) -> None:
        target = self._well()
        donor = self._well(y=10.0, tvt=105.0)
        idx = build_spatial_index([("donor", donor)])
        donor_node = idx.nodes["donor"]
        observed_before = donor_node.tvt.copy()
        out = attach_spatial_features(self._stations(target), target, idx, "test_only")
        self.assertEqual(float(out["neighbor_count"].iloc[0]), 1.0)
        self.assertTrue(np.isfinite(out["neighbor_tvt"]).all())
        self.assertTrue(np.isfinite(out["neighbor_tvt_wmean"]).all())
        self.assertEqual(list(idx.nodes), ["donor"])
        self.assertIs(idx.nodes["donor"], donor_node)
        np.testing.assert_array_equal(donor_node.tvt, observed_before)

    def test_hidden_labels_do_not_change_transfers(self) -> None:
        target = self._well()
        donor = self._well(y=10.0, tvt=105.0)
        before = attach_spatial_features(
            self._stations(target), target,
            build_spatial_index([("target", target), ("donor", donor)]), "target",
        )
        # Includes labeled training toes and held-out validation toes. No TVT
        # value is an observation under the default graph policy.
        target["TVT"] = -1e9
        donor["TVT"] = 1e9
        after = attach_spatial_features(
            self._stations(target), target,
            build_spatial_index([("target", target), ("donor", donor)]), "target",
        )
        pd.testing.assert_frame_equal(before, after)

    def test_hidden_toe_labels_do_not_change_station_features(self) -> None:
        target = self._well()
        donor = self._well(y=10.0, tvt=105.0)
        target["GR"] = 80 + 5 * np.sin(target["MD"] / 6.0)
        sp = resolve_middle_out(target, well_id="target")
        before = build_station_frame(
            target, None, sp, station_idx=sp.predict_idx,
            spatial_index=build_spatial_index([("target", target), ("donor", donor)]),
        )
        target.loc[target["TVT_input"].isna(), "TVT"] = -1e9
        donor.loc[donor["TVT_input"].isna(), "TVT"] = 1e9
        after = build_station_frame(
            target, None, sp, station_idx=sp.predict_idx,
            spatial_index=build_spatial_index([("target", target), ("donor", donor)]),
        )
        features_before, columns_before = feature_matrix(before, impute=False)
        features_after, columns_after = feature_matrix(after, impute=False)
        self.assertEqual(columns_before, columns_after)
        np.testing.assert_array_equal(features_before, features_after)

    def test_observed_labels_remain_the_transfer_source(self) -> None:
        donor = self._well(y=10.0, tvt=105.0)
        node = well_spatial_node_from_lateral("donor", donor)
        np.testing.assert_array_equal(node.tvt, donor["TVT_input"].to_numpy())
        self.assertEqual(node.label_policy, "observed_only")
        self.assertTrue(np.isnan(node.tvt[8:]).all())

    def test_missing_observations_never_fall_back_to_full_labels(self) -> None:
        target = self._well()
        for mode in ("missing", "null", "nonfinite"):
            with self.subTest(mode=mode):
                donor = self._well(y=10.0)
                if mode == "missing":
                    donor = donor.drop(columns="TVT_input")
                else:
                    donor["TVT_input"] = np.nan if mode == "null" else np.inf
                idx = build_spatial_index([("donor", donor)])
                self.assertTrue(np.isnan(idx.nodes["donor"].tvt).all())
                out = attach_spatial_features(self._stations(target), target, idx, "target")
                self.assertTrue(out["neighbor_tvt"].isna().all())
                self.assertTrue(out["neighbor_tvt_wmean"].isna().all())
                self.assertTrue((out["neighbor_transfer_ok"] == 0.0).all())

    def test_full_donor_labels_require_explicit_policy(self) -> None:
        donor = self._well(y=10.0)
        observed = build_spatial_index([("donor", donor)])
        full = build_spatial_index([("donor", donor)], label_policy="full_tvt")
        self.assertEqual(observed.label_policy, "observed_only")
        self.assertEqual(full.label_policy, "full_tvt")
        self.assertEqual(full.nodes["donor"].label_policy, "full_tvt")
        self.assertTrue(np.isnan(observed.nodes["donor"].tvt[8:]).all())
        np.testing.assert_array_equal(full.nodes["donor"].tvt, donor["TVT"].to_numpy())
        with self.assertRaises(ValueError):
            build_spatial_index([], label_policy="typo")

    def test_self_is_excluded_even_with_external_query(self) -> None:
        target = self._well()
        idx = build_spatial_index([("target", target)])
        out = attach_spatial_features(self._stations(target), target, idx, "target")
        self.assertTrue((out["neighbor_count"] == 0.0).all())
        self.assertTrue(out["neighbor_tvt"].isna().all())

    def test_missing_coordinates_do_not_create_origin_or_labels(self) -> None:
        target = self._well()
        donor = self._well(y=10.0)
        invalid = donor.copy()
        invalid["X"] = np.nan
        idx = build_spatial_index([("invalid", invalid), ("donor", donor)])
        self.assertTrue(np.isnan(idx.nodes["invalid"].cx))
        self.assertTrue(np.isnan(idx.nodes["invalid"].cy))
        query = well_spatial_node_from_lateral("target", target)
        self.assertEqual([wid for wid, _ in idx.neighbors("target", query_node=query)], ["donor"])
        target.loc[3, "X"] = np.nan
        target.loc[7, "Y"] = np.inf
        out = attach_spatial_features(self._stations(target), target, idx, "target")
        for column in ("neighbor_tvt", "neighbor_xy_dist", "neighbor_tvt_wmean"):
            self.assertTrue(out.loc[[3, 7], column].isna().all())
        self.assertTrue((out.loc[[3, 7], "neighbor_transfer_ok"] == 0.0).all())
        all_missing = target.copy()
        all_missing["X"] = np.nan
        out_missing = attach_spatial_features(self._stations(all_missing), all_missing, idx, "target")
        self.assertTrue((out_missing["neighbor_count"] == 0.0).all())

    def test_neighbor_ties_are_independent_of_index_order(self) -> None:
        target = self._well()
        a = self._well(y=10.0)
        b = self._well(y=-10.0)
        query = well_spatial_node_from_lateral("target", target)
        for donors in ([("b", b), ("a", a)], [("a", a), ("b", b)]):
            idx = build_spatial_index(donors)
            self.assertEqual([wid for wid, _ in idx.neighbors("target", query_node=query)], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
