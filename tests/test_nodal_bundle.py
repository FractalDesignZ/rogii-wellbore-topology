"""Check observation identity, backward weight revision and held-well scoring on synthetic wells."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import nodal_worker
from rwt.block_model import FEATURE_NAMES, INPUT_COLUMNS
from rwt.nodal_bundle import (
    MODEL_COLUMNS, REFERENCE_CHANNELS, build_well_bundle, revise_edge_weights, trace_node,
    window_messages,
)


def synthetic_well(index):
    heel = 15 + index
    station = np.arange(heel + 89 + 9 * index)
    depth = 150.0 + index + 0.15 * station + 2 * np.sin(station / 12)
    axis = np.arange(0.0, 325.0)
    reference = pd.DataFrame({"TVT": axis, "GR": 50 + 12 * np.sin(axis / 7)})
    lateral = pd.DataFrame({
        "MD": 10_000.0 + 3 * station, "X": 1000.0 * index + 2 * station,
        "Y": 0.4 * station, "Z": -400.0 + station / 9,
        "GR": np.interp(depth, axis, reference["GR"]),
        "TVT_input": np.where(station < heel, depth, np.nan), "TVT": depth,
    })
    lateral.loc[0, "Z"] = -9999.25  # Finite coordinate in a channel with no curve null code.
    lateral.loc[heel + 3, "GR"] = -9999.25  # Curve null retained in the original reading.
    return lateral, reference


def test_later_observation_revises_earlier_edge_only_with_a_connected_context():
    local = np.zeros((4, 7))
    local[0, 1:] = 10.0  # The first block supports the lowest candidate.
    measured = local.copy()
    measured[3, :-1] = 10.0  # Only the last block changes, two blocks beyond edge (0,1).
    continuity = np.ones(3)

    original, original_forward, _ = revise_edge_weights(local, continuity)
    revised, forward, _ = revise_edge_weights(measured, continuity)
    np.testing.assert_array_equal(original, continuity)
    assert 0 < revised[0] < original[0]
    np.testing.assert_array_equal(forward[:3], original_forward[:3])
    assert revise_edge_weights(measured, continuity, hops=1)[0][0] == original[0]

    cut = np.array([1.0, 0.0, 1.0])
    before_cut, _, before_back = revise_edge_weights(local, cut)
    after_cut, _, after_back = revise_edge_weights(measured, cut)
    np.testing.assert_array_equal(after_cut[:2], before_cut[:2])
    np.testing.assert_array_equal(after_back[:2], before_back[:2])
    assert after_cut[1] == 0
    _, _, left, right = window_messages(measured, after_cut)
    np.testing.assert_array_equal(left, [0, 1, 0, 1])
    np.testing.assert_array_equal(right, [1, 0, 1, 0])


def test_bundle_preserves_raw_values_and_is_indifferent_to_hidden_lateral_targets():
    lateral, reference = synthetic_well(0)
    expected = build_well_bundle(lateral, reference)
    lateral["TVT"] = np.nan
    lateral["TVT_true"] = np.arange(len(lateral)) * 1234567.0
    lateral["ANCC"] = -3.0
    actual = build_well_bundle(lateral, reference)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
    np.testing.assert_array_equal(actual["lateral"], lateral.loc[:, list(INPUT_COLUMNS)].to_numpy())
    np.testing.assert_array_equal(actual["typewell"], reference.loc[:, list(REFERENCE_CHANNELS)].to_numpy())

    start = actual["first_station"][0]
    center = actual["center_station"]
    anchor = lateral.iloc[start - 1]
    # Invert the 45-degree rotation from the model columns, independently of its helper.
    along, cross = actual["features"][:, [
        MODEL_COLUMNS.index("draft_along_ft"), MODEL_COLUMNS.index("draft_cross_ft"),
    ]].T.astype(float)
    np.testing.assert_allclose(anchor.X + (along - cross) / np.sqrt(2), lateral.X.iloc[center], atol=2e-5)
    np.testing.assert_allclose(anchor.Y + (along + cross) / np.sqrt(2), lateral.Y.iloc[center], atol=2e-5)


def test_three_well_worker_views_match_original_rows_and_scored_targets(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    tables = data_root / "train"
    tables.mkdir(parents=True)
    run = tmp_path / "run"
    run.mkdir()
    wells = ["Z", "A", "M"]  # Same local station numbering, deliberately unsorted IDs.
    sources = [synthetic_well(i) for i in range(len(wells))]
    hashes = {}
    for wid, (lateral, reference) in zip(wells, sources):
        for suffix, table in (("horizontal_well", lateral), ("typewell", reference)):
            path = tables / f"{wid}__{suffix}.csv"
            table.to_csv(path, index=False)
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    contract = {"data_root": str(data_root), "wells": wells, "folds": 3, "seed": 42, "input_files": hashes}
    (run / "contract.json").write_text(json.dumps(contract), encoding="utf-8")

    def stage(name):
        monkeypatch.setattr(sys, "argv", ["nodal_worker", name, str(run)])
        nodal_worker.main()

    stage("prepare")
    with np.load(run / "bundle.npz", allow_pickle=False) as bundle:
        for wi, (wid, (lateral, reference)) in enumerate(zip(wells, sources)):
            begin, end = bundle["block_ptr"][wi:wi+2]
            blocks = slice(begin, end)
            stand_alone = build_well_bundle(lateral, reference)
            np.testing.assert_array_equal(bundle["features"][blocks], stand_alone["features"])
            np.testing.assert_array_equal(bundle["graph_prediction"][blocks], stand_alone["graph_prediction"])
            first = int(lateral.TVT_input.count())
            trace = trace_node(bundle, wid, first)
            assert trace["node"] == [wid, first]
            assert trace["block_rows_half_open"] == [first, first + 8]
            assert trace["center_gr"] == {"key": [wid, "lateral", first + 3, "GR"], "parsed_value": -9999.25}
            assert trace["heel_tvt"]["parsed_value"] == lateral.TVT_input.iloc[first - 1]
            assert trace["typewell_row_range_half_open"] == [0, len(reference)]
            for reading, axis in zip(trace["center_xyz"], ("X", "Y", "Z")):
                assert reading["key"] == [wid, "lateral", first + 3, axis]
                assert reading["parsed_value"] == lateral[axis].iloc[first + 3]
            assert [(e["from"], e["to"]) for e in trace["incident_edges"]] == [([wid, first], [wid, first + 8])]
            assert len(np.unique(bundle["fold"][blocks])) == 1
        assert np.array_equal(bundle["well_index"][bundle["edge_source"]], bundle["well_index"][bundle["edge_target"]])
        with pytest.raises(ValueError):
            trace_node(bundle, wells[0], int(sources[0][0].TVT_input.count()) + 1)

    for arm in nodal_worker.ARMS:
        stage(arm)
        receipt = json.loads((run / f"{arm}.json").read_text(encoding="utf-8"))
        assert receipt["fits"] == (0 if arm == "graph" else 6)
        for fit in receipt["fits_by_fold"]:
            assert set(fit["training_wells"]).isdisjoint(fit["held_out_wells"])
            assert len(fit["training_wells"]) == 2
            assert len(fit["held_out_wells"]) == 1
            assert fit["columns"] == (len(MODEL_COLUMNS) if fit["view"] == "bundle" else len(FEATURE_NAMES))
    stage("audit")
    scores = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    assert scores["n_wells"] == 3
    assert scores["n_stations"] == sum(lateral.TVT_input.isna().sum() for lateral, _ in sources)
    for arm in nodal_worker.ARMS:
        with np.load(run / f"{arm}.npz", allow_pickle=False) as pred:
            for wi, (wid, (lateral, _)) in enumerate(zip(wells, sources)):
                part = pred["well_index"] == wi
                truth = lateral.TVT[lateral.TVT_input.isna()].to_numpy()
                for key, method in (("prediction", arm), ("control_prediction", arm + "_control")):
                    station_prediction = np.repeat(pred[key][part], pred["row_count"][part])
                    sse = np.square(station_prediction - truth).sum()
                    assert scores["by_well"][wid]["sse_ft2"][method] == pytest.approx(sse)
