"""Prepare, fit and independently score common nodal inputs on development wells."""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
from io import BytesIO
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from threadpoolctl import threadpool_limits

from rwt.block_model import FEATURE_NAMES, INPUT_COLUMNS, apply_medians, fit_train_medians
from rwt.load import normalize_curve_nulls
from rwt.nodal_bundle import HOPS, MODEL_COLUMNS, REFERENCE_CHANNELS, build_well_bundle, trace_node

ARMS = ("graph", "tree", "forest", "neural")
KEYS = ("well_index", "first_station", "row_count", "fold")


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


@contextmanager
def _artifact(path: Path, reads):
    """Give a parser the same upstream artifact bytes identified by its read receipt."""
    if reads is None:
        with path.open("rb") as handle:
            yield handle
        return
    snapshot = path.read_bytes()
    with BytesIO(snapshot) as handle:
        yield handle
    reads.append({"artifact": path.name, "sha256": hashlib.sha256(snapshot).hexdigest()})


def _write(path: Path, value):
    with path.open("x", encoding="utf-8") as out:
        json.dump(value, out, indent=2, allow_nan=False)


def _save(path: Path, **arrays):
    with path.open("xb") as out:
        np.savez_compressed(out, **arrays)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _table(contract, well, source):
    return Path(contract["data_root"]) / "train" / f"{well}__{source}.csv"


def prepare_run(run: Path):
    started = time.perf_counter()
    contract = _json(run / "contract.json")
    wells = contract["wells"]
    blocks, raw, references, edge_source, edge_target, weights, initial = [], [], [], [], [], [], []
    block_ptr, lateral_ptr, typewell_ptr = [0], [0], [0]
    for well in wells:
        lateral = pd.read_csv(_table(contract, well, "horizontal_well"), usecols=INPUT_COLUMNS,
                              float_precision="round_trip")
        ref = pd.read_csv(_table(contract, well, "typewell"), usecols=REFERENCE_CHANNELS,
                          float_precision="round_trip")
        item = build_well_bundle(lateral, ref)
        raw.append(item.pop("lateral"))
        references.append(item.pop("typewell"))
        first = block_ptr[-1]
        end = first + len(item["row_count"])
        edge_source.append(np.arange(first, end - 1, dtype=np.int64))
        edge_target.append(np.arange(first + 1, end, dtype=np.int64))
        weights.append(item.pop("continuity"))
        initial.append(item.pop("initial_continuity"))
        blocks.append(item)
        block_ptr.append(end)
        lateral_ptr.append(lateral_ptr[-1] + len(lateral))
        typewell_ptr.append(typewell_ptr[-1] + len(ref))
    data = {key: np.concatenate([item[key] for item in blocks]) for key in blocks[0]}
    groups = np.repeat(np.arange(len(wells), dtype=np.int32), np.diff(block_ptr))
    fold = np.full(len(groups), -1, dtype=np.int16)
    for i, (_, held) in enumerate(GroupKFold(contract["folds"]).split(groups, groups=groups)):
        fold[held] = i
    data.update(well_ids=np.asarray(wells), well_index=groups, fold=fold, block_ptr=np.asarray(block_ptr),
                lateral_ptr=np.asarray(lateral_ptr), typewell_ptr=np.asarray(typewell_ptr),
                lateral_values=np.concatenate(raw), typewell_values=np.concatenate(references),
                edge_source=np.concatenate(edge_source), edge_target=np.concatenate(edge_target),
                edge_weight=np.concatenate(weights), edge_initial_weight=np.concatenate(initial))
    _save(run / "bundle.npz", **data)
    changed = np.flatnonzero(data["edge_initial_weight"] != data["edge_weight"])
    example = int(data["edge_target"][changed[0]]) if len(changed) else 0
    _write(run / "bundle.json", {
        "schema": "rwt.nodal-bundle.v1", "well_ids": wells, "features": list(MODEL_COLUMNS),
        "raw_channels": {"lateral": list(INPUT_COLUMNS), "typewell": list(REFERENCE_CHANNELS)},
        "units": {"MD":"ft", "X":"ft", "Y":"ft", "Z":"ft", "TVT_input":"ft", "TVT":"ft",
                  "GR":"source gamma-ray scale"},
        "reading_key": ["well_id", "table", "original_zero_based_row", "channel"],
        "window_key": ["well_id", "first_station", "row_count"],
        "source_sha256": contract["input_files"],
        "incidence": "block_ptr/lateral_ptr/typewell_ptr delimit each well; first_station and row_count index raw lateral rows",
        "lineage": {
            "raw": "original parsed numeric values; source file hashes bind original CSV bytes",
            "drafting": "45-degree projection and cross coordinate share the original X/Y parents",
            "candidate_cost": "block GR rows, visible heel calibration and paired typewell TVT/GR",
            "conditional_messages": "preceding and following edge_weight values, at most four connected edges each way",
            "graph_prediction": "all candidate layers in this well through a full path backtrace",
        }, "edges": int(len(data["edge_weight"])), "cut_edges": int((data["edge_weight"] == 0).sum()),
        "revised_edges": int(np.sum(data["edge_weight"] != data["edge_initial_weight"])),
        "max_message_hops": HOPS, "blocks": int(len(fold)), "toe_rows": int(data["row_count"].sum()),
        "example_trace": trace_node(data, wells[groups[example]], int(data["first_station"][example])),
        "hidden_tvt_read": False, "supervised_fits": 0, "visible_heel_calibration":"per well",
        "seconds": time.perf_counter() - started,
    })


def _training_targets(contract, wells, ptr, first, count, last, train_wells):
    residuals = []
    for wi in train_wells:
        truth = _truth(contract, wells[wi])
        run = slice(ptr[wi], ptr[wi + 1])
        starts = first[run]
        n = count[run]
        if starts[-1] + n[-1] != len(truth) or not np.isfinite(truth[starts[0]:]).all():
            raise ValueError("training well lacks complete finite toe labels")
        means = np.add.reduceat(truth, starts) / n
        residuals.append(means - last[run])
    return np.concatenate(residuals)


def _truth(contract, well):
    frame = pd.read_csv(_table(contract, well, "horizontal_well"), usecols=["TVT"], float_precision="round_trip")
    return normalize_curve_nulls(frame["TVT"])


def _train(arm, x, target, valid_x, seed):
    medians = fit_train_medians(x)
    x = apply_medians(x, medians)
    valid_x = apply_medians(valid_x, medians)
    epochs = 0
    offset, scale = 0., 1.
    if arm == "neural":
        scaler = StandardScaler().fit(x)
        x = scaler.transform(x)
        valid_x = scaler.transform(valid_x)
        offset, scale = float(target.mean()), max(float(target.std()), 1.)
        model = MLPRegressor(hidden_layer_sizes=(32, 16), solver="adam", batch_size=128,
                             max_iter=24, n_iter_no_change=25, tol=0., random_state=seed)
    elif arm == "tree":
        model = DecisionTreeRegressor(max_leaf_nodes=31, min_samples_leaf=12, random_state=seed)
    elif arm == "forest":
        model = RandomForestRegressor(n_estimators=48, max_leaf_nodes=127, min_samples_leaf=8,
                                      max_features="sqrt", n_jobs=2, random_state=seed)
    else:
        raise ValueError(f"unknown model arm {arm}")
    model.fit(x, (target - offset) / scale)
    prediction = offset + scale * model.predict(valid_x)
    if arm == "neural":
        epochs = int(model.n_iter_)
    return prediction, epochs


def predict_run(run: Path, arm: str, *, reads=None):
    if arm not in ARMS:
        raise ValueError("unregistered nodal method")
    started = time.perf_counter()
    contract = _json(run / "contract.json")
    history = []
    with _artifact(run / "bundle.npz", reads) as stream, np.load(stream, allow_pickle=False) as data:
        keys = {key:data[key] for key in KEYS}
        if arm == "graph":
            augmented, plain = data["graph_prediction"], data["graph_control"]
        else:
            x, last, ptr, wells = data["features"], data["last_known"], data["block_ptr"], data["well_ids"]
            augmented, plain = np.empty(len(x)), np.empty(len(x))
            for fold in range(contract["folds"]):
                valid = keys["fold"] == fold
                train = ~valid
                train_wi = np.unique(keys["well_index"][train])
                valid_wi = np.unique(keys["well_index"][valid])
                if np.intersect1d(train_wi, valid_wi).size or not train_wi.size or not valid_wi.size:
                    raise ValueError("non-disjoint or empty well partition")
                target = _training_targets(contract, wells, ptr, keys["first_station"], keys["row_count"], last, train_wi)
                for tag, ncols, output in (("bundle", x.shape[1], augmented), ("control", len(FEATURE_NAMES), plain)):
                    tick = time.perf_counter()
                    with threadpool_limits(limits=2):
                        pred, epochs = _train(arm, x[train, :ncols], target, x[valid, :ncols], contract["seed"] + fold)
                    output[valid] = last[valid] + pred
                    history.append({"fold":fold, "view":tag, "columns":ncols, "epochs":epochs,
                                    "training_wells":wells[train_wi].tolist(), "held_out_wells":wells[valid_wi].tolist(),
                                    "train_blocks":int(train.sum()), "validation_blocks":int(valid.sum()),
                                    "fit_and_predict_seconds":time.perf_counter()-tick})
    if not np.isfinite(augmented).all() or not np.isfinite(plain).all():
        raise ValueError("nonfinite selected depth")
    _save(run / f"{arm}.npz", **keys, prediction=augmented, control_prediction=plain)
    bundle_digest = reads[-1]["sha256"] if reads is not None else _sha(run / "bundle.npz")
    _write(run / f"{arm}.json", {"arm":arm, "bundle_sha256":bundle_digest, "fits":len(history),
                                  "fits_by_fold":history, "seconds":time.perf_counter()-started,
                                  "prediction_sha256":_sha(run / f"{arm}.npz")})


def audit_run(run: Path, *, reads=None) -> dict:
    """Reopen the raw wells, prove exact toe coverage, then score sealed predictions."""
    contract = _json(run / "contract.json")
    with _artifact(run / "bundle.npz", reads) as stream, np.load(stream, allow_pickle=False) as data:
        keys = {key:data[key] for key in KEYS}
        wells, ptr, last = data["well_ids"], data["block_ptr"], data["last_known"]
        lateral_ptr, reference_ptr = data["lateral_ptr"], data["typewell_ptr"]
        raw, reference = data["lateral_values"], data["typewell_values"]
        edge_from, edge_to = data["edge_source"], data["edge_target"]
    bundle_digest = reads[-1]["sha256"] if reads is not None else _sha(run / "bundle.npz")
    if wells.tolist() != contract["wells"] or not np.array_equal(
            keys["well_index"], np.repeat(np.arange(len(wells)), np.diff(ptr))):
        raise ValueError("well identity or node ranges changed")
    edge_starts = np.flatnonzero(np.diff(keys["well_index"]) == 0)
    if not np.array_equal(edge_from, edge_starts) or not np.array_equal(edge_to, edge_starts + 1):
        raise ValueError("edge identities differ from within-well adjacency")
    replay_fold = np.full(len(last), -1, dtype=int)
    for f, (_, valid) in enumerate(GroupKFold(contract["folds"]).split(last, groups=keys["well_index"])):
        replay_fold[valid] = f
    if not np.array_equal(keys["fold"], replay_fold):
        raise ValueError("fold assignment differs from independent grouped split")
    predictions = {"last_visible":last}
    for arm in ARMS:
        with _artifact(run / f"{arm}.json", reads) as stream:
            receipt = json.load(stream)
        if receipt["bundle_sha256"] != bundle_digest:
            raise ValueError("arm consumed a different bundle")
        with _artifact(run / f"{arm}.npz", reads) as stream, np.load(stream, allow_pickle=False) as file:
            if any(not np.array_equal(file[k], v) for k,v in keys.items()):
                raise ValueError("arm prediction keys differ")
            for name, key in ((arm, "prediction"), (arm + "_control", "control_prediction")):
                prediction = file[key]
                if prediction.shape != last.shape or not np.isfinite(prediction).all():
                    raise ValueError("incomplete prediction vector")
                predictions[name] = prediction
        prediction_digest = reads[-1]["sha256"] if reads is not None else _sha(run / f"{arm}.npz")
        if receipt["prediction_sha256"] != prediction_digest:
            raise ValueError("arm consumed a different prediction artifact")
    by_well = {}
    by_fold = {str(i): {k:0. for k in predictions} for i in range(contract["folds"])}
    n_rows = 0
    for wi, well in enumerate(wells):
        lateral = pd.read_csv(_table(contract, well, "horizontal_well"), usecols=[*INPUT_COLUMNS,"TVT"],
                              float_precision="round_trip")
        ref = pd.read_csv(_table(contract, well, "typewell"), usecols=REFERENCE_CHANNELS, float_precision="round_trip")
        if (not np.array_equal(lateral.loc[:, list(INPUT_COLUMNS)].to_numpy(float),
                               raw[lateral_ptr[wi]:lateral_ptr[wi+1]], equal_nan=True)
                or not np.array_equal(ref.loc[:, list(REFERENCE_CHANNELS)].to_numpy(float),
                                      reference[reference_ptr[wi]:reference_ptr[wi+1]], equal_nan=True)):
            raise ValueError("original measurement identity/value mismatch")
        known = np.flatnonzero(np.isfinite(normalize_curve_nulls(lateral["TVT_input"])))
        if not len(known) or not np.array_equal(known, np.arange(len(known))):
            raise ValueError("visible prefix is not contiguous")
        s = slice(ptr[wi], ptr[wi+1])
        first, counts = keys["first_station"][s], keys["row_count"][s]
        expected = np.arange(len(known), len(lateral), 8)
        if (not np.array_equal(first, expected) or not np.array_equal(counts, np.minimum(8, len(lateral)-expected))
                or not np.all(last[s] == lateral["TVT_input"].iloc[known[-1]])):
            raise ValueError("original block keys or anchor mismatch")
        truth = normalize_curve_nulls(lateral["TVT"])[len(known):]
        if not np.isfinite(truth).all():
            raise ValueError("finite truth required on every selected original row")
        sse = {k:float(np.sum(np.square(np.repeat(v[s], counts) - truth), dtype=np.float64))
               for k,v in predictions.items()}
        f = str(int(keys["fold"][ptr[wi]]))
        for k,v in sse.items():
            by_fold[f][k] += v
        by_well[str(well)] = {"rows":len(truth), "fold":int(f), "sse_ft2":sse}
        n_rows += len(truth)
    totals = {k:sum(row["sse_ft2"][k] for row in by_well.values()) for k in predictions}
    methods = {k:{"sse_ft2":v, "rmse_ft":float(np.sqrt(v/n_rows)), "rows":n_rows} for k,v in totals.items()}
    merit = {k:{"beats_last_visible":totals[k] < totals["last_visible"],
                "beats_matched_control":totals[k] < totals[k+"_control"],
                "wells_improved_vs_control":sum(row["sse_ft2"][k] < row["sse_ft2"][k+"_control"]
                                                  for row in by_well.values())} for k in ARMS}
    return {"n_wells":len(wells), "n_blocks":len(last), "n_stations":n_rows, "methods":methods,
            "merit":merit, "by_fold":by_fold, "by_well":by_well, "confirmation_evaluated":False}
