"""Experimental BASE92 block residual model with grouped development OOF.

This original standard feature route uses GR, visible heel and geometry only.
It is independent of optional operator providers and makes no admission claim.
"""
from __future__ import annotations
import warnings
import argparse
import hashlib
import importlib.metadata
import json
import os
from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Any, Callable, Mapping

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from rwt.gr_match import fit_heel_gr_affine
from rwt.load import apply_curve_nulls_frame
from rwt.load import normalize_curve_nulls

INPUT_COLUMNS = ("MD", "X", "Y", "Z", "GR", "TVT_input")
OFFSETS = np.arange(-100., 101., 5.)
BASE_NAMES = (
    "last_tvt", "last_md", "last_x", "last_y", "last_z", "n_heel", "heel_tvt_slope", "heel_gr_median", "heel_gr_std", "gr_cal_a", "gr_cal_b", "gr_noise",
    "center_md", "relative_md", "relative_station", "toe_fraction", "row_count", "block_md_span",
    "center_x", "center_y", "center_z", "relative_x", "relative_y", "relative_z", "block_dx", "block_dy", "block_dz", "dx_dmd", "dy_dmd", "dz_dmd", "geometry_support",
    "gr_mean", "gr_std", "gr_min", "gr_max", "gr_q25", "gr_median", "gr_q75", "gr_support", "gr_first", "gr_last", "gr_change", "gr_vs_heel", "gr_scaled_heel",
    "profile_argmin_offset", "profile_soft_offset", "profile_soft_spread", "profile_entropy", "profile_best_cost", "profile_second_gap", "profile_support",
)
FEATURE_NAMES = BASE_NAMES + tuple(f"gr_cost_{int(offset):+d}" for offset in OFFSETS)
RECIPE = {"block_rows": 8, "profile_offsets_ft": OFFSETS.tolist(), "profile_student_df": 4., "heel_window_md_ft": 500., "minimum_gr_sigma": 1., "affine_slope_clip": [.45, 2.25], "minimum_heel_pairs": 8,
          "mapping": "center label at lower-middle raw station; one block residual broadcast to every original toe row represented by that block; observed heel locked by API",
          "missing": "nonfinite features retained asNaN; fold-train-only median fills, all-missing train columns fill0; GR/TVT_input null curves decoded but finite coordinates preserved",
          "profile": "mean Student-t cost on finite blockGR at each constantTVT last+offset, scaled by heel residualMAD floor1; out-of-typewell states or all-missingGR costNaN; descriptors use supported costs only"}


def _padded(values, size=8):
    n = len(values)
    out = np.full(((n + size - 1) // size) * size, np.nan)
    out[:n] = values
    return out.reshape(-1, size)


def _finite_stats(values):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return (np.nanmean(values, axis=1), np.nanstd(values, axis=1), np.nanmin(values, axis=1), np.nanmax(values, axis=1), *np.nanquantile(values, [.25, .5, .75], axis=1))


def build_block_features(lateral: pd.DataFrame, typewell: pd.DataFrame):
    """Reads only the allowlisted input columns, with no cross-well state."""
    if set(INPUT_COLUMNS) - set(lateral):
        raise ValueError("required input columns absent")
    frame = apply_curve_nulls_frame(lateral.loc[:, INPUT_COLUMNS], columns=("GR", "TVT_input"))
    data = {name: frame[name].to_numpy(dtype=float, copy=True) for name in INPUT_COLUMNS}
    for values in data.values():
        values[~np.isfinite(values)] = np.nan
    known = np.isfinite(data["TVT_input"])
    rows = np.flatnonzero(known)
    if not len(rows) or not known[:rows[-1] + 1].all():
        raise ValueError("nonempty contiguous visible heel required")
    md = data["MD"]
    if not np.isfinite(md).all() or (np.diff(md) <= 0).any():
        raise ValueError("finite increasing MD required")
    heel_end = int(rows[-1])
    start = heel_end + 1
    n_toe = len(frame) - start
    if n_toe == 0:
        raise ValueError("no toe rows")
    first = np.arange(start, len(frame), 8, dtype=np.int32)
    count = np.minimum(8, len(frame) - first).astype(np.int32)
    center = first + (count - 1) // 2
    last = data["TVT_input"][heel_end]
    near = rows[md[rows] >= md[heel_end] - 500.]
    slopes = np.diff(data["TVT_input"][near]) / np.diff(md[near])
    slopes = slopes[np.isfinite(slopes)]
    drift = float(np.median(slopes)) if len(slopes) else 0.
    book = apply_curve_nulls_frame(typewell.loc[:, ["TVT", "GR"]], columns=("TVT", "GR"))
    book = book[np.isfinite(book.TVT) & np.isfinite(book.GR)].groupby("TVT", sort=True, as_index=False).GR.mean()
    tw_t, tw_g = book.TVT.to_numpy(float), book.GR.to_numpy(float)
    heel_gr = data["GR"][rows]
    finite_gr = heel_gr[np.isfinite(heel_gr)]
    heel_med = float(np.median(finite_gr)) if len(finite_gr) else np.nan
    heel_std = float(np.std(finite_gr)) if len(finite_gr) else np.nan
    a, b, sigma = 1., 0., 1.
    reference = np.full(len(OFFSETS), np.nan)
    if len(tw_t) >= 2:
        calibrated_known = np.where(known & (data["TVT_input"] >= tw_t[0]) & (data["TVT_input"] <= tw_t[-1]), data["TVT_input"], np.nan)
        pairs = np.isfinite(calibrated_known) & np.isfinite(data["GR"])
        a, b = fit_heel_gr_affine(tw_g, tw_t, data["GR"], calibrated_known)
        a = float(np.clip(a, .45, 2.25))
        if pairs.sum() >= 8:
            reference_heel = np.interp(calibrated_known[pairs], tw_t, tw_g)
            b = float(np.median(data["GR"][pairs] - a * reference_heel))
            residual = data["GR"][pairs] - (a * reference_heel + b)
            sigma = max(1., float(1.4826 * np.median(np.abs(residual - np.median(residual)))))
        supported = (last + OFFSETS >= tw_t[0]) & (last + OFFSETS <= tw_t[-1])
        reference[supported] = a * np.interp(last + OFFSETS[supported], tw_t, tw_g) + b
    values = {}
    constants = (last, md[heel_end], data["X"][heel_end], data["Y"][heel_end], data["Z"][heel_end], len(rows), drift, heel_med, heel_std, a, b, sigma)
    for name, value in zip(BASE_NAMES[:12], constants):
        values[name] = np.full(len(first), value)
    values.update(center_md=md[center], relative_md=md[center] - md[heel_end], relative_station=center - heel_end,
                  toe_fraction=(center - start) / max(1, n_toe - 1), row_count=count,
                  block_md_span=md[first + count - 1] - md[first])
    for axis in ("X", "Y", "Z"):
        low = axis.lower()
        values[f"center_{low}"] = data[axis][center]
        values[f"relative_{low}"] = data[axis][center] - data[axis][heel_end]
        delta = data[axis][first + count - 1] - data[axis][first]
        values[f"block_d{low}"] = delta
        span = values["block_md_span"]
        values[f"d{low}_dmd"] = np.divide(delta, span, out=np.full(len(first), np.nan), where=span > 0)
    geo = np.isfinite(np.column_stack([data[k] for k in ("X", "Y", "Z")])).all(axis=1)
    values["geometry_support"] = np.nansum(_padded(geo[start:].astype(float)), axis=1) / count
    gr_blocks = _padded(data["GR"][start:])
    for name, value in zip(("gr_mean", "gr_std", "gr_min", "gr_max", "gr_q25", "gr_median", "gr_q75"), _finite_stats(gr_blocks)):
        values[name] = value
    values["gr_support"] = np.isfinite(gr_blocks).sum(axis=1) / count
    values["gr_first"], values["gr_last"] = data["GR"][first], data["GR"][first + count - 1]
    values["gr_change"] = values["gr_last"] - values["gr_first"]
    values["gr_vs_heel"] = values["gr_mean"] - heel_med
    values["gr_scaled_heel"] = values["gr_vs_heel"] / sigma
    error = (gr_blocks[:, :, None] - reference[None, None, :]) / sigma
    kernel = 2.5 * np.log1p(error ** 2 / 4.)
    support = np.isfinite(kernel).sum(axis=1)
    costs = np.divide(np.nansum(kernel, axis=1), support, out=np.full((len(first), 41), np.nan), where=support > 0)
    valid = np.isfinite(costs)
    safe_cost = np.where(valid, costs, np.inf)
    best_index = np.argmin(safe_cost, axis=1)
    has = valid.any(axis=1)
    best = safe_cost[np.arange(len(first)), best_index]
    minimum = np.where(has, best, 0.)
    weights = np.where(valid, np.exp(-np.where(valid, costs - minimum[:, None], 0.)), 0.)
    denominator = weights.sum(axis=1)
    weights = np.divide(weights, denominator[:, None], out=np.zeros_like(weights), where=denominator[:, None] > 0)
    expectation = (weights * OFFSETS).sum(axis=1)
    spread = np.sqrt((weights * (OFFSETS[None, :] - expectation[:, None]) ** 2).sum(axis=1))
    entropy = -(weights * np.log(np.maximum(weights, 1e-30))).sum(axis=1) / np.log(41)
    sorted_cost = np.sort(safe_cost, axis=1)
    gap = np.full(len(first), np.nan)
    has_two = valid.sum(axis=1) >= 2
    gap[has_two] = sorted_cost[has_two, 1] - sorted_cost[has_two, 0]
    values.update(profile_argmin_offset=np.where(has, OFFSETS[best_index], np.nan), profile_soft_offset=np.where(has, expectation, np.nan),
                  profile_soft_spread=np.where(has, spread, np.nan), profile_entropy=np.where(has, entropy, np.nan),
                  profile_best_cost=np.where(has, best, np.nan), profile_second_gap=gap, profile_support=valid.mean(axis=1))
    for i, name in enumerate(FEATURE_NAMES[len(BASE_NAMES):]):
        values[name] = costs[:, i]
    matrix = np.column_stack([values[name] for name in FEATURE_NAMES]).astype(np.float32)
    matrix[~np.isfinite(matrix)] = np.nan
    return matrix, {"first_station": first, "center_station": center.astype(np.int32), "row_count": count,
                    "last_known": np.full(len(first), last), "n_rows": len(frame), "n_known": len(rows), "n_toe": n_toe}


def block_to_station(predictions, row_count):
    predictions, counts = np.asarray(predictions, float), np.asarray(row_count)
    if predictions.ndim != 1 or counts.shape != predictions.shape or (counts <= 0).any() or not np.issubdtype(counts.dtype, np.integer):
        raise ValueError("invalid block mapping")
    return np.repeat(predictions, counts)


def fit_train_medians(matrix):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        medians = np.nanmedian(np.where(np.isfinite(matrix), matrix, np.nan), axis=0)
    return np.where(np.isfinite(medians), medians, 0.).astype(np.float32)


def apply_medians(matrix, medians):
    matrix = np.array(matrix, dtype=np.float32)
    if matrix.ndim != 2 or np.asarray(medians).shape != (matrix.shape[1],):
        raise ValueError("median schema mismatch")
    missing = ~np.isfinite(matrix)
    matrix[missing] = np.broadcast_to(medians, matrix.shape)[missing]
    return matrix


def reconstruct_target(visible_tvt_input, meta, residual):
    """Serving mapping: untouched visible heel; residual block broadcast on toe."""
    visible = normalize_curve_nulls(np.asarray(visible_tvt_input, float))
    residual = np.asarray(residual, float)
    if residual.shape != meta["last_known"].shape or len(visible) != meta["n_rows"]:
        raise ValueError("reconstruction schema mismatch")
    selected = meta["last_known"] + np.where(np.isfinite(residual), residual, 0.)
    prediction = visible.copy()
    prediction[meta["n_known"]:] = block_to_station(selected, meta["row_count"])
    if not np.isfinite(prediction).all():
        raise ValueError("incomplete serving coverage")
    return prediction


MODEL_KIND = "rwt.experimental_base92_block_residual.v1"
FEATURE_SCHEMA = "rwt.base92.v1"
BOOSTING_ROUNDS = 600
MAX_FEATURE_MATRIX_BYTES = 2_000_000_000
LGBM_PARAMETERS = {
    "objective": "regression_l2", "metric": "l2", "num_leaves": 31,
    "min_data_in_leaf": 100, "learning_rate": .05, "num_threads": 4,
    "seed": 20261118, "bagging_seed": 20261118,
    "feature_fraction_seed": 20261118, "data_random_seed": 20261118,
    "feature_fraction": 1., "bagging_fraction": 1., "bagging_freq": 0,
    "lambda_l1": 0., "lambda_l2": 0., "max_bin": 255,
    "deterministic": True, "force_col_wise": True, "verbosity": -1,
}


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path: Path, value: Any) -> None:
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def feature_schema_sha256() -> str:
    payload = {"version": FEATURE_SCHEMA, "names": FEATURE_NAMES, "recipe": RECIPE,
               "inputs": INPUT_COLUMNS, "units": "source feet; GR in source curve units"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {name: _sha(root / name) for name in ("block_model.py", "load.py", "gr_match.py", "priors.py")}


def _controller_source_hashes() -> dict[str, str]:
    return {**_source_hashes(), "execution_graph.py": _sha(Path(__file__).resolve().parent / "execution_graph.py")}


def _metadata() -> dict[str, Any]:
    return {"model_kind": MODEL_KIND, "feature_schema": FEATURE_SCHEMA,
            "feature_schema_sha256": feature_schema_sha256(), "feature_names": list(FEATURE_NAMES),
            "feature_recipe": RECIPE, "lightgbm_parameters": dict(LGBM_PARAMETERS),
            "boosting_rounds_requested": BOOSTING_ROUNDS, "residual_base": "last_visible_TVT_input",
            "residual_clamp": None, "mapping": RECIPE["mapping"], "units": "ft",
            "experimental": True, "predictive_admitted": False,
            "lightgbm_version": importlib.metadata.version("lightgbm")}


def _validate_schema(metadata: dict, medians: np.ndarray) -> None:
    if (metadata.get("model_kind") != MODEL_KIND or metadata.get("feature_schema") != FEATURE_SCHEMA
            or metadata.get("feature_schema_sha256") != feature_schema_sha256()
            or metadata.get("feature_names") != list(FEATURE_NAMES)):
        raise ValueError("incompatible block model/feature schema")
    if (metadata.get("residual_base") != "last_visible_TVT_input"
            or metadata.get("residual_clamp") is not None
            or metadata.get("boosting_rounds_requested") != BOOSTING_ROUNDS
            or metadata.get("lightgbm_parameters") != LGBM_PARAMETERS
            or metadata.get("feature_recipe") != RECIPE or metadata.get("units") != "ft"):
        raise ValueError("incompatible fixed block model recipe")
    if metadata.get("experimental") is not True or metadata.get("predictive_admitted") is not False:
        raise ValueError("this model kind is experimental; predictive admission is unsupported")
    if (np.asarray(medians).shape != (len(FEATURE_NAMES),) or np.asarray(medians).dtype != np.dtype("float32")
            or not np.isfinite(medians).all()):
        raise ValueError("invalid stored training medians")


@dataclass
class BlockPredictor:
    """Experimental predictor with fixed train/serve mapping and stored medians."""
    model: Any
    medians: np.ndarray
    metadata: dict[str, Any]

    def predict_well(self, lateral: pd.DataFrame, typewell: pd.DataFrame, *, well_id: str) -> pd.DataFrame:
        _validate_schema(self.metadata, self.medians)
        matrix, meta = build_block_features(lateral, typewell)
        residual = np.asarray(self.model.predict(apply_medians(matrix, self.medians),
                              num_iteration=BOOSTING_ROUNDS, num_threads=4), dtype=float)
        if residual.shape != meta["last_known"].shape:
            raise ValueError("model prediction/block schema mismatch")
        fallback = ~np.isfinite(residual)
        prediction = reconstruct_target(lateral["TVT_input"].to_numpy(float), meta, residual)
        known = meta["n_known"]
        flags = np.concatenate([np.zeros(known, bool), block_to_station(fallback.astype(float), meta["row_count"]).astype(bool)])
        gr = apply_curve_nulls_frame(lateral.loc[:, ["GR"]], columns=("GR",))["GR"].to_numpy(float)
        return pd.DataFrame({"well_id": str(well_id), "station": np.arange(len(lateral), dtype=np.int64),
                             "MD": lateral["MD"].to_numpy(float), "is_predict": np.arange(len(lateral)) >= known,
                             "tvt": prediction, "fallback_used": flags, "gr_observed": np.isfinite(gr)})


def save_block_predictor(predictor: BlockPredictor, output: Path) -> None:
    """Write a fresh, non-pickle LightGBM text/NPY/JSON bundle."""
    _validate_schema(predictor.metadata, predictor.medians)
    if list(predictor.model.feature_name()) != list(FEATURE_NAMES):
        raise ValueError("model feature names disagree with schema")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    predictor.model.save_model(str(output / "model.txt"), num_iteration=BOOSTING_ROUNDS)
    np.save(output / "medians.npy", predictor.medians, allow_pickle=False)
    metadata = dict(predictor.metadata)
    metadata["source_sha256"] = _source_hashes()
    metadata["artifact_sha256"] = {name: _sha(output / name) for name in ("model.txt", "medians.npy")}
    _json(output / "metadata.json", metadata)


def load_block_predictor(directory: Path) -> BlockPredictor:
    import lightgbm as lgb
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    medians = np.load(directory / "medians.npy", allow_pickle=False)
    _validate_schema(metadata, medians)
    if metadata.get("source_sha256") != _source_hashes():
        raise ValueError("feature implementation/source schema mismatch")
    if metadata.get("lightgbm_version") != importlib.metadata.version("lightgbm"):
        raise ValueError("LightGBM version differs from the saved model")
    if set(metadata.get("artifact_sha256", {})) != {"model.txt", "medians.npy"}:
        raise ValueError("invalid model artifact manifest")
    for name, expected in metadata["artifact_sha256"].items():
        if _sha(directory / name) != expected:
            raise ValueError(f"model artifact SHA mismatch: {name}")
    model = lgb.Booster(model_file=str(directory / "model.txt"))
    if model.num_feature() != len(FEATURE_NAMES) or model.feature_name() != list(FEATURE_NAMES):
        raise ValueError("LightGBM model feature schema mismatch")
    return BlockPredictor(model, medians, metadata)


def evaluate_block_oof(evidence: pd.DataFrame, *, expected_keys: pd.DataFrame | None = None):
    """Exact pooled, per-well and per-fold scores; never drop unscored rows."""
    required = ("well_id", "station", "fold", "y_true", "last_known", "pred_residual", "pred_selected", "fallback_used")
    if any(name not in evidence for name in required) or evidence.empty:
        raise ValueError("missing OOF evidence schema")
    if evidence.well_id.isna().any() or not evidence.well_id.map(lambda value: isinstance(value, str) and bool(value)).all():
        raise ValueError("invalid OOF well identity")
    if (not pd.api.types.is_integer_dtype(evidence.station.dtype) or evidence.station.isna().any()
            or (evidence.station < 0).any()):
        raise ValueError("OOF station keys must be finite nonnegative integers")
    if not pd.api.types.is_integer_dtype(evidence.fold.dtype) or evidence.fold.isna().any():
        raise ValueError("OOF fold IDs must be integers")
    if not pd.api.types.is_bool_dtype(evidence.fallback_used.dtype) or evidence.fallback_used.isna().any():
        raise ValueError("OOF fallback_used must be nonmissing boolean values")
    if evidence.duplicated(["well_id", "station"]).any():
        raise ValueError("duplicate OOF station keys")
    if expected_keys is not None:
        if (len(evidence) != len(expected_keys)
                or not np.array_equal(evidence.well_id.to_numpy(dtype=str), expected_keys.well_id.to_numpy(dtype=str))
                or not np.array_equal(evidence.station.to_numpy(), expected_keys.station.to_numpy())):
            raise ValueError("incomplete/out-of-order OOF station coverage")
    arrays = evidence[["y_true", "last_known", "pred_residual", "pred_selected"]].to_numpy(float)
    if not np.isfinite(arrays).all():
        raise ValueError("nonfinite OOF evidence; no finite-subset scoring")
    if set(evidence.fold.unique()) != {0, 1, 2} or not evidence.groupby("well_id").fold.nunique().eq(1).all():
        raise ValueError("OOF requires three grouped folds and one fold per well")
    expected = evidence.last_known.to_numpy(float) + evidence.pred_residual.to_numpy(float)
    fallback = evidence.fallback_used.to_numpy(bool)
    expected[fallback] = evidence.last_known.to_numpy(float)[fallback]
    if not np.array_equal(expected, evidence.pred_selected.to_numpy(float)):
        raise ValueError("selected prediction/residual parity mismatch")
    scored = evidence.assign(sse_candidate=(evidence.pred_selected - evidence.y_true) ** 2,
                             sse_last_known=(evidence.last_known - evidence.y_true) ** 2)
    if not np.isfinite(scored[["sse_candidate", "sse_last_known"]].to_numpy()).all():
        raise ValueError("nonfinite squared error")
    per_well = scored.groupby("well_id", sort=False).agg(fold=("fold", "first"), n=("station", "size"),
                    sse_candidate=("sse_candidate", "sum"), sse_last_known=("sse_last_known", "sum")).reset_index()
    for name in ("candidate", "last_known"):
        per_well[f"rmse_{name}_ft"] = np.sqrt(per_well[f"sse_{name}"] / per_well.n)
    metrics = {}
    for name in ("candidate", "last_known"):
        sse = float(scored[f"sse_{name}"].sum())
        metrics[name] = {"n": len(scored), "sse": sse, "rmse_ft": float(np.sqrt(sse / len(scored)))}
    per_fold = []
    for fold, rows in scored.groupby("fold", sort=True):
        per_fold.append({"fold": int(fold), "n": len(rows), **{f"sse_{name}": float(rows[f"sse_{name}"].sum()) for name in ("candidate", "last_known")}})
    report = {"mode": "experimental_development_oof", "n_wells": len(per_well), "n_stations": len(scored),
              "methods": metrics, "per_fold": per_fold, "complete_coverage_asserted": expected_keys is not None,
              "fallback_stations": int(fallback.sum()), "predictive_admitted": False,
              "decision": "DEVELOPMENT_OOF_IMPROVEMENT_REQUIRES_CONFIRMATION" if metrics["candidate"]["sse"] < metrics["last_known"]["sse"] else "REJECTED_DEVELOPMENT_OOF_ERROR"}
    return per_well, report


@dataclass
class BlockFitResult:
    predictor: BlockPredictor
    oof_evidence: pd.DataFrame
    per_well: pd.DataFrame
    report: dict[str, Any]
    fold_medians: tuple[np.ndarray, ...]


def fit_block_model(
    well_pairs: Mapping[str, tuple[pd.DataFrame, pd.DataFrame]], *,
    labels: Mapping[str, np.ndarray] | Callable[[str], np.ndarray] | None = None,
    role: str = "development",
) -> BlockFitResult:
    """Fit fixed three-fold OOF and a final development refit.

    Mapping order defines the original block order. Labels may be supplied by a
    lazy loader; it is invoked only after every label-blind feature is built.
    This in-memory API uses a caller-declared development cohort. The CLI adds
    mandatory manifest-based development/confirmation role enforcement.
    """
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold
    if role != "development" or len(well_pairs) < 3 or any(not isinstance(wid, str) or not wid for wid in well_pairs):
        raise ValueError("at least three explicitly identified development wells required")
    matrices, records = [], []
    groups, weights, last_values = [], [], []
    block_begin = matrix_bytes = 0
    for wi, (wid, (lateral, book)) in enumerate(well_pairs.items()):
        matrix, meta = build_block_features(lateral, book)
        matrix_bytes += matrix.nbytes
        if matrix_bytes * 2 > MAX_FEATURE_MATRIX_BYTES:
            raise MemoryError("float64-equivalent feature matrix exceeds 2 GB; fit not launched")
        matrices.append(matrix)
        groups.append(np.full(len(matrix), wi, np.int32))
        weights.append(meta["row_count"])
        last_values.append(meta["last_known"])
        records.append((wid, lateral, meta, block_begin, block_begin + len(matrix)))
        block_begin += len(matrix)
    matrix = np.concatenate(matrices)
    groups, weights, last = np.concatenate(groups), np.concatenate(weights), np.concatenate(last_values)
    # Labels are accessed only after the full label-blind matrix exists.
    center_truth, station_truth = [], []
    for wid, lateral, meta, _, _ in records:
        values = (labels(wid) if callable(labels) else labels[wid]) if labels is not None else lateral["TVT"].to_numpy(float)
        truth = normalize_curve_nulls(np.asarray(values, float))
        if truth.shape != (meta["n_rows"],) or not np.isfinite(truth[meta["n_known"]:]).all():
            raise ValueError(f"{wid}: incomplete development target coverage")
        center_truth.append(truth[meta["center_station"]])
        station_truth.append(truth[meta["n_known"]:])
    target = np.concatenate(center_truth) - last
    oof_residual = np.full(len(matrix), np.nan)
    oof_fold = np.full(len(matrix), -1, np.int8)
    oof_fallback = np.zeros(len(matrix), bool)
    medians_by_fold = []

    def train_model(train_rows, medians):
        dataset = lgb.Dataset(apply_medians(matrix[train_rows], medians), label=target[train_rows],
                              weight=weights[train_rows], feature_name=list(FEATURE_NAMES), free_raw_data=True)
        return lgb.train(dict(LGBM_PARAMETERS), dataset, num_boost_round=BOOSTING_ROUNDS)

    for fold, (train, validation) in enumerate(GroupKFold(n_splits=3).split(matrix, groups=groups)):
        if not set(groups[train]).isdisjoint(groups[validation]):
            raise RuntimeError("grouped fold overlap")
        medians = fit_train_medians(matrix[train])
        medians_by_fold.append(medians)
        model = train_model(train, medians)
        residual = np.asarray(model.predict(apply_medians(matrix[validation], medians), num_iteration=BOOSTING_ROUNDS, num_threads=4), float)
        if residual.shape != (len(validation),):
            raise ValueError("validation/block prediction shape mismatch")
        fallback = ~np.isfinite(residual)
        oof_residual[validation] = np.where(fallback, 0., residual)
        oof_fallback[validation] = fallback
        oof_fold[validation] = fold
    if (oof_fold < 0).any() or not np.isfinite(oof_residual).all():
        raise RuntimeError("incomplete OOF block coverage")
    frames, key_frames = [], []
    for wi, (wid, lateral, meta, begin, end) in enumerate(records):
        counts = meta["row_count"]
        station = np.arange(meta["n_known"], meta["n_rows"], dtype=np.int64)
        residual = block_to_station(oof_residual[begin:end], counts)
        last_toe = np.repeat(last[begin:end], counts)
        key_frames.append(pd.DataFrame({"well_id": wid, "station": station}))
        frames.append(pd.DataFrame({"well_id": wid, "station": station,
            "MD": lateral["MD"].to_numpy(float)[station], "block_index": np.repeat(np.arange(begin, end), counts),
            "fold": np.repeat(oof_fold[begin:end], counts), "y_true": station_truth[wi],
            "last_known": last_toe, "pred_residual": residual, "pred_selected": last_toe + residual,
            "fallback_used": np.repeat(oof_fallback[begin:end], counts)}))
    evidence = pd.concat(frames, ignore_index=True)
    per_well, report = evaluate_block_oof(evidence, expected_keys=pd.concat(key_frames, ignore_index=True))
    report.update({"n_blocks": len(matrix), "n_features": len(FEATURE_NAMES), "feature_matrix_bytes": matrix.nbytes,
                   "fallback_blocks": int(oof_fallback.sum()), "grouped_folds": 3,
                   "feature_schema_sha256": feature_schema_sha256(), "feature_source_sha256": _source_hashes(),
                   "scope": "Caller-supplied development cohort; no protected holdout is inferred by the in-memory API"})
    final_medians = fit_train_medians(matrix)
    final_model = train_model(np.arange(len(matrix)), final_medians)
    metadata = _metadata()
    metadata.update({"training_role": "development", "training_well_count": len(records),
                     "training_block_count": len(matrix), "training_toe_count": len(evidence),
                     "final_refit": "all selected development blocks, same fixed recipe",
                     "training_cohort_sha256": hashlib.sha256(json.dumps(list(well_pairs), separators=(",", ":")).encode()).hexdigest()})
    return BlockFitResult(BlockPredictor(final_model, final_medians, metadata), evidence, per_well, report, tuple(medians_by_fold))


def select_development_wells(manifest: dict, requested: list[str] | None = None) -> list[str]:
    """Validate roles before constructing/opening any selected source CSV path."""
    development = manifest.get("development_well_ids")
    confirmation = manifest.get("confirmation_well_ids")

    def valid(items):
        return (isinstance(items, list)
                and all(isinstance(wid, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", wid) for wid in items)
                and len({wid.casefold() for wid in items}) == len(items))

    if not valid(development) or not valid(confirmation) or not development:
        raise ValueError("manifest requires case-unambiguous explicit development/confirmation well lists")
    confirmation_ids = {wid.casefold() for wid in confirmation}
    if {wid.casefold() for wid in development} & confirmation_ids:
        raise ValueError("development/confirmation well lists must be disjoint ignoring case")
    selected = list(development) if requested is None else list(requested)
    if (not valid(selected) or not selected or not set(selected).issubset(development)
            or {wid.casefold() for wid in selected} & confirmation_ids):
        raise ValueError("selected wells must use unique exact manifest development IDs; confirmation is reserved ignoring case")
    if len(selected) < 3:
        raise ValueError("fixed three-fold training requires at least three selected development wells")
    return selected


def _save_evidence(path: Path, evidence: pd.DataFrame) -> None:
    arrays = {name: evidence[name].to_numpy() for name in evidence}
    arrays["well_id"] = evidence.well_id.to_numpy(dtype=str)
    np.savez_compressed(path, **arrays)


def _load_evidence(path: Path) -> pd.DataFrame:
    with np.load(path, allow_pickle=False) as values:
        return pd.DataFrame({name: values[name] for name in values.files})


def _fit_cli_worker(contract_path: Path) -> None:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    manifest_path = Path(contract["manifest_path"])
    if _sha(manifest_path) != contract["manifest_sha256"]:
        raise ValueError("cohort manifest changed after freeze")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = select_development_wells(manifest, contract["well_ids"])
    if contract["source_sha256"] != _controller_source_hashes():
        raise ValueError("public feature/training source changed after freeze")
    if contract["lightgbm_parameters"] != LGBM_PARAMETERS or contract["boosting_rounds"] != BOOSTING_ROUNDS or contract["feature_schema_sha256"] != feature_schema_sha256():
        raise ValueError("fixed training contract/schema mismatch")
    for path, expected in contract["input_sha256"].items():
        if _sha(Path(path)) != expected:
            raise ValueError("selected development source changed after freeze")
    pairs = {}
    data_root = Path(contract["data_root"])
    for wid in selected:
        lateral = pd.read_csv(data_root / "train" / f"{wid}__horizontal_well.csv", usecols=list(INPUT_COLUMNS))
        book = pd.read_csv(data_root / "train" / f"{wid}__typewell.csv", usecols=["TVT", "GR"])
        pairs[wid] = (lateral, book)

    def labels(wid):
        # Lazy: fit_block_model calls this after all label-blind features exist.
        return pd.read_csv(data_root / "train" / f"{wid}__horizontal_well.csv", usecols=["TVT"])["TVT"].to_numpy(float)

    result = fit_block_model(pairs, labels=labels)
    # Recheck the frozen boundary after all OOF/refit work, before claiming a
    # completed result. Byte hashing does not parse additional source labels.
    if (_sha(manifest_path) != contract["manifest_sha256"]
            or contract["source_sha256"] != _controller_source_hashes()
            or any(_sha(Path(path)) != expected for path, expected in contract["input_sha256"].items())):
        raise RuntimeError("frozen inputs/source changed during fitting")
    out = contract_path.parent
    path = out / "oof_predictions.npz"
    _save_evidence(path, result.oof_evidence)
    result.oof_evidence.to_csv(out / "oof_predictions.csv", index=False)
    # Reopen exact saved arrays: coverage and scalar metrics must survive IO.
    _, replay = evaluate_block_oof(_load_evidence(path), expected_keys=result.oof_evidence[["well_id", "station"]])
    if replay["methods"] != result.report["methods"]:
        raise RuntimeError("saved OOF metric replay mismatch")
    result.per_well.to_csv(out / "per_well.csv", index=False)
    result.report.update({"manifest_sha256": contract["manifest_sha256"], "contract_sha256": _sha(contract_path),
        "confirmation_reserved_by_manifest": True, "confirmation_well_count": len(manifest["confirmation_well_ids"]),
        "confirmation_labels_parsed": False, "oof_predictions_sha256": _sha(path),
        "exact_saved_vector_replay_passed": True,
        "replay_scope": "Same implementation reopens saved arrays; not an independent method or human review",
        "scope": "Manifest-selected development OOF; raw selected sources byte-hashed; scoring labels parsed through separate development label loader"})
    result.predictor.metadata.update({"manifest_sha256": contract["manifest_sha256"],
        "confirmation_reserved_by_manifest": True, "confirmation_well_count": len(manifest["confirmation_well_ids"])})
    for fold, medians in enumerate(result.fold_medians):
        np.save(out / f"fold{fold}_medians.npy", medians, allow_pickle=False)
    save_block_predictor(result.predictor, out / "model")
    _json(out / "report.json", result.report)
    print(json.dumps({"methods": result.report["methods"], "n_stations": len(result.oof_evidence),
                      "fallback_stations": result.report["fallback_stations"], "predictive_admitted": False}), flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit", help="Experimental fixed grouped OOF and development refit")
    fit.add_argument("--data-root", type=Path, required=True)
    fit.add_argument("--manifest", type=Path, required=True)
    fit.add_argument("--well-id", action="append", help="Optional explicit development subset; repeat per well")
    fit.add_argument("--out", type=Path, required=True, help="Fresh output directory")
    predict = sub.add_parser("predict", help="Label-blind inference with an experimental text-model bundle")
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--lateral", type=Path, required=True)
    predict.add_argument("--typewell", type=Path, required=True)
    predict.add_argument("--well-id", required=True)
    predict.add_argument("--out", type=Path, required=True)
    worker = sub.add_parser("_fit", help=argparse.SUPPRESS)
    worker.add_argument("contract", type=Path)
    args = parser.parse_args(argv)
    if args.command == "_fit":
        _fit_cli_worker(args.contract)
        return 0
    if args.command == "predict":
        if args.out.exists():
            raise FileExistsError("prediction output already exists")
        model = load_block_predictor(args.model)
        lateral = pd.read_csv(args.lateral, usecols=list(INPUT_COLUMNS))
        book = pd.read_csv(args.typewell, usecols=["TVT", "GR"])
        result = model.predict_well(lateral, book, well_id=args.well_id)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.out, index=False)
        print(json.dumps({"n_rows": len(result), "n_toe": int(result.is_predict.sum()),
                          "fallback_rows": int(result.fallback_used.sum()), "predictive_admitted": False}))
        return 0
    # Role selection is deliberately before source CSV hashing or parsing.
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = select_development_wells(manifest, args.well_id)
    data_root, out = args.data_root.resolve(), args.out.resolve()
    if out.exists():
        raise FileExistsError("fit output must be fresh")
    sources = [data_root / "train" / f"{wid}__{suffix}.csv" for wid in selected for suffix in ("horizontal_well", "typewell")]
    contract = {"schema": "rwt.experimental-block-fit.v1", "model_kind": MODEL_KIND,
        "manifest_path": str(manifest_path), "manifest_sha256": _sha(manifest_path),
        "well_ids": selected, "data_root": str(data_root), "role": "development",
        "source_sha256": _controller_source_hashes(), "feature_schema_sha256": feature_schema_sha256(),
        "input_sha256": {str(path): _sha(path) for path in sources},
        "lightgbm_parameters": dict(LGBM_PARAMETERS), "boosting_rounds": BOOSTING_ROUNDS,
        "confirmation_labels_parsed": False, "predictive_admitted": False}
    out.mkdir(parents=True, exist_ok=False)
    contract_path = out / "fit_contract.json"
    _json(contract_path, contract)
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "4"
    from rwt.execution_graph import run_bounded
    receipt = run_bounded([sys.executable, str(Path(__file__).resolve()), "_fit", str(contract_path)],
        cwd=Path.cwd(), trace_path=out / "fit.trace.jsonl", node_id="measure", timeout_s=600)
    _json(out / "execution_receipt.json", receipt)
    return 0 if receipt["status"] == "SUCCEEDED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
