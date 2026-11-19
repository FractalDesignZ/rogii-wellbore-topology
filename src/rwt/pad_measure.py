"""Exact metric and gate audit for retained synthetic wellbore station arrays.

This evaluator consumes declared predictions and offline truth. It constructs no
predictions and imports no private dynamics implementation.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


STATION_FIELDS = (
    "truth", "last", "neighbor", "is_predict",
    "gr_error_true", "gr_error_last", "gr_error_neighbor",
)
PER_WELL_COLUMNS = (
    "well_id", "pad_role", "n_toe", "rmse_last_known", "rmse_neighbor_idw",
    "rmse_oracle_true", "gr_mae_at_true_tvt", "gr_mae_at_last_known",
    "gr_mae_at_neighbor", "gr_n", "neighbor_beats_last",
    "gr_true_better_than_last", "n_toe_expected", "gr_n_expected",
    "placement_coverage_complete", "gr_coverage_complete",
)


def _finite_mean(frame: pd.DataFrame, column: str) -> float | None:
    """An unscored well makes the aggregate unavailable, never silently absent."""
    if frame.empty:
        return None
    values = frame[column].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        value = float(np.mean(values))
    return value if np.isfinite(value) else None


def evaluate_station_metrics(
    station_arrays: dict[str, np.ndarray],
    roles: dict[str, str],
    *,
    book_mae_limit: float | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Recompute per-well metrics and the joint synthetic construct gate.

    Placement comparisons share finite truth/last/neighbor prediction support.
    GR comparisons share finite true/last/neighbor error support. Every well must
    retain all stations where its reference truth/error is available: candidate
    NaNs cannot remove difficult rows from the admitted score. Empty support or
    an unscored well rejects the construct, while preserving that well's row.

    ``book_mae_limit`` is an optional explicitly declared strict upper bound.
    No noise threshold is inferred. A synthetic pass never admits field merit.
    """
    if not isinstance(roles, dict) or not roles:
        raise ValueError("roles must declare at least one well")
    for well_id, role in roles.items():
        if not isinstance(well_id, str) or not well_id.strip() or "/" in well_id:
            raise ValueError("well IDs must be nonempty strings without '/' separators")
        if not isinstance(role, str) or role not in ("train", "test"):
            raise ValueError(f"invalid role for {well_id!r}: {role!r}")
    if not isinstance(station_arrays, dict) or any(not isinstance(k, str) for k in station_arrays):
        raise ValueError("station_arrays must be a string-keyed array dictionary")
    expected_keys = {f"{well_id}/{field}" for well_id in roles for field in STATION_FIELDS}
    if set(station_arrays) != expected_keys:
        missing = sorted(expected_keys - set(station_arrays))
        extra = sorted(set(station_arrays) - expected_keys)
        raise ValueError(f"station array keys do not match roles: missing={missing}, extra={extra}")
    if book_mae_limit is not None:
        if isinstance(book_mae_limit, bool) or not isinstance(book_mae_limit, (int, float, np.integer, np.floating)):
            raise ValueError("book_mae_limit must be a finite nonnegative number or None")
        book_mae_limit = float(book_mae_limit)
        if not np.isfinite(book_mae_limit) or book_mae_limit < 0:
            raise ValueError("book_mae_limit must be a finite nonnegative number or None")

    rows = []
    for well_id, role in sorted(roles.items()):
        values = {}
        lengths = set()
        for field in STATION_FIELDS:
            array = np.asarray(station_arrays[f"{well_id}/{field}"])
            if array.ndim != 1 or array.dtype.kind not in "biuf":
                raise ValueError(f"{well_id}/{field} must be a one-dimensional real numeric array")
            lengths.add(len(array))
            if field == "is_predict":
                if not np.isfinite(array).all() or not np.isin(array, [False, True]).all():
                    raise ValueError(f"{well_id}/is_predict must contain only boolean or 0/1 values")
                values[field] = array.astype(bool)
            else:
                values[field] = array.astype(float)
        if len(lengths) != 1:
            raise ValueError(f"station array lengths differ for {well_id!r}")

        predict = values["is_predict"]
        truth, last, neighbor = values["truth"], values["last"], values["neighbor"]
        expected_toe = predict & np.isfinite(truth)
        toe = expected_toe & np.isfinite(last) & np.isfinite(neighbor)
        n_toe, n_toe_expected = int(toe.sum()), int(expected_toe.sum())
        expected_gr = predict & np.isfinite(values["gr_error_true"])
        gr_mask = expected_gr & np.isfinite(values["gr_error_last"]) & np.isfinite(values["gr_error_neighbor"])
        gr_n, gr_n_expected = int(gr_mask.sum()), int(expected_gr.sum())
        with np.errstate(over="ignore", invalid="ignore"):
            r_last = float(np.sqrt(np.mean((truth[toe] - last[toe]) ** 2))) if n_toe else float("nan")
            r_neighbor = float(np.sqrt(np.mean((truth[toe] - neighbor[toe]) ** 2))) if n_toe else float("nan")
            gr_metrics = {
                field: float(np.mean(np.abs(values[field][gr_mask]))) if gr_n else float("nan")
                for field in ("gr_error_true", "gr_error_last", "gr_error_neighbor")
            }
        rows.append({
            "well_id": well_id, "pad_role": role,
            "n_toe": n_toe, "rmse_last_known": r_last, "rmse_neighbor_idw": r_neighbor,
            "rmse_oracle_true": 0.0 if n_toe else float("nan"),
            "gr_mae_at_true_tvt": gr_metrics["gr_error_true"],
            "gr_mae_at_last_known": gr_metrics["gr_error_last"],
            "gr_mae_at_neighbor": gr_metrics["gr_error_neighbor"], "gr_n": gr_n,
            "neighbor_beats_last": bool(np.isfinite(r_neighbor) and np.isfinite(r_last) and r_neighbor < r_last),
            "gr_true_better_than_last": bool(
                np.isfinite(gr_metrics["gr_error_true"]) and np.isfinite(gr_metrics["gr_error_last"])
                and gr_metrics["gr_error_true"] < gr_metrics["gr_error_last"]
            ),
            "n_toe_expected": n_toe_expected, "gr_n_expected": gr_n_expected,
            "placement_coverage_complete": bool(n_toe > 0 and n_toe == n_toe_expected),
            "gr_coverage_complete": bool(gr_n > 0 and gr_n == gr_n_expected),
        })

    table = pd.DataFrame(rows, columns=PER_WELL_COLUMNS)
    summary: dict[str, Any] = {
        "n_wells": len(table), "n_train": int(table.pad_role.eq("train").sum()),
        "n_test": int(table.pad_role.eq("test").sum()), "book_mae_limit": book_mae_limit,
    }
    metric_names = {
        "rmse_last_known": "mean_rmse_last_known",
        "rmse_neighbor_idw": "mean_rmse_neighbor_idw",
        "rmse_oracle_true": "mean_rmse_oracle_true",
        "gr_mae_at_true_tvt": "mean_gr_mae_at_true_tvt",
        "gr_mae_at_last_known": "mean_gr_mae_at_last_known",
        "gr_mae_at_neighbor": "mean_gr_mae_at_neighbor",
    }
    for role in (None, "train", "test"):
        subset = table if role is None else table[table.pad_role.eq(role)]
        suffix = "" if role is None else f"_{role}"
        for column, name in metric_names.items():
            summary[name + suffix] = _finite_mean(subset, column)
    finite_metrics = np.isfinite(table[list(metric_names)].to_numpy(dtype=float)).all(axis=1)
    complete = finite_metrics & table.placement_coverage_complete.to_numpy(bool) & table.gr_coverage_complete.to_numpy(bool)
    true_gr, last_gr = summary["mean_gr_mae_at_true_tvt"], summary["mean_gr_mae_at_last_known"]
    test_last, test_neighbor = summary["mean_rmse_last_known_test"], summary["mean_rmse_neighbor_idw_test"]
    book_ok = bool(
        true_gr is not None and last_gr is not None and true_gr < last_gr
        and (book_mae_limit is None or true_gr < book_mae_limit)
    )
    pad_ok = bool(test_last is not None and test_neighbor is not None and test_neighbor < test_last)
    ok = bool(book_ok and pad_ok and summary["n_test"] > 0 and complete.all())
    summary.update({
        "n_test_wells_neighbor_beats_last": int(table.loc[table.pad_role.eq("test"), "neighbor_beats_last"].sum()),
        "complete_per_well_metrics": bool(complete.all()),
        "unscored_or_incomplete_wells": table.loc[~complete, "well_id"].tolist(),
        "falsifier_book_ok": book_ok, "falsifier_pad_transfer_ok": pad_ok,
        "ok": ok, "status": "synthetic_construct_passed" if ok else "rejected",
        "predictive_admitted": False,
    })
    return table, summary
