#!/usr/bin/env python3
"""Audit source identity, cohort membership and the retained selected OOF metric."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rwt.execution_graph import verify_source_manifest  # noqa: E402
from rwt.provenance import write_sealed_json  # noqa: E402


def _station_keys(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    if any(name not in frame for name in ("well_id", "station")):
        raise ValueError(f"{label} needs explicit well_id/station keys")
    if frame["well_id"].isna().any() or frame["well_id"].astype(str).str.strip().eq("").any():
        raise ValueError(f"{label} contains missing well IDs")
    station = frame["station"].to_numpy(dtype=float)
    if not np.isfinite(station).all() or np.any(station < 0) or not np.equal(station, np.floor(station)).all():
        raise ValueError(f"{label} station IDs must be finite nonnegative integers")
    frame = frame.copy()
    frame["well_id"] = frame["well_id"].astype(str)
    frame["station"] = station.astype(np.int64)
    if frame.duplicated(["well_id", "station"]).any():
        raise ValueError(f"{label} contains duplicate well/station keys")
    return frame


def _configured_folds(contract: dict) -> int:
    value = contract.get("folds")
    if value is None:
        command = contract.get("command", [])
        if not isinstance(command, list) or command.count("--folds") != 1:
            raise ValueError("baseline contract must freeze the requested fold count")
        position = command.index("--folds")
        if position + 1 >= len(command):
            raise ValueError("baseline contract is missing its fold-count argument")
        try:
            value = int(command[position + 1])
        except (TypeError, ValueError) as exc:
            raise ValueError("baseline fold count is invalid") from exc
    if isinstance(value, bool) or not isinstance(value, int) or value < 2:
        raise ValueError("baseline fold count must be an integer >= 2")
    return value


def _require_count(mapping: dict, key: str, expected: int) -> None:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise ValueError(f"reported row/group count mismatch: {key}")


def audit(run: Path) -> dict:
    contract = json.loads((run / "contract.json").read_text())["payload"]
    source = json.loads((run / "ast.json").read_text())
    if source["manifest_sha256"] != contract["source_manifest_sha256"] or not verify_source_manifest(ROOT, source)["passed"]:
        raise ValueError("source changed after baseline planning")
    cohort_path = Path(contract["cohort_path"])
    if hashlib.sha256(cohort_path.read_bytes()).hexdigest() != contract["cohort_sha256"]:
        raise ValueError("cohort manifest changed")
    cohort = json.loads(cohort_path.read_text())
    report = json.loads((run / "model/cv_report.json").read_text())
    oof = _station_keys(pd.read_csv(run / "model/oof_predictions.csv", dtype={"well_id": str}), "OOF evidence")
    ids = set(oof["well_id"].astype(str))
    pilot = set(contract["pilot_well_ids"])
    if (pilot != set(cohort["pilot_well_ids"]) or not pilot <= set(cohort["development_well_ids"])
            or ids != pilot or ids & set(cohort["confirmation_well_ids"])):
        raise ValueError("OOF rows differ from the development pilot")
    truth = oof["y_true"].to_numpy(dtype=float)
    selected = oof["pred_selected"].to_numpy(dtype=float)
    if not len(oof) or not np.isfinite(truth).all() or not np.isfinite(selected).all():
        raise ValueError("selected OOF coverage is incomplete")
    stations = _station_keys(pd.read_csv(run / "model/stations.csv", dtype={"well_id": str}), "input stations")
    if set(stations["well_id"]) != pilot:
        raise ValueError("input stations differ from the development pilot")
    bases = report.get("residual_bases")
    if not isinstance(bases, list) or not bases or any(not isinstance(name, str) for name in bases) or len(set(bases)) != len(bases):
        raise ValueError("report must retain unique residual base columns")
    if any(name not in stations for name in ["TVT", *bases]):
        raise ValueError("input stations are missing truth or a retained residual base")
    predict = np.ones(len(stations), dtype=bool)
    if "is_predict" in stations:
        values = stations["is_predict"].to_numpy()
        if values.dtype.kind not in "biuf" or not np.isfinite(values).all() or not np.isin(values, [0, 1]).all():
            raise ValueError("input is_predict must contain boolean or 0/1 values")
        predict = values.astype(bool)
    labeled = stations.loc[predict & stations["TVT"].notna()].copy()
    finite = np.isfinite(labeled["TVT"].to_numpy(dtype=float))
    for base in bases:
        finite &= np.isfinite(labeled[base].to_numpy(dtype=float))
    eligible = labeled.loc[finite].copy()
    n_eligible = len(eligible)
    n_groups = eligible["well_id"].nunique()
    metadata = report.get("oof_evidence", {})
    if metadata.get("cv_strategy") != "GroupKFold_by_well" or n_groups < 2:
        raise ValueError("master baseline requires grouped OOF validation")
    _require_count(report, "n_stations", n_eligible)
    _require_count(report, "n_wells", n_groups)
    for key in ("eligible_rows", "validation_rows", "shared_rows", "selected_finite_rows"):
        _require_count(metadata, key, n_eligible)
    if (len(oof) != n_eligible or metadata.get("selected_shared_coverage_complete") is not True
            or metadata.get("selected_shared_coverage_fraction") != 1.0):
        raise ValueError("retained OOF row coverage differs from the eligible input rows")
    requested_folds = _configured_folds(contract)
    actual_folds = min(requested_folds, n_groups)
    if len(report.get("fold_rmse", [])) != actual_folds:
        raise ValueError("reported fold count differs from the contract")
    fold = oof["fold_id"].to_numpy(dtype=float) if "fold_id" in oof else np.full(len(oof), np.nan)
    if not np.isfinite(fold).all() or np.any(fold < 0) or not np.equal(fold, np.floor(fold)).all():
        raise ValueError("OOF fold IDs must be nonnegative integers")
    if not oof.groupby("well_id")["fold_id"].nunique().eq(1).all():
        raise ValueError("a validation well is split across folds")
    expected_folds = np.full(n_eligible, -1, dtype=int)
    groups = eligible["well_id"].to_numpy(dtype=str)
    for fold_id, (_train, validation) in enumerate(GroupKFold(n_splits=actual_folds).split(np.zeros((n_eligible, 1)), groups=groups)):
        expected_folds[validation] = fold_id
    expected = eligible[["well_id", "station", "TVT"]].copy()
    expected["expected_fold_id"] = expected_folds
    aligned = oof.merge(expected, on=["well_id", "station"], how="outer", validate="one_to_one", indicator=True)
    if not aligned["_merge"].eq("both").all():
        raise ValueError("OOF station keys differ from the exact eligible input scope")
    if not np.array_equal(aligned["y_true"].to_numpy(dtype=float), aligned["TVT"].to_numpy(dtype=float)):
        raise ValueError("OOF truth labels differ from input stations")
    if not np.array_equal(aligned["fold_id"].to_numpy(dtype=float), aligned["expected_fold_id"].to_numpy(dtype=float)):
        raise ValueError("OOF fold assignments do not replay the frozen GroupKFold configuration")
    measured = float(np.sqrt(np.mean((truth - selected) ** 2)))
    if not np.isclose(measured, report["oof_rmse_selected"], rtol=1e-10, atol=1e-10):
        raise ValueError("selected OOF score does not match retained predictions")
    return {"schema": "rwt.master-baseline-audit.v2", "passed": True, "n_rows": len(oof), "n_wells": len(ids),
            "selected_oof_rmse": measured, "confirmation_unmeasured": True, "predictive_admitted": False,
            "input_station_rows": len(stations), "input_labeled_prediction_rows": len(labeled),
            "eligible_rows": n_eligible, "excluded_nonfinite_truth_or_base_rows": int((~finite).sum()),
            "excluded_nonprediction_or_unlabeled_rows": len(stations) - len(labeled),
            "exact_station_keys_and_truth_match": True, "fold_assignment_replayed": True,
            "requested_folds": requested_folds, "actual_folds": actual_folds,
            "stations_sha256": hashlib.sha256((run / "model/stations.csv").read_bytes()).hexdigest(),
            "oof_sha256": hashlib.sha256((run / "model/oof_predictions.csv").read_bytes()).hexdigest(),
            "scope": "development diagnostic; model/gate selection used these OOF labels; no independent confirmation or official rank"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run)
    target = args.run / "metric_audit.json"
    if target.exists():
        raise FileExistsError(target)
    write_sealed_json(target, result, kind="master_baseline_metric_audit")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
