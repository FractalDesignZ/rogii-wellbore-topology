#!/usr/bin/env python3
"""Recompute synthetic placement metrics from frozen station evaluation arrays."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rwt.pad_measure import evaluate_station_metrics  # noqa: E402
from rwt.provenance import public_code_digest, write_sealed_json  # noqa: E402


def audit(directory: Path) -> dict:
    directory = Path(directory)
    contract = json.loads((directory / "contract.json").read_text(encoding="utf-8"))["payload"]
    if public_code_digest()["tree_sha256"] != contract["public_code_tree_sha256"]:
        raise ValueError("public code changed after the measurement contract was recorded")
    closure = json.loads((directory / "closure.json").read_text(encoding="utf-8"))["payload"]
    required = {"contract.json", "station_evaluation.npz", "per_well.csv", "typewell_book.csv", "summary.json"}
    if set(closure) != required:
        raise ValueError("measurement closure has an unexpected artifact set")
    for name, expected in closure.items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"artifact digest mismatch: {name}")
    worker = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if worker["config"] != contract["config"]:
        raise ValueError("summary configuration differs from the recorded contract")
    with np.load(directory / "station_evaluation.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    table, summary = evaluate_station_metrics(arrays, worker["roles"], book_mae_limit=contract["config"]["book_mae_limit"])
    for key, value in summary.items():
        if worker.get(key) != value:
            raise ValueError(f"metric/decision mismatch: {key}")
    pd.testing.assert_frame_equal(table, pd.read_csv(directory / "per_well.csv"), check_dtype=False, rtol=1e-12, atol=1e-12)
    return {"schema": "rwt.pad-audit.v1", "audit_passed": True, "decision": summary["status"],
            "predictive_admitted": False, "closure_sha256": hashlib.sha256((directory / "closure.json").read_bytes()).hexdigest(),
            "scope": "separate-process recomputation of frozen station metrics and gates; not an independent GR model or field evaluation"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run)
    target = args.run / "audit.json"
    if target.exists():
        raise FileExistsError(target)
    write_sealed_json(target, result, kind="pad_metric_audit")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
