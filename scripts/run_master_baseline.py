#!/usr/bin/env python3
"""Run a development-only baseline through source, dependency and execution checks."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rwt.execution_graph import (  # noqa: E402
    build_source_manifest, default_stage_graph, dependency_ready, run_bounded,
    validate_stage_graph, verify_execution_trace, verify_source_manifest,
)
from rwt.provenance import write_sealed_json  # noqa: E402


def finalize_result(source: dict, graph: list[dict], statuses: dict[str, str], trace_path: Path) -> dict:
    """Final evidence admission fails closed on changed sources or an invalid trace."""
    statuses = dict(statuses)
    trace = verify_execution_trace(trace_path)
    source_check = verify_source_manifest(ROOT, source)
    ready = dependency_ready("admit", graph, statuses)
    # This finite controller executes exactly two processes, each with a start
    # and finish event. An empty/missing trace is append-valid, not complete.
    trace_complete = trace.get("passed") is True and trace.get("events") == 4
    checks_passed = trace_complete and source_check.get("passed") is True
    statuses["admit"] = "SUCCEEDED" if ready and checks_passed else "BLOCKED"
    return {"statuses": statuses, "trace": trace, "trace_complete": trace_complete, "final_source_check": source_check,
            "baseline_measured_and_audited": statuses["admit"] == "SUCCEEDED" and checks_passed,
            "confirmation_unmeasured": True, "predictive_admitted": False, "official_rank": None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=600)
    args = parser.parse_args()
    if args.workers < 1 or args.folds < 2:
        parser.error("workers >= 1 and folds >= 2 are required")
    cohort = json.loads(args.cohort.read_text(encoding="utf-8"))
    pilot = set(cohort["pilot_well_ids"])
    if not pilot or not pilot <= set(cohort["development_well_ids"]) or pilot & set(cohort["confirmation_well_ids"]):
        raise ValueError("pilot must be a nonempty development-only well set")
    data = Path(cohort["trainer_data_argument"])
    expected_files = {row["path"] for wid in pilot for row in cohort["files"][wid]}
    actual_files = {path.relative_to(data).as_posix() for path in data.rglob("*.csv")}
    if expected_files != actual_files:
        raise ValueError("pilot file set differs from the recorded cohort")
    for wid in pilot:
        for row in cohort["files"][wid]:
            if hashlib.sha256((data / row["path"]).read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError("pilot file digest mismatch: " + row["path"])
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    graph = default_stage_graph()
    validate_stage_graph(graph)
    source = build_source_manifest(ROOT, ["scripts/run_master_baseline.py", "scripts/audit_master_baseline.py", "src/rwt/train.py"])
    if not verify_source_manifest(ROOT, source)["passed"]:
        raise ValueError("source manifest failed before launch")
    (out / "ast.json").write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
    (out / "graph.json").write_text(json.dumps(graph, indent=2) + "\n", encoding="utf-8")
    environment = {"PYTHONPATH": str(ROOT / "src"), "RWT_KERNEL": "public", "RWT_BLAS_THREADS": "1", "RWT_LGBM_THREADS": str(args.workers)}
    os.environ.update(environment)
    command = [sys.executable, "-m", "rwt.train", "--data", str(data.resolve()), "--out", str(out / "model"),
               "--workers", str(args.workers), "--folds", str(args.folds), "--prior-col", "prior_stack"]
    config = {"schema": "rwt.master-baseline.v1", "cohort_path": str(args.cohort.resolve()),
              "cohort_sha256": hashlib.sha256(args.cohort.read_bytes()).hexdigest(),
              "pilot_well_ids": sorted(pilot), "command": command, "environment": environment,
              "python": sys.version, "packages": {name: version(name) for name in ("numpy", "scipy", "pandas", "scikit-learn", "lightgbm")},
              "source_manifest_sha256": source["manifest_sha256"], "timeout_seconds": args.timeout_seconds,
              "folds": args.folds,
              "criteria": {"source_unchanged": True, "valid_execution_trace": True,
                           "exact_selected_oof_metric": True, "exact_grouped_row_coverage": True,
                           "fold_assignments_replayed": True, "development_only": True},
              "confirmation_unmeasured": True, "predictive_admitted": False}
    write_sealed_json(out / "contract.json", config, kind="master_baseline_contract")
    statuses = {row["id"]: "PENDING" for row in graph}
    statuses["plan"] = "SUCCEEDED"
    if not dependency_ready("measure", graph, statuses):
        raise ValueError("measurement dependencies are unresolved")
    measure = run_bounded(command, cwd=ROOT, trace_path=out / "trace.jsonl", node_id="measure",
                          timeout_s=args.timeout_seconds, dependencies=("plan",), statuses=statuses)
    statuses["measure"] = measure["status"]
    write_sealed_json(out / "measure_receipt.json", measure, kind="master_measurement")
    if statuses["measure"] == "SUCCEEDED" and dependency_ready("audit", graph, statuses):
        audited = run_bounded([sys.executable, str(ROOT / "scripts/audit_master_baseline.py"), "--run", str(out)],
                              cwd=ROOT, trace_path=out / "trace.jsonl", node_id="audit", timeout_s=60,
                              dependencies=("measure",), statuses=statuses)
        statuses["audit"] = audited["status"]
        write_sealed_json(out / "audit_receipt.json", audited, kind="master_audit_execution")
    result = finalize_result(source, graph, statuses, out / "trace.jsonl")
    write_sealed_json(out / "result.json", result, kind="master_baseline_result")
    print(json.dumps(result, indent=2))
    return 0 if result["baseline_measured_and_audited"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
