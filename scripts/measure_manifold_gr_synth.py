#!/usr/bin/env python3
"""WP8 synthetic falsifier with declared donors, fixed criteria and metric audit."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rwt.gr_match import gr_mismatch_at_tvt  # noqa: E402
from rwt.load import apply_curve_nulls_frame  # noqa: E402
from rwt.pad_measure import evaluate_station_metrics  # noqa: E402
from rwt.pad_synth import last_known_series, make_pad_laterals, neighbor_idw_tvt  # noqa: E402
from rwt.provenance import public_code_digest, write_sealed_json  # noqa: E402


def record_step(out: Path, stage: str, status: str, detail: dict) -> None:
    path = out / "steps_log.jsonl"
    previous = json.loads(path.read_text().splitlines()[-1])["entry_hash"] if path.exists() else "0" * 64
    entry = {"stage": stage, "status": status, "detail": detail, "prev_hash": previous}
    raw = json.dumps(entry, sort_keys=True, separators=(",", ":"), allow_nan=False)
    entry["entry_hash"] = hashlib.sha256((previous + raw).encode()).hexdigest()
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry, sort_keys=True, allow_nan=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-wells", type=int, default=6)
    parser.add_argument("--n-stations", type=int, default=180)
    parser.add_argument("--n-train", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gr-noise", type=float, default=3.0)
    parser.add_argument("--donor-policy", choices=("observed_only", "full_tvt"), default="observed_only")
    parser.add_argument("--book-mae-limit", type=float, help="optional declared GR MAE ceiling; no noise-independent default")
    parser.add_argument("--out", type=Path, help="fresh directory, default timestamped artifacts/manifold_gr_synth_...")
    args = parser.parse_args(argv)
    if args.n_wells < 2 or args.n_stations < 40:
        parser.error("need at least 2 wells and 40 stations")
    if args.n_train is not None and not 0 <= args.n_train < args.n_wells:
        parser.error("n-train must be between 0 and n-wells - 1")
    if not np.isfinite(args.gr_noise) or args.gr_noise < 0:
        parser.error("gr-noise must be finite and nonnegative")
    if args.book_mae_limit is not None and (not np.isfinite(args.book_mae_limit) or args.book_mae_limit <= 0):
        parser.error("book-mae-limit must be finite and positive")
    out = args.out or ROOT / "artifacts" / ("manifold_gr_synth_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    out.mkdir(parents=True, exist_ok=False)
    config = {name: getattr(args, name) for name in ("n_wells", "n_stations", "n_train", "seed", "gr_noise", "donor_policy", "book_mae_limit")}
    contract = {
        "schema": "rwt.pad-iteration.v1", "mode": "MEASURE", "synthetic": True,
        "config": config, "public_code_tree_sha256": public_code_digest()["tree_sha256"],
        "criteria": {"required": ["complete_common_support", "book_true_beats_last", "test_pad_beats_last"],
                     "book_mae_limit": args.book_mae_limit, "predictive_admitted": False},
        "scope": "synthetic construct checks; no field-performance admission or native engine phase changes",
    }
    write_sealed_json(out / "contract.json", contract, kind="pad_iteration_contract")
    record_step(out, "plan", "checked", {"contract_sha256": hashlib.sha256((out / "contract.json").read_bytes()).hexdigest()})
    pad = make_pad_laterals(n_wells=args.n_wells, n_stations=args.n_stations, n_train=args.n_train, seed=args.seed, gr_noise=args.gr_noise)
    wells = [apply_curve_nulls_frame(well) for well in pad["wells"]]
    donors = [well.drop(columns=["TVT_true"]) for well in wells if str(well["pad_role"].iloc[0]) == "train"]
    roles: dict[str, str] = {}
    arrays: dict[str, np.ndarray] = {}
    typewell = pad["typewell"]
    for well in wells:
        wid = str(well["well_id"].iloc[0])
        roles[wid] = str(well["pad_role"].iloc[0])
        # Oracle truth never enters either placement predictor.
        predictor_input = well.drop(columns=["TVT_true"])
        last = last_known_series(predictor_input)
        neighbor = neighbor_idw_tvt(predictor_input, donors, max_dist=900.0, k=4, label_policy=args.donor_policy)
        neighbor = np.where(np.isfinite(neighbor), neighbor, last)
        pred = well["is_predict"].to_numpy(dtype=bool)
        truth = well["TVT_true"].to_numpy(dtype=float)
        known = np.where(~pred, well["TVT_input"].to_numpy(dtype=float), np.nan)
        for name, value in (("truth", truth), ("last", last), ("neighbor", neighbor), ("is_predict", pred)):
            arrays[f"{wid}/{name}"] = value
        for name, prediction in (("true", truth), ("last", last), ("neighbor", neighbor)):
            arrays[f"{wid}/gr_error_{name}"] = gr_mismatch_at_tvt(
                well["GR"].to_numpy(dtype=float), typewell["GR"].to_numpy(dtype=float),
                typewell["TVT"].to_numpy(dtype=float), prediction, known_tvt=known,
            )
    table, summary = evaluate_station_metrics(arrays, roles, book_mae_limit=args.book_mae_limit)
    summary.update({"schema": "rwt.pad-measure.v1", "synthetic": True, "config": config, "roles": roles,
                    "field_params": pad["field_params"], "n_wells": len(roles),
                    "n_train": len(pad["train_ids"]), "n_test": len(pad["test_ids"]),
                    "seed": args.seed, "n_stations": args.n_stations,
                    "law": "RESTRICT; truth evaluator-only; no competition submit; every required falsifier must pass"})
    np.savez_compressed(out / "station_evaluation.npz", **arrays)
    table.to_csv(out / "per_well.csv", index=False)
    typewell.to_csv(out / "typewell_book.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    closure = {name: hashlib.sha256((out / name).read_bytes()).hexdigest() for name in ("contract.json", "station_evaluation.npz", "per_well.csv", "typewell_book.csv", "summary.json")}
    write_sealed_json(out / "closure.json", closure, kind="pad_measure_closure")
    record_step(out, "mutate", "checked", {"closure": closure})
    audited = subprocess.run([sys.executable, str(ROOT / "scripts" / "audit_pad_measurement.py"), "--run", str(out.resolve())], cwd=ROOT, capture_output=True, text=True, timeout=60)
    if audited.returncode:
        record_step(out, "verify", "reopened", {"audit_exit_code": audited.returncode})
        raise RuntimeError("Metric audit failed: " + audited.stderr[-2000:] + audited.stdout[-2000:])
    record_step(out, "verify", "checked", {"audit_sha256": hashlib.sha256((out / "audit.json").read_bytes()).hexdigest()})
    record_step(out, "admit", "checked" if summary["ok"] else "reopened", {"decision": summary["status"], "predictive_admitted": False})
    print(json.dumps(summary, indent=2, allow_nan=False))
    print(f"WP8 {summary['status']} audited; wrote {out}")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
