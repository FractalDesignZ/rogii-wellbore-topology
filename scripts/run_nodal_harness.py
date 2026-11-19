#!/usr/bin/env python3
"""Run the fixed development mission, or verify its archived evidence in place."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rwt.block_model import select_development_wells
from rwt.execution_graph import (
    build_source_manifest, run_bounded, verify_execution_trace, verify_source_manifest,
)

ARMS = ("graph", "tree", "forest", "neural")
DAG = {"prepare": (), **{arm: ("prepare",) for arm in ARMS}, "audit": ARMS}
OUTPUTS = {"prepare": ("bundle.npz", "bundle.json"),
           **{arm: (arm + ".npz", arm + ".json") for arm in ARMS}, "audit": ("metrics.json",)}
SOURCES = ["scripts/nodal_worker.py", "scripts/run_nodal_harness.py", "src/rwt/ast_evidence.py"]
TIMEOUT = 1800
THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")
REFERENCE = "https://github.com/ZuluYokohama/mission-lab/tree/dfdf8b7dded59f06d0fc2bc581a9292de305e977"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_hash(path):
    require(path.is_file() and not path.is_symlink(), f"missing or linked file: {path}")
    before = path.stat()
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            sha.update(chunk)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            f"file changed while hashing: {path}")
    return sha.hexdigest()


def digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def read(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write(path, data, *, compact=False):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, indent=None if compact else 2,
                  separators=(",", ":") if compact else None, allow_nan=False)
        handle.write("\n")


def stage_spec():
    return [{"id": name, "dependencies": list(deps), "outputs": list(OUTPUTS[name])}
            for name, deps in DAG.items()]


def selected(manifest, mode):
    if mode == "all-development":
        return select_development_wells(manifest)
    require(mode == "fixed-pilot" and isinstance(manifest.get("pilot_well_ids"), list),
            "fixed pilot requires explicit pilot_well_ids")
    return select_development_wells(manifest, manifest["pilot_well_ids"])


def raw_files(data_root, wells, manifest):
    train = data_root / "train"
    require(train.is_dir() and train.resolve() == train, "train must be a regular data directory")
    files = {}
    for wid in wells:
        records = {r["path"]: r["sha256"] for r in manifest["files"][wid]} if "files" in manifest else None
        for suffix in ("__horizontal_well.csv", "__typewell.csv"):
            name = f"train/{wid}{suffix}"
            path = data_root / name
            require(path.resolve().parent == train, f"source outside train: {name}")
            value = file_hash(path)
            if records is not None:
                require(records.get(name) == value, f"raw input differs from cohort: {name}")
            files[name] = value
    return files


def make_contract(repo, cohort, data_root, mode, source, ast_graph):
    cohort_hash = file_hash(cohort)
    manifest = read(cohort)
    wells = selected(manifest, mode)
    files = raw_files(data_root, wells, manifest)
    require(cohort_hash == file_hash(cohort), "cohort changed during planning")
    return {"schema": "rwt.nodal-mission.v1", "mission": "Compare fixed nodal-tree readouts on held-out development wells.",
            "controller_reference": REFERENCE, "source_graph_sha256": source["manifest_sha256"],
            "ast_graph_sha256": ast_graph["graph_sha256"],
            "repo_root": str(repo), "cohort_path": str(cohort), "cohort_sha256": cohort_hash,
            "selection": mode, "wells": wells, "data_root": str(data_root), "input_files": files,
            "folds": 3, "seed": 42, "stages": stage_spec(), "python": os.path.abspath(sys.executable),
            "resources": {"concurrent_workers": 1, "worker_wall_timeout_s": TIMEOUT, "library_thread_env": 1},
            "scientific_origin": {
                "observations": "Paired development lateral survey/GR/visible TVT and reference TVT/GR from the hashed raw CSVs.",
                "relationships": "First-party RWT block observations, drafting coordinates, and bounded, ordered messages from the requested nodal-tree framework.",
                "models": "Fixed graph readout, decision tree, random forest and feed-forward neural readouts with matched controls. The source graph freezes the first-party parameters and functions before fitting.",
                "dependencies": "First-party rwt code, NumPy, pandas, SciPy and scikit-learn. Mission Lab informs the execution design."},
            "evaluation": "Compare each readout with last-visible TVT and its matched relationship control on identical scored stations."}


def validate_freeze(run, repo, contract, source):
    require(contract["schema"] == "rwt.nodal-mission.v1" and contract["repo_root"] == str(repo)
            and contract["folds"] == 3 and contract["seed"] == 42, "mission identity mismatch")
    require(contract["stages"] == stage_spec() and source["entries"] == sorted(SOURCES), "mission DAG or entry points changed")
    require(contract["resources"] == {"concurrent_workers": 1, "worker_wall_timeout_s": TIMEOUT, "library_thread_env": 1},
            "mission execution bounds changed")
    require(source["manifest_sha256"] == contract["source_graph_sha256"], "source graph binding mismatch")
    require(verify_source_manifest(repo, source)["passed"], "live source differs from frozen source graph")
    return boundary(run, repo, contract, source)


def boundary(run, repo, contract, source):
    cohort = Path(contract["cohort_path"])
    require(file_hash(cohort) == contract["cohort_sha256"], "cohort manifest changed")
    manifest = read(cohort)
    wells = selected(manifest, contract["selection"])
    require(wells == contract["wells"], "frozen well order or role changed")
    raw = raw_files(Path(contract["data_root"]), wells, manifest)
    require(raw == contract["input_files"], "raw inputs differ from mission")
    for row in source["files"]:
        require(file_hash(repo / row["path"]) == row["sha256"], f"changed source: {row['path']}")
    return {"contract.json": file_hash(run / "contract.json"), "source_graph.json": file_hash(run / "source_graph.json"),
            "ast_graph.json": file_hash(run / "ast_graph.json"),
            "raw_file_map_sha256": digest(raw), "cohort_sha256": contract["cohort_sha256"]}


def ancestors(stage):
    needed = set(DAG[stage])
    for name in reversed(DAG):
        if name in needed:
            needed.update(DAG[name])
    return [name for name in DAG if name in needed]


def stage_inputs(run, stage, frozen, records):
    parents, artifacts = {}, {}
    for name in ancestors(stage):
        receipt_path = run / f"{name}.receipt.json"
        row = records[name]
        require(read(receipt_path) == row, f"changed {name} receipt")
        parents[name] = file_hash(receipt_path)
        for key, expected in row["outputs"].items():
            value = file_hash(run / key)
            require(value == expected, f"changed dependency output: {key}")
            artifacts[key] = value
    return {"mission": frozen, "upstream_receipts": parents, "upstream_outputs": artifacts}


def command(repo, run, contract, stage):
    return [contract["python"], "-B", str(repo / SOURCES[0]), stage, str(run)]


@contextmanager
def bounded_threads():
    original = {key: os.environ.get(key) for key in THREAD_VARS}
    try:
        os.environ.update({key: "1" for key in THREAD_VARS})
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def execute(repo, cohort, data_root, run, *, all_development=False):
    from rwt.ast_evidence import build_ast_graph

    repo, cohort, data_root, run = (Path(p).resolve() for p in (repo, cohort, data_root, run))
    require(run != data_root and data_root not in run.parents, "run output overlaps raw dataset")
    source = build_source_manifest(repo, SOURCES)
    ast_graph = build_ast_graph(repo, source)
    contract = make_contract(repo, cohort, data_root, "all-development" if all_development else "fixed-pilot",
                             source, ast_graph)
    run.mkdir(parents=True, exist_ok=False)
    write(run / "contract.json", contract)
    write(run / "source_graph.json", source, compact=True)
    write(run / "ast_graph.json", ast_graph, compact=True)
    del ast_graph
    frozen = validate_freeze(run, repo, contract, source)
    records, statuses = {}, {}
    with bounded_threads():
        for stage, deps in DAG.items():
            require(boundary(run, repo, contract, source) == frozen, "mission inputs changed before execution")
            inputs = stage_inputs(run, stage, frozen, records)
            require(all(not os.path.lexists(run / n) for n in OUTPUTS[stage]),
                    f"{stage} outputs existed before dispatch")
            process = run_bounded(command(repo, run, contract, stage), cwd=repo, trace_path=run / "trace.jsonl",
                                  node_id=stage, timeout_s=TIMEOUT, dependencies=deps, statuses=statuses)
            if process["status"] != "SUCCEEDED":
                write(run / f"{stage}.failure.json", process)
                raise RuntimeError(f"{stage} {process['status']}; see {run / (stage + '.stderr.log')}")
            require(boundary(run, repo, contract, source) == frozen, "mission inputs changed during execution")
            require(stage_inputs(run, stage, frozen, records) == inputs, f"{stage} dependency changed during execution")
            outputs = {key: file_hash(run / key) for key in OUTPUTS[stage]}
            row = {"stage": stage, "process": process, "input_root_sha256": digest(inputs), "inputs": inputs,
                   "outputs": outputs, "output_root_sha256": digest(outputs)}
            write(run / f"{stage}.receipt.json", row)
            records[stage], statuses[stage] = row, process["status"]
            print(f"{stage} completed in {process['wall_seconds']:.2f} s", flush=True)
    evidence = replay(run, repo)
    write(run / "execution_evidence.json", evidence)
    return summary(run, evidence)


def replay(run, repo):
    from rwt.ast_evidence import build_ast_graph

    contract, source = read(run / "contract.json"), read(run / "source_graph.json")
    frozen = validate_freeze(run, repo, contract, source)
    graph = read(run / "ast_graph.json")
    require(graph["graph_sha256"] == contract["ast_graph_sha256"] and graph == build_ast_graph(repo, source),
            "AST graph differs from source replay")
    del graph
    trace_path = run / "trace.jsonl"
    chain = verify_execution_trace(trace_path)
    require(chain["passed"] and chain["events"] == 2*len(DAG), "incomplete or invalid execution trace")
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    records, observed_nodes, observed_edges, stage_seals = {}, [], [], {}
    for i, (stage, deps) in enumerate(DAG.items()):
        start, finish = events[2*i:2*i+2]
        require(start["event"] == "process_started" and finish["event"] == "process_finished"
                and start["node_id"] == finish["node_id"] == stage, "invalid stage order or terminal event")
        require(start["command"] == command(repo, run, contract, stage) and start["cwd"] == str(repo)
                and start["timeout_s"] == TIMEOUT, f"{stage} launch differs from contract")
        require(start["dependencies"] == finish["dependencies"] == list(deps), f"{stage} dependencies changed")
        require(type(start["pid"]) is int and start["pid"] > 0 and start["pid"] == finish["pid"]
                and finish["returncode"] == 0 and finish["status"] == "SUCCEEDED", f"unsuccessful {stage} process")
        receipt_path = run / f"{stage}.receipt.json"
        row = read(receipt_path)
        process = {k: v for k, v in finish.items() if k not in
                   {"sequence","previous_sha256","sha256","utc","monotonic_s","event"}}
        require(row["stage"] == stage and row["process"] == process, f"{stage} receipt differs from trace")
        for stream in ("stdout","stderr"):
            log = run / f"{stage}.{stream}.log"
            require(process[stream+"_path"] == str(log) and process[stream+"_sha256"] == file_hash(log),
                    f"changed {stage} {stream}")
        inputs = stage_inputs(run, stage, frozen, records)
        require(row["inputs"] == inputs and row["input_root_sha256"] == digest(inputs), f"{stage} input binding mismatch")
        outputs = {key: file_hash(run / key) for key in OUTPUTS[stage]}
        require(row["outputs"] == outputs and row["output_root_sha256"] == digest(outputs), f"{stage} outputs changed")
        observed_nodes.append(start["node_id"])
        observed_edges.extend([[dep, stage] for dep in start["dependencies"]])
        records[stage], stage_seals[stage] = row, file_hash(receipt_path)
    expected_edges = {(dep, name) for name, deps in DAG.items() for dep in deps}
    actual_edges = {tuple(edge) for edge in observed_edges}
    graph = {"executed_nodes": observed_nodes, "executed_edges": observed_edges,
             "missing_nodes": sorted(set(DAG)-set(observed_nodes)),
             "extra_nodes": sorted(set(observed_nodes)-set(DAG)),
             "missing_edges": sorted(map(list, expected_edges-actual_edges)),
             "extra_edges": sorted(map(list, actual_edges-expected_edges))}
    metrics = replay_metrics(run)
    body = {"schema": "rwt.nodal-execution.v1", "mission": frozen, "trace": chain,
            "stage_receipt_sha256": stage_seals, "execution_graph_delta": graph,
            "integration_completed": True, "merit_evidence": "metrics.json", "metric_replay": metrics,
            "interpretation": "Receipt replay checks the observed run. The worker audit is repeated and a separate verifier uses a within-block variance decomposition to recalculate raw station errors. The digest chain binds local evidence and provides no external signature."}
    return {**body, "evidence_root_sha256": digest(body)}


def replay_metrics(run):
    from rwt.nodal_study import audit_run

    metrics = read(run / "metrics.json")
    require(audit_run(run) == metrics, "scientific audit differs from stored metrics")
    methods = metrics["methods"]
    controls = [arm + "_control" for arm in ARMS]
    require(set(methods) == {*ARMS, *controls, "last_visible"} and set(metrics["merit"]) == set(ARMS),
            "scientific audit is missing an arm or comparator")
    for arm in ARMS:
        control = methods[arm + "_control"]
        require(methods[arm]["rows"] == methods["last_visible"]["rows"] == control["rows"] == metrics["n_stations"]
                and metrics["merit"][arm]["beats_last_visible"] == (methods[arm]["sse_ft2"] < methods["last_visible"]["sse_ft2"])
                and metrics["merit"][arm]["beats_matched_control"] == (methods[arm]["sse_ft2"] < control["sse_ft2"]),
                f"{arm} merit or population mismatch")
    independent = replay_station_errors(run, metrics)
    return {"n_wells": metrics["n_wells"], "n_stations": metrics["n_stations"],
            "rmse_ft": {name: row["rmse_ft"] for name, row in methods.items()},
            "merit": {arm: {"beats_last_visible": metrics["merit"][arm]["beats_last_visible"],
                            "beats_matched_control": metrics["merit"][arm]["beats_matched_control"],
                            "passes_both_development_comparisons": (metrics["merit"][arm]["beats_last_visible"]
                                                                    and metrics["merit"][arm]["beats_matched_control"])}
                      for arm in ARMS},
            "independent_station_sse_ft2": independent}


def replay_station_errors(run, metrics):
    """Independent block-mean/within-block variance decomposition of raw station SSE."""
    import numpy as np
    import pandas as pd
    from rwt.load import normalize_curve_nulls

    contract = read(run / "contract.json")
    identity = ("well_index", "first_station", "row_count", "fold")
    with np.load(run / "bundle.npz", allow_pickle=False) as bundle:
        require(bundle["well_ids"].tolist() == contract["wells"], "bundle well order differs from mission")
        keys = {name: bundle[name] for name in (*identity, "last_known", "center_station")}
    predictions = {"last_visible": keys["last_known"]}
    for arm in ARMS:
        with np.load(run / (arm + ".npz"), allow_pickle=False) as arrays:
            for key in identity:
                actual, expected = arrays[key], keys[key]
                require(actual.dtype == expected.dtype and np.array_equal(actual, expected), f"{arm} {key} mismatch")
            predictions[arm] = arrays["prediction"]
            predictions[arm+"_control"] = arrays["control_prediction"]
    offset, total_rows = 0, 0
    per_well = {name: [] for name in predictions}
    for wi, well in enumerate(contract["wells"]):
        path = Path(contract["data_root"]) / "train" / (well + "__horizontal_well.csv")
        raw = pd.read_csv(path, usecols=["TVT", "TVT_input"], float_precision="round_trip")
        visible = normalize_curve_nulls(raw["TVT_input"])
        known = np.flatnonzero(np.isfinite(visible))
        require(len(known) > 0 and np.array_equal(known, np.arange(len(known))), f"invalid heel: {well}")
        heel = visible[known[-1]]
        truth = normalize_curve_nulls(raw["TVT"])[len(known):] - heel
        require(len(truth) > 0 and np.isfinite(truth).all(), f"invalid scored toe: {well}")
        start = np.arange(0, len(truth), 8)
        size = np.minimum(8, len(truth)-start)
        block = slice(offset, offset+len(start))
        require(np.array_equal(keys["first_station"][block], len(known)+start)
                and np.array_equal(keys["center_station"][block], len(known)+start+(size-1)//2)
                and np.array_equal(keys["row_count"][block], size) and np.all(keys["well_index"][block] == wi)
                and np.all(keys["last_known"][block] == heel), f"source station footprint mismatch: {well}")
        mean = np.add.reduceat(truth, start)/size
        deviations = truth - mean[np.arange(len(truth)) // 8]
        within = np.sum(deviations**2)
        reported = metrics["by_well"][well]
        require(reported["rows"] == len(truth) and np.all(keys["fold"][block] == reported["fold"]),
                f"independent well coverage or fold differs: {well}")
        for name, values in predictions.items():
            between = size * ((values[block].astype(float) - heel) - mean)**2
            squared_error = float(within + np.sum(between))
            require(math.isclose(squared_error, reported["sse_ft2"][name], rel_tol=1e-10, abs_tol=1e-8),
                    f"independent station score differs: {well}/{name}")
            per_well[name].append(squared_error)
        total_rows += len(truth)
        offset += len(start)
    require(offset == len(keys["well_index"]) == metrics["n_blocks"] and total_rows == metrics["n_stations"]
            and len(contract["wells"]) == metrics["n_wells"], "independent population mismatch")
    pooled = {name: math.fsum(values) for name, values in per_well.items()}
    for name, sse in pooled.items():
        require(math.isclose(sse, metrics["methods"][name]["sse_ft2"], rel_tol=1e-10, abs_tol=1e-8)
                and math.isclose(math.sqrt(sse/total_rows), metrics["methods"][name]["rmse_ft"],
                                 rel_tol=1e-10, abs_tol=1e-8), f"independent station score differs: {name}")
    return pooled


def verify(run, repo=ROOT):
    run, repo = Path(run).resolve(), Path(repo).resolve()
    stored = read(run / "execution_evidence.json")
    actual = replay(run, repo)
    require(stored == actual, "stored execution evidence differs from replay")
    return summary(run, actual)


def summary(run, evidence):
    return {"integration_verified": True, "executed_nodes": evidence["execution_graph_delta"]["executed_nodes"],
            "evidence_root_sha256": evidence["evidence_root_sha256"], "metrics": str(run / "metrics.json"),
            "development_comparison": evidence["metric_replay"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, help="development cohort manifest")
    parser.add_argument("--data-root", type=Path, help="competition directory containing train/")
    parser.add_argument("--out", type=Path, help="fresh evidence directory")
    parser.add_argument("--all-development", action="store_true", help="use all manifest development wells")
    parser.add_argument("--verify", type=Path, metavar="EXISTING_RUN", help="read-only verification of a completed run")
    args = parser.parse_args(argv)
    if args.verify:
        if args.cohort or args.data_root or args.out or args.all_development:
            parser.error("--verify takes only the existing run directory")
        result = verify(args.verify)
    else:
        if not all((args.cohort, args.data_root, args.out)):
            parser.error("execution requires --cohort, --data-root, and --out")
        result = execute(ROOT, args.cohort, args.data_root, args.out, all_development=args.all_development)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
