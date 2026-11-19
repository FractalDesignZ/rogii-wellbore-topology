"""Process/receipt falsifiers for the nodal mission, using a tiny fixture worker."""
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_nodal_harness as harness


WORKER = """
import json
from pathlib import Path
import sys

stage, out = sys.argv[1], Path(sys.argv[2])
print(stage, flush=True)
if stage == "audit":
    (out / "metrics.json").write_text(json.dumps({"fixture": True}))
else:
    stem = "bundle" if stage == "prepare" else stage
    (out / (stem + ".npz")).write_bytes(stem.encode())
    (out / (stem + ".json")).write_text(json.dumps({"written_by": stage}))
"""


def fixture_inputs(root):
    repo, data = root / "project tree", root / "data"
    (repo / "scripts").mkdir(parents=True)
    (repo / "src/rwt").mkdir(parents=True)
    (repo / harness.SOURCES[0]).write_text(WORKER)
    (repo / harness.SOURCES[1]).write_text("def launch(): pass\n")
    (repo / harness.SOURCES[2]).write_text("def emit(): pass\n")
    (data / "train").mkdir(parents=True)
    for well in ("A","B","C"):
        for suffix in ("horizontal_well","typewell"):
            (data / "train" / (well+"__"+suffix+".csv")).write_text(well+suffix)
    cohort = root / "cohort.json"
    harness.write(cohort, {"pilot_well_ids": ["C","A","B"], "development_well_ids": ["A","B","C","D"],
                           "confirmation_well_ids": ["HOLD"]})
    return repo, cohort, data


@contextmanager
def replace_bytes(path, replace):
    original = path.read_bytes()
    if replace is None:
        path.unlink()
    else:
        path.write_bytes(replace(original))
    try:
        yield
    finally:
        path.write_bytes(original)


def rechain(events):
    previous = "0"*64
    for i, event in enumerate(events):
        event.pop("sha256")
        event.update(sequence=i+1, previous_sha256=previous)
        previous = harness.digest(event)
        event["sha256"] = previous
    return ("\n".join(map(json.dumps, events))+"\n").encode()


def test_fixture_dag_replay_detects_source_raw_artifacts_logs_and_trace(tmp_path):
    repo, cohort, data = fixture_inputs(tmp_path)
    run = tmp_path / "pilot"
    # Isolate the process evidence contract. The study tests own numerical replay.
    with patch.object(harness, "replay_metrics", return_value={"fixture": True}):
        result = harness.execute(repo,cohort,data,run)
        assert result["integration_verified"]
        assert harness.read(run/"contract.json")["wells"] == ["C","A","B"]
        assert len(harness.read(run/"contract.json")["input_files"]) == 6
        stat = {p.name:p.stat().st_mtime_ns for p in run.iterdir()}
        assert harness.verify(run,repo) == result
        assert stat == {p.name:p.stat().st_mtime_ns for p in run.iterdir()}
        audit = harness.read(run/"audit.receipt.json")
        assert set(audit["inputs"]["upstream_receipts"]) == {"prepare", *harness.ARMS}
        assert set(audit["inputs"]["upstream_outputs"]) == {
            name for stage in audit["inputs"]["upstream_receipts"] for name in harness.OUTPUTS[stage]}
        for path in (run/"tree.npz",run/"graph.stdout.log",run/"prepare.receipt.json",
                     data/"train/A__typewell.csv",run/"trace.jsonl",run/"metrics.json"):
            with replace_bytes(path, None):
                with pytest.raises((ValueError,FileNotFoundError)):
                    harness.verify(run,repo)
        for path in (run/"forest.npz",run/"tree.stderr.log",cohort,
                     data/"train/C__horizontal_well.csv",repo/"scripts/nodal_worker.py"):
            with replace_bytes(path,lambda raw:raw+b"\n"):
                with pytest.raises(ValueError):
                    harness.verify(run,repo)
        for name,key in (("execution_evidence.json","evidence_root_sha256"),
                         ("source_graph.json","manifest_sha256"),("ast_graph.json","graph_sha256"),
                         ("neural.receipt.json","input_root_sha256")):
            with replace_bytes(run/name, lambda raw:json.dumps({**json.loads(raw),key:"0"*64}).encode()):
                with pytest.raises(ValueError):
                    harness.verify(run,repo)
        trace=run/"trace.jsonl"
        for edit, message in (("dependencies","dependencies changed"),("order","stage order"),
                              ("incomplete","incomplete"),("pid","unsuccessful")):
            rows=[json.loads(line) for line in trace.read_bytes().splitlines()]
            if edit == "dependencies":
                rows[2]["dependencies"] = []
                rows[3]["dependencies"] = []
            elif edit == "order":
                rows[2:6] = rows[4:6] + rows[2:4]
            elif edit == "incomplete":
                rows.pop()
            else:
                rows[1]["pid"] += 1
            with replace_bytes(trace, lambda _:rechain(rows)):
                assert harness.verify_execution_trace(trace)["passed"]
                with pytest.raises(ValueError,match=message):
                    harness.verify(run,repo)


def test_worker_cannot_replace_a_sealed_upstream_artifact(tmp_path):
    repo, cohort, data = fixture_inputs(tmp_path)
    run=tmp_path/"tampered"
    worker=repo/"scripts/nodal_worker.py"
    with worker.open("a") as f:
        f.write('if stage == "graph": (out / "bundle.npz").write_bytes(b"changed")\n')
    with pytest.raises(ValueError, match="changed dependency output"):
        harness.execute(repo,cohort,data,run)
    assert (run/"prepare.receipt.json").is_file()
    assert not (run/"graph.receipt.json").exists()
    assert not (run/"tree.stdout.log").exists()
    assert not (run/"execution_evidence.json").exists()


def test_confirmation_role_rejected_before_opening_measurements(tmp_path):
    repo,cohort,data=fixture_inputs(tmp_path)
    manifest=harness.read(cohort)
    manifest["pilot_well_ids"] = ["A","B","HOLD"]
    cohort.write_text(json.dumps(manifest))
    with patch.object(harness, "raw_files", side_effect=AssertionError("opened measurements")):
        with pytest.raises(ValueError,match="development IDs"):
            harness.execute(repo,cohort,data,tmp_path/"rejected")


def test_separate_station_variance_replay_catches_erroneous_scores_and_keys(tmp_path):
    data, run = tmp_path / "data", tmp_path / "run"
    (data / "train").mkdir(parents=True)
    run.mkdir()
    raw = data / "train/A__horizontal_well.csv"
    raw.write_text("TVT,TVT_input\n111,111\n115,115\n" + "".join(f"{n},\n" for n in range(117,128)))
    harness.write(run / "contract.json", {"wells":["A"], "data_root":str(data)})
    first = np.array([2,10])
    counts = np.array([8,3])
    keys = {"first_station":first, "row_count":counts,"well_index":np.array([0,0]),"fold":np.array([1,1])}
    heel = np.array([115.,115.])
    np.savez(run / "bundle.npz", **keys, well_ids=np.array(["A"]), last_known=heel, center_station=[5,11])
    predictions = {"last_visible":heel}
    for i,arm in enumerate(harness.ARMS):
        main = np.array([118.,127.])+i
        control = np.array([121.,123.])-i
        np.savez(run / (arm+".npz"), **keys, prediction=main, control_prediction=control)
        predictions.update({arm:main, arm+"_control":control})
    truth = np.arange(117,128)
    sse = {name:float(np.sum((np.repeat(pred, counts)-truth)**2)) for name,pred in predictions.items()}
    metrics = {"n_blocks":2,"n_wells":1,"n_stations":11,
               "methods":{name:{"sse_ft2":v,"rmse_ft":(v/11)**.5} for name,v in sse.items()},
               "by_well":{"A":{"fold":1,"rows":11,"sse_ft2":dict(sse)}}}
    assert harness.replay_station_errors(run,metrics) == pytest.approx(sse)
    metrics["by_well"]["A"]["sse_ft2"]["tree_control"] += .1
    with pytest.raises(ValueError,match="independent station score differs"):
        harness.replay_station_errors(run,metrics)
    np.savez(run / "tree.npz", **{**keys, "first_station":first+1},
             prediction=predictions["tree"], control_prediction=predictions["tree_control"])
    with pytest.raises(ValueError,match="first_station mismatch"):
        harness.replay_station_errors(run,metrics)
