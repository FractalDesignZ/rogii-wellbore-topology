import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import render_nodal_evidence as report


def test_graph_distinguishes_scheduling_and_file_reads_including_legacy():
    receipts = {s: {"process": {"pid": 2, "returncode": 0, "wall_seconds": .1}}
                for s in ("prepare", "graph", "audit")}
    legacy = {"executed_nodes": list(receipts), "executed_edges": [["prepare", "graph"]]}
    assert 'class="artifact"' not in report.execution_svg(receipts, legacy)
    assert "scheduling dependencies only" in report.artifact_read_section(legacy)
    delta = {"executed_nodes": list(receipts), "scheduled_edges": [["prepare", "graph"]],
             "artifact_reads": [{"producer": "prepare", "consumer": "audit", "artifact": "bundle.npz", "sha256": "a"*64}]}
    svg = report.execution_svg(receipts, delta)
    assert "scheduled: prepare → graph" in svg and "artifact: prepare → audit" in svg
    assert "artifact: prepare → graph" not in svg
    legend = report.artifact_read_section(delta)
    assert "Dashed arrows" in legend and "Solid arrows" in legend and "bundle.npz" in legend and "a"*64 in legend


def test_report_binds_the_content_it_displays(tmp_path, monkeypatch):
    source = tmp_path / "source_graph.json"
    source.write_bytes(b'{"files": []}')
    body = {"mission": {source.name: hashlib.sha256(source.read_bytes()).hexdigest()}}
    root = report.digest(body)
    monkeypatch.setattr(report, "verify", lambda run: {"evidence_root_sha256": root})
    stored = tmp_path / "execution_evidence.json"
    stored.write_text(json.dumps({**body, "evidence_root_sha256": root}), encoding="utf-8")

    source.write_bytes(b'{"files": ["changed"]}')
    with pytest.raises(ValueError, match="source_graph.json"):
        report.render(tmp_path)

    body["mission"][source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
    stored.write_text(json.dumps({**body, "evidence_root_sha256": root}), encoding="utf-8")
    with pytest.raises(ValueError, match="evidence root changed"):
        report.render(tmp_path)
