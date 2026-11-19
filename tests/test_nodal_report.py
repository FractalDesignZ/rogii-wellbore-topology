import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import render_nodal_evidence as report


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
