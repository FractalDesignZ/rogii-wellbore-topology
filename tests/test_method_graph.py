import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from verify_method_graph import verify


def test_reference_links_resolve_to_actual_python_definitions():
    spec = json.loads((ROOT / "qa/method_references.json").read_bytes())
    graph, source, syntax = verify(ROOT, spec)
    assert len(graph["references"]) == 6
    assert graph["ast_graph_sha256"] == syntax["graph_sha256"]
    assert graph["source_manifest_sha256"] == source["manifest_sha256"]
    assert any(node["symbol"] == "artifact_read_evidence" for node in graph["definitions"])


@pytest.mark.parametrize("failure", ["symbol", "page", "unimplemented", "inflated_pages", "control_link", "false_check", "duplicate_pdf"])
def test_reference_graph_rejects_false_links(failure):
    spec = json.loads((ROOT / "qa/method_references.json").read_bytes())
    changed = copy.deepcopy(spec)
    if failure == "symbol":
        changed["concepts"][0]["operators"][0]["symbol"] = "missing_operator"
    elif failure == "page":
        changed["concepts"][0]["citations"][0]["pages"] = [999]
    elif failure == "inflated_pages":
        changed["references"][0]["pages"] = 200
        changed["concepts"][0]["citations"][0]["pages"] = [150]
    elif failure == "control_link":
        changed["flows"][0]["target"] = "missing_process"
    elif failure == "false_check":
        changed["concepts"][0]["checks"][0] = changed["concepts"][0]["operators"][0]
    elif failure == "duplicate_pdf":
        changed["references"].append({**changed["references"][0], "id": "duplicate"})
    else:
        changed["concepts"][-1]["status"] = "implemented"
    with pytest.raises(ValueError):
        verify(ROOT, changed)
