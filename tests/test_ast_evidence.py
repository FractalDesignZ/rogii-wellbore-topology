import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rwt.ast_evidence import build_ast_graph
from rwt.execution_graph import build_source_manifest


WORKER = """\
class Survey:
    def ingest(self, rows):
        def measure(row):
            if accepted(row):
                return convert(row)
            else:
                return recover(row)
        for row in rows:
            self.log(measure(row))
        else:
            self.finish()
"""


def make_graph(tmp_path, code=WORKER):
    path = tmp_path / "worker.py"
    path.write_text(code, encoding="utf-8")
    manifest = build_source_manifest(tmp_path, [path])
    return path, manifest, build_ast_graph(tmp_path, manifest)


def test_file_and_graph_digests_and_replay(tmp_path):
    path, manifest, graph = make_graph(tmp_path)
    assert build_ast_graph(tmp_path, manifest) == graph
    body = {k: v for k, v in graph.items() if k != "graph_sha256"}
    assert graph["graph_sha256"] == hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    file, = graph["files"]
    assert file["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    tree = ast.parse(path.read_bytes())
    assert file["parse_tree_sha256"] == hashlib.sha256(ast.dump(tree).encode()).hexdigest()
    assert len(graph["nodes"]) < file["parse_node_count"]
    node_ids = [n["id"] for n in graph["nodes"]]
    assert len(node_ids) == len(set(node_ids))
    assert {e["target"] for e in graph["edges"]} == set(node_ids) - {file["root"]}
    assert graph["syntax_only"] and not graph["dynamic_calls_resolved"]


def test_branch_loop_enclosing_definition_and_call_identity(tmp_path):
    _, _, graph = make_graph(tmp_path)
    nodes = {n["id"]:n for n in graph["nodes"]}
    incoming = {e["target"]:e for e in graph["edges"]}
    defs = {d["qualified_name"]:d["node"] for d in graph["definitions"]}
    calls = {c["expression"]:c["node"] for c in graph["calls"]}
    assert nodes[defs["Survey.ingest.measure"]]["enclosing_definition"] == defs["Survey.ingest"]
    assert nodes[calls["accepted"]]["enclosing_definition"] == defs["Survey.ingest.measure"]
    assert nodes[calls["measure"]]["enclosing_definition"] == defs["Survey.ingest"]
    assert nodes[calls["convert"]]["span"] == [5, 23, 5, 35]
    true_edge = incoming[incoming[calls["convert"]]["source"]]
    false_edge = incoming[incoming[calls["recover"]]["source"]]
    assert (true_edge["role"],false_edge["role"]) == ("if_true","if_false")
    assert true_edge["source"] == false_edge["source"]
    assert (false_edge["field"],false_edge["index"],false_edge["field_path"]) == ("orelse",0,"orelse[0]")
    assert incoming[incoming[calls["self.log"]]["source"]]["role"] == "loop_body"
    assert incoming[incoming[calls["self.finish"]]["source"]]["role"] == "loop_else"
    assert incoming[calls["measure"]]["source"] == calls["self.log"]


def test_omitted_expression_path_and_literal_changes(tmp_path):
    path, first_manifest, first = make_graph(tmp_path, "def f(): return 1 + g()\n")
    call, = first["calls"]
    edge, = [e for e in first["edges"] if e["target"] == call["node"]]
    assert edge["field_path"] == "value.right"
    path.write_text("def f(): return 2 + g()\n")
    with pytest.raises(ValueError, match="source manifest verification"):
        build_ast_graph(tmp_path, first_manifest)
    second = build_ast_graph(tmp_path, build_source_manifest(tmp_path, [path]))
    assert second["nodes"] == first["nodes"]
    assert second["files"][0]["parse_tree_sha256"] != first["files"][0]["parse_tree_sha256"]
    assert second["graph_sha256"] != first["graph_sha256"]


def test_manifest_tamper_and_same_name_definitions(tmp_path):
    _, manifest, graph = make_graph(tmp_path, "def f(): pass\ndef f(): pass\n")
    first, second = graph["definitions"]
    assert first["qualified_name"] == second["qualified_name"] == "f"
    assert first["node"] != second["node"]
    forged = deepcopy(manifest)
    forged["files"][0]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="manifest_digest_mismatch"):
        build_ast_graph(tmp_path, forged)
