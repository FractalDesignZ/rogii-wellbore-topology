"""Deterministic, source-bound syntax evidence for finite worker plans."""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import sys
from typing import Any

from .execution_graph import _digest, verify_source_manifest


_DEFINITIONS = (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
_PROJECTED = (ast.Module, ast.stmt, ast.Call, ast.Lambda, ast.match_case,
              ast.ExceptHandler, ast.comprehension, ast.IfExp)
_LOOPS = (ast.For, ast.AsyncFor, ast.While)
_FIELD_ROLES = {
    "If": {"test": "condition", "body": "if_true", "orelse": "if_false"},
    "IfExp": {"test": "condition", "body": "if_true", "orelse": "if_false"},
    "Match": {"subject": "match_subject", "cases": "match_case"},
    "match_case": {"pattern": "case_pattern", "guard": "case_guard", "body": "case_body"},
    "Try": {"body": "try_body", "handlers": "exception_handler", "orelse": "try_else",
            "finalbody": "finally_body"},
    "ExceptHandler": {"type": "exception_type", "body": "handler_body"},
    "comprehension": {"target": "loop_target", "iter": "loop_iterable", "ifs": "loop_filter"},
}


def _role(parent: ast.AST, field: str) -> str:
    if isinstance(parent, _LOOPS):
        return {"body": "loop_body", "orelse": "loop_else", "test": "condition",
                "iter": "loop_iterable", "target": "loop_target"}.get(field, field)
    kind = type(parent).__name__
    return _FIELD_ROLES.get("Try" if kind == "TryStar" else kind, {}).get(field, field)


def _children(node: ast.AST):
    """Yield ordered AST children, preserving the field name and original list index."""
    for field, value in ast.iter_fields(node):
        if isinstance(value, ast.AST):
            yield field, None, value
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, ast.AST):
                    yield field, index, item


def _project(tree: ast.Module, filename: str, graph: dict[str, Any]) -> str:
    def visit(node, address, ancestor, field_path, owner, qualified_name):
        if isinstance(node, _PROJECTED):
            node_id = filename + "#" + address
            kind = type(node).__name__
            line = getattr(node, "lineno", None)
            span = None if line is None else [line, node.col_offset, node.end_lineno, node.end_col_offset]
            graph["nodes"].append({"id": node_id, "kind": kind, "span": span,
                                   "enclosing_definition": owner})
            if ancestor is not None:
                parent_id, parent_node = ancestor
                field, index = field_path[0]
                segments = [name if i is None else f"{name}[{i}]" for name, i in field_path]
                graph["edges"].append({"source": parent_id, "target": node_id, "field": field,
                                       "index": index, "field_path": ".".join(segments),
                                       "role": _role(parent_node, field)})
            if isinstance(node, _DEFINITIONS):
                name = "<lambda>" if isinstance(node, ast.Lambda) else node.name
                qualified_name = name if not qualified_name else qualified_name + "." + name
                graph["definitions"].append({"node": node_id, "qualified_name": qualified_name})
                owner = node_id
            if isinstance(node, ast.Call):
                graph["calls"].append({"node": node_id, "expression": ast.unparse(node.func)})
            ancestor = (node_id, node)
            field_path = ()

        for ordinal, (field, index, child) in enumerate(_children(node)):
            visit(child, f"{address}.{ordinal}", ancestor, (*field_path, (field, index)),
                  owner, qualified_name)

    visit(tree, "0", None, (), None, "")
    return filename + "#0"


def build_ast_graph(repo_root: Path, source_manifest: dict[str, Any]) -> dict[str, Any]:
    """Project manifest-verified Python syntax and bind the result to the full trees.

    File-relative path plus ordered AST-child address identifies each retained node.
    Each edge gives the full field/index path through any omitted expression nodes.
    Scope means syntactic containment, including decorators and defaults. Line numbers
    are one-based and columns are zero-based UTF-8 byte offsets, as in Python's AST.
    Calls identify syntax only. Traversal neither imports nor executes worker source.
    """
    root = Path(repo_root).resolve()
    verified = verify_source_manifest(root, source_manifest)
    if not verified["passed"]:
        raise ValueError("source manifest verification failed: " + ", ".join(verified["findings"]))
    graph: dict[str, Any] = {
        "schema": "rwt.ast-evidence.v1",
        "source_manifest_sha256": source_manifest["manifest_sha256"],
        "parser_version": list(sys.version_info[:2]),
        "projection": "Module, every statement, Call, Lambda, match_case, ExceptHandler, comprehension, IfExp",
        "parse_tree_digest_format": "UTF-8 ast.dump(annotate_fields=True, include_attributes=False)",
        "span_format": "[start line, start byte column, end line, end byte column] or null",
        "syntax_only": True, "dynamic_calls_resolved": False,
        "files": [], "nodes": [], "edges": [], "definitions": [], "calls": [],
    }
    for source in source_manifest["files"]:
        relative = source["path"]
        path = (root / relative).resolve()
        path.relative_to(root)
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != source["sha256"]:
            raise ValueError(f"source hash changed before AST parsing: {relative}")
        tree = ast.parse(raw, filename=relative)
        full_dump = ast.dump(tree, annotate_fields=True, include_attributes=False)
        file_root = _project(tree, relative, graph)
        graph["files"].append({
            "path": relative, "source_id": source["id"], "sha256": source["sha256"],
            "parse_tree_sha256": hashlib.sha256(full_dump.encode("utf-8")).hexdigest(),
            "parse_node_count": sum(1 for _ in ast.walk(tree)), "root": file_root,
        })
    return {**graph, "graph_sha256": _digest(graph)}
