"""Resolve cited method concepts to source-bound AST definitions and checks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rwt.ast_evidence import build_ast_graph
from rwt.execution_graph import build_source_manifest


# Physical page counts authenticated during the supplied-document audit.
DOCUMENT_PAGES = {
    "7f0e822fdde98bdb7348fe565481e59b70470b240e43d23cc2b7ffac13d9cd75": 33,
    "52efecdf62380ee3881e5f4d61d3a241b80d2dd33e45ed305ea78cdeef6083b3": 118,
    "1ccf23ade43557a1468f7777486d4062768edfc39906526297b83a1764ed00c6": 9,
    "6e12260cd1994e29f946294b0913e4d14df5ba646e90159a2b923127a0748846": 135,
    "c6313bab3b7412fb5810ebe4c9753cc383c2eec6646bac0e7e1347c1f97d1096": 12,
    "d2b967107fc3c59cead2cc0a12dcc681482bd9d874e4c2c9c96f1cfafeb951ba": 27,
}


def require(value, message):
    if not value:
        raise ValueError(message)


def verify(root: Path, specification: dict, pdf_dir: Path | None = None):
    require(specification.get("schema") == "rwt.method-references.v1", "unsupported reference graph schema")
    refs = specification["references"]
    concepts = specification["concepts"]
    identities = [item["id"] for item in [*refs, *concepts]]
    require(len(identities) == len(set(identities)) and all(isinstance(k, str) and k for k in identities),
            "reference and concept IDs must be distinct strings")
    by_ref = {ref["id"]: ref for ref in refs}
    require(len({ref["sha256"] for ref in refs}) == len(refs), "duplicate source document identity")
    for ref in refs:
        require(type(ref["pages"]) is int and ref["pages"] > 0 and len(ref["sha256"]) == 64
                and set(ref["sha256"]) <= set("0123456789abcdef"), "invalid reference identity")
        require(DOCUMENT_PAGES.get(ref["sha256"]) == ref["pages"], "document identity or physical page count is unverified")
    targets = [target for concept in concepts for kind in ("operators", "checks") for target in concept[kind]]
    source = build_source_manifest(root, sorted({target["path"] for target in targets}))
    syntax = build_ast_graph(root, source)
    definitions = {(definition["node"].split("#", 1)[0], definition["qualified_name"]): definition["node"]
                   for definition in syntax["definitions"]}
    symbols = []
    edges = []
    cited = set()
    for concept in concepts:
        require(concept["status"] in ("implemented", "hypothesis"), "invalid concept status")
        require(concept["citations"] and concept["claim"], "each concept needs a claim and source")
        if concept["status"] == "implemented":
            require(concept["operators"] and concept["checks"], "implemented concepts need operators and checks")
        else:
            require(not concept["operators"] and not concept["checks"], "hypotheses cannot claim implementation")
        for citation in concept["citations"]:
            ref = by_ref.get(citation["reference"])
            require(ref is not None, "unknown cited reference")
            cited.add(ref["id"])
            pages = citation["pages"]
            require(pages and all(type(page) is int and 1 <= page <= ref["pages"] for page in pages)
                    and len(pages) == len(set(pages)), "citation pages outside the physical PDF")
            edges.append({"source": ref["id"], "target": concept["id"], "kind": "informs", "pages": pages})
        for kind, relation in (("operators", "implemented_by"), ("checks", "checked_by")):
            for target in concept[kind]:
                if kind == "checks":
                    require(target["path"].startswith("tests/test_") and target["path"].endswith(".py")
                            and target["symbol"].split(".")[-1].startswith("test_"), "a check must name a discoverable test")
                identity = (target["path"], target["symbol"])
                require(identity in definitions, "unresolved Python definition " + "::".join(identity))
                node = definitions[identity]
                symbols.append({**target, "ast_node": node})
                edges.append({"source": concept["id"], "target": node, "kind": relation})
    require(cited == set(by_ref), "a declared reference has no concept link")
    by_concept = {concept["id"]: concept for concept in concepts}
    for flow in specification["flows"]:
        require(flow["source"] in by_concept and flow["target"] in by_concept
                and flow["source"] != flow["target"] and flow["kind"] in ("data", "control", "proposed"),
                "invalid method process or control link")
        if by_concept[flow["source"]]["status"] == "hypothesis":
            require(flow["kind"] == "proposed", "a hypothesis needs a proposed process link")
        edges.append(flow)
    documents = {ref["id"]: "unchecked" for ref in refs}
    if pdf_dir is not None:
        observed = {}
        for path in Path(pdf_dir).glob("*.pdf"):
            with path.open("rb") as handle:
                observed[hashlib.file_digest(handle, "sha256").hexdigest()] = path.name
        require(all(ref["sha256"] in observed for ref in refs), "a cited PDF is missing or has changed")
        documents = {ref["id"]: observed[ref["sha256"]] for ref in refs}
    graph = {"schema": "rwt.verified-method-graph.v1", "references": refs, "concepts": concepts,
             "definitions": symbols, "edges": edges, "document_matches": documents,
             "source_manifest_sha256": source["manifest_sha256"], "ast_graph_sha256": syntax["graph_sha256"],
             "scope": "Source and citation identity checks. Conceptual applicability and test execution require their own evidence."}
    return graph, source, syntax


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=ROOT / "qa/method_references.json")
    parser.add_argument("--pdf-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    graph, source, syntax = verify(ROOT, json.loads(args.spec.read_bytes()), args.pdf_dir)
    args.out.mkdir(parents=True, exist_ok=False)
    for name, value in (("method_graph", graph), ("source_graph", source), ("ast_graph", syntax)):
        with (args.out / (name + ".json")).open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
    print(json.dumps({"concepts": len(graph["concepts"]), "edges": len(graph["edges"]),
                      "pdf_identities_checked": args.pdf_dir is not None, "out": str(args.out)}))


if __name__ == "__main__":
    main()
