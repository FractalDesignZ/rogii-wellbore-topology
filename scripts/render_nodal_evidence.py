"""Render a verified nodal mission as a self-contained, offline HTML report."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
from html import escape
from io import BytesIO
import json
from pathlib import Path

import numpy as np

from run_nodal_harness import ARMS, digest, verify


STYLE = """
:root{color-scheme:light;font:16px/1.5 Inter,'Segoe UI',sans-serif;color:#182f31;background:#f4f5ef}
*{box-sizing:border-box}body{max-width:78rem;margin:auto;padding:2rem clamp(1rem,4vw,4rem) 5rem}
header,section{margin:0 0 2rem}h1{font-size:clamp(2rem,5vw,3.6rem);line-height:1.05;margin:.6rem 0}
h2{font-size:1.55rem;letter-spacing:-.025em;margin:.4rem 0}p{max-width:78ch}
a{color:#00675a}nav,summary{font-weight:600}nav a{margin-right:1.25rem}
table{width:100%;border-collapse:collapse;text-align:left;font-variant-numeric:tabular-nums}
th,td{padding:.55rem .7rem;border-bottom:1px solid #cbd6cd}thead{background:#dde7dc}
.scroll{overflow:auto}.number{text-align:right}.graph{width:100%;min-width:620px;height:auto}
.graph rect{fill:#e3ede5;stroke:#608a78}.graph path{fill:none;stroke:#7a9085;stroke-width:1.6}
.graph .scheduled{stroke-dasharray:5 4}.graph .artifact{stroke:#00675a;stroke-width:2.2}
.graph text{fill:#16372e;text-anchor:middle;font-size:15px}.graph .small{font-size:12px}
details{border-left:1px solid #bdcdc0;padding-left:.7rem;margin:.25rem 0}summary{cursor:pointer}
.module{background:#fff;padding:.6rem .8rem;border:1px solid #cbd6cd;margin:.5rem 0}
.syntax{font:13px/1.6 'Cascadia Mono',Consolas,monospace}.syntax summary{font-weight:500}
.leaf{margin-left:1.2rem}.meta{color:#50685e;font-size:.9em}code{overflow-wrap:anywhere}
"""


def execution_svg(receipts, delta):
    """Draw scheduling and application-reported artifact reads as separate edge types."""
    positions = {"prepare": (90, 119), "audit": (570, 119),
                 **{arm: (330, 33 + 57 * i) for i, arm in enumerate(ARMS)}}
    lines = ['<svg class="graph" viewBox="0 0 660 280" role="img" '
             'aria-label="Worker processes, scheduling dependencies and recorded artifact reads">'
             '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
             'markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10 Z"/>'
             '</marker></defs>']
    scheduled = delta.get("scheduled_edges", delta.get("executed_edges", []))
    artifact_pairs = sorted({(r["producer"], r["consumer"]) for r in delta.get("artifact_reads", [])})
    for source, target, kind in [(*edge, "scheduled") for edge in scheduled] + [(*edge, "artifact") for edge in artifact_pairs]:
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        mid = (x1 + x2) // 2
        if source == "prepare" and target == "audit":
            path = f"M{x1} {y1+22} C{x1} 274 {x1} 262 {mid} 262 S{x2} 274 {x2} {y2+25}"
        else:
            gap = 5 if kind == "artifact" else -5
            path = f"M{x1+71} {y1+gap} C{mid} {y1+gap} {mid} {y2+gap} {x2-74} {y2+gap}"
        lines.append(f'<path class="{kind}" marker-end="url(#arrow)" d="{path}"><title>'
                     f'{kind}: {escape(source)} → {escape(target)}</title></path>')
    for stage in delta["executed_nodes"]:
        proc = receipts[stage]["process"]
        x, y = positions[stage]
        lines.append(f'<g><title>{escape(stage)} · PID {int(proc["pid"])} · exit {int(proc["returncode"])}</title>'
                     f'<rect x="{x-71}" y="{y-22}" width="142" height="44" rx="5"/>'
                     f'<text x="{x}" y="{y-2}">{escape(stage)}</text><text class="small" '
                     f'x="{x}" y="{y+15}">PID {int(proc["pid"])} · {proc["wall_seconds"]:.2f} s</text></g>')
    return "".join(lines) + "</svg>"


def artifact_read_section(delta):
    if "artifact_reads" not in delta:
        return "<p>This historical run recorded scheduling dependencies only.</p>"
    rows = []
    for edge in delta["artifact_reads"]:
        rows.append(f'<tr><td>{escape(edge["producer"])}</td><td>{escape(edge["consumer"])}</td>'
                    f'<td><code>{escape(edge["artifact"])}</code></td><td><code>{escape(edge["sha256"])}</code></td></tr>')
    return ('<p>Dashed arrows show scheduling dependencies. Solid arrows show a worker parsing a file '
            'whose SHA-256 matched a producer output. One arrow can represent multiple files.</p>'
            '<details><summary>Worker artifact reads</summary><div class="scroll"><table><thead><tr>'
            '<th>Producer</th><th>Consumer</th><th>File</th><th>SHA-256 of parsed input</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div></details>')


def source_and_syntax(source, ast_graph):
    modules = {row["id"]: row["path"] for row in source["files"]}
    anchors = {path: f"module-{i}" for i, path in enumerate(modules.values())}
    imports = defaultdict(set)
    for edge in source["edges"]:
        imports[modules[edge["source"]]].add((modules[edge["target"]], edge["line"]))
    nodes = {n["id"]: n for n in ast_graph["nodes"]}
    children = defaultdict(list)
    for edge in ast_graph["edges"]:
        children[edge["source"]].append(edge)
    names = {d["node"]: d["qualified_name"] for d in ast_graph["definitions"]}
    names.update({c["node"]: c["expression"] for c in ast_graph["calls"]})

    def subtree(node_id, role):
        node = nodes[node_id]
        line = f'L{node["span"][0]}' if node["span"] else ""
        label = (f'<span class="meta">{escape(role)} · {line}</span> '
                 f'{escape(node["kind"])} {escape(names.get(node_id, ""))}')
        edges = children[node_id]
        if not edges:
            return f'<div class="leaf">{label}</div>'
        body = "".join(subtree(e["target"], e["role"]) for e in edges)
        return f"<details><summary>{label}</summary>{body}</details>"

    sections = []
    for f in ast_graph["files"]:
        path = f["path"]
        linked = ", ".join(f'<a href="#{anchors[target]}">{escape(target)} L{line}</a>'
                           for target, line in sorted(imports[path]))
        syntax = "".join(subtree(e["target"], e["role"]) for e in children[f["root"]])
        sections.append(f'<details class="module" id="{anchors[path]}"><summary><code>{escape(path)}</code>'
                        f'</summary><p>Resolved first-party imports · {linked or "none"}</p>'
                        f'<p class="meta">Source SHA-256 · <code>{escape(f["sha256"])}</code></p>'
                        f'<div class="syntax">{syntax}</div></details>')
    return "".join(sections)


def checked_bytes(path, expected_hash):
    snapshot = path.read_bytes()
    if hashlib.sha256(snapshot).hexdigest() != expected_hash:
        raise ValueError(f"display input differs from verified evidence: {path.name}")
    return snapshot


def checked_json(path, expected_hash):
    return json.loads(checked_bytes(path, expected_hash))


def observation_table(run, receipts):
    bundle = checked_json(run / "bundle.json", receipts["prepare"]["outputs"]["bundle.json"])
    trace = bundle["example_trace"]
    well, first = trace["node"]
    well_index = bundle["well_ids"].index(well)
    start, end = trace["block_rows_half_open"]
    rows = []
    for arm in ARMS:
        name = f"{arm}.npz"
        snapshot = checked_bytes(run / name, receipts[arm]["outputs"][name])
        with np.load(BytesIO(snapshot), allow_pickle=False) as pred:
            hits = np.flatnonzero((pred["well_index"] == well_index) & (pred["first_station"] == first))
            if len(hits) != 1 or int(pred["row_count"][hits[0]]) != end - start:
                raise ValueError(f"ambiguous or missing {arm} block in source trace")
            i = hits[0]
            rows.append(f'<tr><th scope="row">{arm.title()}</th><td class="number">{pred["prediction"][i]:.4f}</td>'
                        f'<td class="number">{pred["control_prediction"][i]:.4f}</td></tr>')
    reading = trace["center_gr"]
    gr = "missing" if reading["parsed_value"] is None else f'{reading["parsed_value"]:.4f}'
    edges = "".join(f'<li>Rows {int(e["from"][1])} → {int(e["to"][1])} · '
                    f'edge weight {e["initial_weight"]:.4f} → {e["revised_weight"]:.4f}</li>'
                    for e in trace["incident_edges"])
    return f"""<p>Well <code>{escape(well)}</code>, lateral rows {start}–{end-1}. The original GR
reading at row {int(reading['key'][2])} is {gr}. The last visible TVT comes from row
{int(trace['heel_tvt']['key'][2])}. Each prediction below has the same well, starting row and block length.</p>
<ul>{edges}</ul><div class="scroll"><table><thead><tr><th>Readout</th><th class="number">Predicted TVT (ft)</th>
<th class="number">Control TVT (ft)</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p class="meta">The shared key traces a reading to its model row. The predictions can use other readings and whole-well context.</p>"""


def render(run):
    verified = verify(run)
    evidence = json.loads((run / "execution_evidence.json").read_bytes())
    root = evidence.pop("evidence_root_sha256")
    if root != verified["evidence_root_sha256"] or digest(evidence) != root:
        raise ValueError("evidence root changed after verification")
    source, ast_graph = (checked_json(run / name, evidence["mission"][name])
                         for name in ("source_graph.json", "ast_graph.json"))
    receipts = {stage: checked_json(run / f"{stage}.receipt.json", seal)
                for stage, seal in evidence["stage_receipt_sha256"].items()}
    delta = evidence["execution_graph_delta"]
    metrics = evidence["metric_replay"]
    errors = metrics["rmse_ft"]
    rows = []
    for arm in ARMS:
        passed = metrics["merit"][arm]["passes_both_development_comparisons"]
        rows.append(f'<tr><th scope="row">{arm.title()}</th><td class="number">{errors[arm]:.4f}</td>'
                    f'<td class="number">{errors[arm+"_control"]:.4f}</td>'
                    f'<td>{"Passed" if passed else "Fell short"}</td></tr>')
    return f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Nodal mission evidence</title>
<style>{STYLE}</style><header><h1>Nodal mission evidence</h1>
<p>{metrics['n_wells']:,} development wells · {metrics['n_stations']:,} original toe stations · three well-grouped folds</p>
<nav><a href="#runtime">Execution graph</a><a href="#scores">Model comparison</a><a href="#observation">Trace a block</a>
<a href="#source">Source and AST</a></nav></header>
<section id="runtime"><h2>Observed execution graph</h2>
<p>Each node shows its process ID and elapsed time. The controller dispatched one worker at a time.</p>
<div class="scroll">{execution_svg(receipts, delta)}</div>{artifact_read_section(delta)}</section>
<section id="scores"><h2>Model comparison</h2><p>Station-weighted RMSE in feet.
Carrying forward the last visible TVT scored <strong>{errors['last_visible']:.4f} ft</strong>.
A readout passes when its error is lower than both that baseline and its matched control.</p>
<div class="scroll"><table><thead><tr><th>Readout</th><th class="number">Augmented</th>
<th class="number">Matched control</th><th>Development gate</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div></section>
<section id="observation"><h2>Trace a block</h2>{observation_table(run, receipts)}</section>
<section id="source"><h2>Source dependencies and syntax</h2>
<p>{len(source['files'])} first-party files · {len(ast_graph['nodes']):,} projected syntax nodes ·
{len(ast_graph['edges']):,} syntax edges. Expand a file to follow its imports, branch structure and call expressions.
Line numbers refer to its recorded source. Calls show their static spelling.</p>
{source_and_syntax(source, ast_graph)}</section>
<footer><p>Verified local evidence root · <code>{escape(verified['evidence_root_sha256'])}</code></p></footer></html>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path, help="new standalone HTML file")
    args = parser.parse_args()
    report = render(args.run.resolve())
    with args.output.open("x", encoding="utf-8") as out:
        out.write(report)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
