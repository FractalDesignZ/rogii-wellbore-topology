"""Render a verified nodal mission as a self-contained, offline HTML report."""
from __future__ import annotations

import argparse
from collections import defaultdict
from html import escape
from pathlib import Path

from run_nodal_harness import ARMS, read, verify


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
.graph text{fill:#16372e;text-anchor:middle;font-size:15px}.graph .small{font-size:12px}
details{border-left:1px solid #bdcdc0;padding-left:.7rem;margin:.25rem 0}summary{cursor:pointer}
.module{background:#fff;padding:.6rem .8rem;border:1px solid #cbd6cd;margin:.5rem 0}
.syntax{font:13px/1.6 'Cascadia Mono',Consolas,monospace}.syntax summary{font-weight:500}
.leaf{margin-left:1.2rem}.meta{color:#50685e;font-size:.9em}code{overflow-wrap:anywhere}
"""


def execution_svg(run, delta):
    """Render only vertices and dependencies present in the checked process graph."""
    positions = {"prepare": (90, 119), "audit": (570, 119),
                 **{arm: (330, 33 + 57 * i) for i, arm in enumerate(ARMS)}}
    lines = ['<svg class="graph" viewBox="0 0 660 240" role="img" '
             'aria-label="Prepare feeds four readouts, each feeding the audit">'
             '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
             'markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10 Z"/>'
             '</marker></defs>']
    for source, target in delta["executed_edges"]:
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        mid = (x1 + x2) // 2
        lines.append(f'<path marker-end="url(#arrow)" d="M{x1 + 71} {y1} '
                     f'C{mid} {y1} {mid} {y2} {x2 - 74} {y2}"/>')
    for stage in delta["executed_nodes"]:
        proc = read(run / f"{stage}.receipt.json")["process"]
        x, y = positions[stage]
        lines.append(f'<g><title>{escape(stage)} · PID {int(proc["pid"])} · exit {int(proc["returncode"])}</title>'
                     f'<rect x="{x-71}" y="{y-22}" width="142" height="44" rx="5"/>'
                     f'<text x="{x}" y="{y-2}">{escape(stage)}</text><text class="small" '
                     f'x="{x}" y="{y+15}">PID {int(proc["pid"])} · {proc["wall_seconds"]:.2f} s</text></g>')
    return "".join(lines) + "</svg>"


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


def render(run):
    verified = verify(run)
    evidence = read(run / "execution_evidence.json")
    if evidence["evidence_root_sha256"] != verified["evidence_root_sha256"]:
        raise ValueError("evidence root changed after verification")
    source, ast_graph = (read(run / name) for name in ("source_graph.json", "ast_graph.json"))
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
<nav><a href="#runtime">Execution graph</a><a href="#scores">Model comparison</a>
<a href="#source">Source and AST</a></nav></header>
<section id="runtime"><h2>Observed execution graph</h2>
<p>Each node shows its process ID and elapsed time. Arrows record dependencies. The controller dispatched one worker at a time.</p>
<div class="scroll">{execution_svg(run, delta)}</div></section>
<section id="scores"><h2>Model comparison</h2><p>Station-weighted RMSE in feet.
Carrying forward the last visible TVT scored <strong>{errors['last_visible']:.4f} ft</strong>.
A readout passes when its error is lower than both that baseline and its matched control.</p>
<div class="scroll"><table><thead><tr><th>Readout</th><th class="number">Augmented</th>
<th class="number">Matched control</th><th>Development gate</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div></section>
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
