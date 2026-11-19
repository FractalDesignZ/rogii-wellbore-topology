"""Pass the admitted topology of a TVT search to an optional local provider."""
from __future__ import annotations

import hashlib
import json
from numbers import Integral

from .relation_search import SearchResult


def summarize_native_lift(result: SearchResult, *, levels: int = 2,
                          pair_cap: int = 20_000) -> dict:
    """Measure shared endpoints across successive native relation-node levels.

    The provider receives only integer endpoint indices. The local edge list
    binds its level-one nodes to the observations that produced each crossing.
    """
    for key, value, minimum in (("levels", levels, 1), ("pair_cap", pair_cap, 0)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{key} must be an integer of at least {minimum}")
    try:
        import network_engine
        from network_engine import from_edges, iterate
    except ImportError as error:
        raise RuntimeError("The native lift requires the separately installed Network Engine") from error

    node_ids = [node.state.node_id for node in result.nodes]
    indices = {node_id: i for i, node_id in enumerate(node_ids)}
    edges = [{"source": indices[step.source_node_id], "target": indices[step.proposed.node_id],
              "observation_id": step.observation_id}
             for step in result.steps if step.admitted]
    graph = {"node_ids": node_ids, "edges": edges}
    canonical = json.dumps(graph, sort_keys=True, separators=(",", ":")).encode("utf-8")
    native = from_edges(len(node_ids), [edge["source"] for edge in edges],
                        [edge["target"] for edge in edges])
    measured = iterate(native, levels=int(levels), cap=int(pair_cap))
    counts = [{"nodes": item["n_nodes"], "edges": item["n_edges"]} for item in measured["levels"]]
    if counts[0] != {"nodes": len(node_ids), "edges": len(edges)}:
        raise RuntimeError("Native level zero does not match the admitted search graph")
    return {"schema": "rwt.native-relation-lift.v1", "provider_version": network_engine.__version__,
            "graph_sha256": hashlib.sha256(canonical).hexdigest(), "graph": graph,
            "search_status": result.status, "search_complete": result.complete,
            "requested_levels": int(levels), "pair_cap": int(pair_cap),
            "levels": counts, "capped": measured["capped"]}
