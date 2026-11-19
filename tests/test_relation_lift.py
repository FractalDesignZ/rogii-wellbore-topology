import importlib.util
import json
from pathlib import Path
import runpy
from unittest.mock import patch

import pytest

from rwt.relation_lift import summarize_native_lift
from rwt.relation_search import BoundaryDecision, CandidateState, ObservedRelation, SearchGoal, search_relations


def test_periodic_readings_resolve_depth_and_repeated_readings_preserve_aliases():
    script = Path(__file__).resolve().parents[1] / "scripts/demo_relation_search.py"
    results = runpy.run_path(str(script))["phase_stack"]()
    assert [len(query.goal_states) for query in results.values()] == [15, 5, 1, 15]
    assert results["2 + 3 + 5 ft"].value_ft == 1017
    assert results["2 ft"].goal_states == results["2 ft repeated three times"].goal_states


def crossing_graph():
    anchor = CandidateState("example", 0, 0)
    candidates = {station: [CandidateState("example", station, x) for x in values]
                  for station, values in [(1, [1, 2]), (2, [1, 9])]}
    observations = [ObservedRelation(str(s), "example", s, s + 1, {"reading": 4}) for s in range(2)]
    return search_relations(anchor, candidates, observations, goal=SearchGoal(2),
                            boundary=lambda o, s, t: BoundaryDecision("unresolved" if t.tvt_ft == 9 else "accepted"))


def test_native_dependency_and_bounds():
    result = crossing_graph()
    with patch.dict("sys.modules", {"network_engine": None}):
        with pytest.raises(RuntimeError, match="separately installed"):
            summarize_native_lift(result)
    for kwargs in ({"levels": True}, {"levels": 0}, {"pair_cap": -1}):
        with pytest.raises(ValueError):
            summarize_native_lift(result, **kwargs)


@pytest.mark.skipif(importlib.util.find_spec("network_engine") is None, reason="optional native provider")
def test_local_native_lift_of_only_admitted_crossings_preserves_search_status():
    result = crossing_graph()
    lifted = summarize_native_lift(result, levels=2)
    assert lifted["search_status"] == "inconclusive" and not lifted["search_complete"]
    # The admitted diamond is a four-cycle; its consecutive line graphs are four-cycles.
    assert lifted["levels"] == [{"nodes": 4, "edges": 4}]*3
    assert [e["observation_id"] for e in lifted["graph"]["edges"]] == ["0","0","1","1"]
    assert len(lifted["graph"]["node_ids"]) == 4
    json.dumps(lifted, allow_nan=False)
    capped = summarize_native_lift(result, pair_cap=0)
    assert capped["capped"] and capped["levels"] == lifted["levels"][:1]
    assert capped["graph_sha256"] == lifted["graph_sha256"]
