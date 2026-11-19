"""Finite graph, observation boundary, ambiguity and overlay falsifiers."""
from pathlib import Path
import json
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rwt.relation_search import (
    BoundaryDecision, CandidateState, ObservedRelation, SearchBudget, SearchGoal,
    overlay_boundaries, search_relations,
)


def state(station, value=100., **kwargs):
    return CandidateState(kwargs.pop("well_id", "w"), station, value, **kwargs)


def observation(source, target, name=None, **kwargs):
    return ObservedRelation(name or f"o{source}_{target}", kwargs.pop("well_id", "w"),
                            source, target, {"measured_delta": 1.}, **kwargs)


def accept(observed, source, proposed):
    return BoundaryDecision("accepted", {"observation_id": observed.observation_id}, {"provider": "fixture"})


class RelationSearchTests(unittest.TestCase):
    def test_condition_can_distinguish_opposite_activities_at_equal_depth(self):
        anchor = state(0)
        rising = state(2, activity="increasing TVT")
        falling = state(2, activity="decreasing TVT")
        candidates = {1: [state(1,99), state(1,101)], 2: [rising, falling],
                      3: [state(3,99), state(3,101)]}
        obs = [observation(0,1), observation(1,2),
               ObservedRelation("final", "w", 2, 3, {"measured_sign": 1})]

        def motion(observed, source, proposed):
            increasing = proposed.tvt_ft > source.tvt_ft
            if proposed.station == 2:
                compatible = (proposed.activity == "increasing TVT") == increasing
            elif proposed.station == 3:
                compatible = increasing and source.activity == "increasing TVT" and observed.supported_values["measured_sign"] > 0
            else:
                compatible = True
            return BoundaryDecision("accepted" if compatible else "rejected")

        middle = search_relations(anchor, candidates, obs, goal=SearchGoal(2), boundary=motion)
        self.assertEqual((middle.value_ft, middle.path_count, middle.unique_path), (100.,2,False))
        self.assertEqual(set(middle.goal_states), {rising, falling})
        self.assertNotEqual(rising.node_id, falling.node_id)
        final = search_relations(anchor, candidates, obs, goal=SearchGoal(3), boundary=motion)
        self.assertEqual((final.value_ft,final.unique_path), (101.,True))
        self.assertEqual(final.nodes[-1].parents[0].source_node_id, rising.node_id)
        for invalid in ("", 1):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                state(2, activity=invalid)

    def test_observed_boundary_and_unique_value_path(self):
        seen = []
        def boundary(observed, source, proposed):
            seen.append((observed.observation_id, dict(observed.supported_values), source.station, proposed.station))
            return accept(observed, source, proposed)
        result = search_relations(state(0), {1: [state(1)], 2: [state(2, 101.)]},
                                  [observation(0, 1), observation(1, 2)],
                                  goal=SearchGoal(2, (100.5, 101.5)), boundary=boundary)
        self.assertEqual(result.status, "found")
        self.assertTrue(result.complete and result.unique_value and result.unique_path)
        self.assertEqual((result.value_ft, result.path_count, len(seen)), (101., 1, 2))
        self.assertEqual(seen[0], ("o0_1", {"measured_delta": 1.}, 0, 1))
        json.dumps(result.to_dict(), allow_nan=False)

    def test_equal_endpoint_merges_parents_but_retains_path_ambiguity(self):
        result = search_relations(state(0), {1: [state(1, 99.), state(1, 101.)], 2: [state(2), state(2)]},
                                  [observation(0, 1), observation(1, 2)], goal=SearchGoal(2), boundary=accept)
        self.assertEqual(result.status, "found")
        self.assertTrue(result.unique_value)
        self.assertFalse(result.unique_path)
        self.assertEqual(result.path_count, 2)
        target = next(n for n in result.nodes if n.state.station == 2)
        self.assertEqual(len(target.parents), 2)
        self.assertEqual(len(result.nodes), 4)

    def test_saturated_path_counts_do_not_expand_complete_paths(self):
        candidates = {i: [state(i, 99.), state(i, 101.)] for i in range(1, 8)}
        candidates[8] = [state(8)]
        result = search_relations(state(0), candidates, [observation(i, i+1) for i in range(8)],
                                  goal=SearchGoal(8), boundary=accept)
        self.assertEqual((result.status, result.path_count, len(result.nodes)), ("found", 2, 16))
        self.assertTrue(all(n.path_count <= 2 for n in result.nodes))

    def test_distinct_accepted_goal_values_are_ambiguous(self):
        result = search_relations(state(0), {1: [state(1, 99.), state(1, 101.)]},
                                  [observation(0, 1)], goal=SearchGoal(1), boundary=accept)
        self.assertEqual(result.status, "ambiguous")
        self.assertFalse(result.unique_value or result.unique_path)
        self.assertIsNone(result.value_ft)

    def test_unresolved_relevant_branch_prevents_false_found(self):
        def boundary(observed, source, proposed):
            return BoundaryDecision("unresolved" if proposed.station == 1 and proposed.tvt_ft == 101. else "accepted",
                                    {"supported": proposed.tvt_ft != 101.})
        result = search_relations(state(0), {1: [state(1), state(1, 101.)], 2: [state(2)]},
                                  [observation(0, 1), observation(1, 2)], goal=SearchGoal(2), boundary=boundary)
        self.assertEqual(result.status, "inconclusive")
        self.assertEqual(len(result.goal_states), 1)
        self.assertFalse(result.complete or result.unique_value)
        self.assertIn("unresolved_boundary", result.incomplete_reasons)

    def test_irrelevant_dead_end_does_not_block_goal(self):
        def boundary(observed, source, proposed):
            if proposed.station == 3: raise AssertionError("irrelevant edge evaluated")
            return accept(observed, source, proposed)
        result = search_relations(state(0), {1: [state(1)], 3: [state(3)]},
                                  [observation(0, 1), observation(0, 3)], goal=SearchGoal(1), boundary=boundary)
        self.assertEqual(result.status, "found")
        self.assertEqual(result.boundary_calls, 1)

    def test_exhaustion_and_goal_interval(self):
        result = search_relations(state(0), {1: [state(1)]}, [observation(0, 1)], goal=SearchGoal(1),
                                  boundary=lambda *args: BoundaryDecision("rejected", {"reason": "relation mismatch"}))
        self.assertEqual((result.status, result.complete), ("exhausted", True))
        outside = search_relations(state(0), {1: [state(1)]}, [observation(0, 1)],
                                   goal=SearchGoal(1, (105., 110.)), boundary=accept)
        self.assertEqual(outside.status, "exhausted")
        self.assertEqual(outside.boundary_calls, 0)

    def test_each_budget_cutoff_is_inconclusive_even_after_one_goal(self):
        for budget, reason in ((SearchBudget(max_checks=1), "max_checks"),
                               (SearchBudget(max_boundary_calls=1), "max_boundary_calls"),
                               (SearchBudget(max_nodes=1), "max_nodes")):
            with self.subTest(reason=reason):
                result = search_relations(state(0), {1: [state(1), state(1, 101.)]}, [observation(0, 1)],
                                          goal=SearchGoal(1), boundary=accept, budget=budget)
                self.assertEqual(result.status, "inconclusive")
                self.assertFalse(result.unique_value)
                self.assertIn(reason, result.incomplete_reasons)

    def test_named_relation_filter_blocks_before_boundary(self):
        def forbidden(*args): raise AssertionError("boundary should not run")
        result = search_relations(state(0), {1: [state(1)]}, [observation(0, 1)], goal=SearchGoal(1),
                                  boundary=forbidden, relation_filters=(("observed_support", lambda *args: False),))
        self.assertEqual(result.status, "exhausted")
        self.assertEqual(result.boundary_calls, 0)
        self.assertEqual(result.steps[0].decision.evidence["filter"], "observed_support")

    def test_well_layer_station_and_observation_identity_guards(self):
        for candidates, observations in (({1: [state(1, well_id="other")]}, [observation(0, 1)]),
                                         ({1: [state(1, layer="other")]}, [observation(0, 1)]),
                                         ({1: [state(2)]}, [observation(0, 1)]),
                                         ({1: [state(1)]}, [observation(0, 1, well_id="other")]),
                                         ({1: [state(1)]}, [observation(0, 1), observation(0, 1)])):
            with self.subTest(candidates=candidates):
                with self.assertRaises(ValueError):
                    search_relations(state(0), candidates, observations, goal=SearchGoal(1), boundary=accept)
        for call in (lambda: observation(1, 1), lambda: state(-1), lambda: state(0, float("nan")),
                     lambda: ObservedRelation("bad", "w", 0, 1, {"missing": float("nan")})):
            with self.assertRaises(ValueError): call()

    def test_boundary_errors_are_not_acceptance(self):
        with self.assertRaises(TypeError):
            search_relations(state(0), {1: [state(1)]}, [observation(0, 1)], goal=SearchGoal(1), boundary=lambda *args: True)
        with self.assertRaises(ValueError): BoundaryDecision("accepted", {"score": float("inf")})

    def test_overlay_all_callbacks_same_objects_veto_and_missing(self):
        observed, source, target = observation(0, 1), state(0), state(1)
        called = []
        def callback(name, verdict):
            def run(o, s, p):
                self.assertIs(o, observed); self.assertIs(s, source); self.assertIs(p, target)
                called.append(name)
                return BoundaryDecision(verdict, {"measurement": name}, {"provider": name})
            return run
        maps = (("veto", callback("veto", "rejected")), ("accept", callback("accept", "accepted")), ("missing", None))
        result = overlay_boundaries(maps)(observed, source, target)
        self.assertEqual(result.outcome, "rejected")
        self.assertEqual(called, ["veto", "accept"])
        self.assertEqual([r["outcome"] for r in result.evidence["overlays"]], ["rejected", "accepted", "unresolved"])
        self.assertTrue(result.evidence["disagreement"])
        self.assertEqual(overlay_boundaries(tuple(reversed(maps)))(observed, source, target).outcome, result.outcome)
        self.assertEqual(overlay_boundaries((("ok", accept), ("missing", None)))(observed, source, target).outcome, "unresolved")

    def test_overlay_keeps_graph_nodes_edges_and_path_counts(self):
        arguments = (state(0), {1: [state(1)], 2: [state(2)]}, [observation(0, 1), observation(1, 2)])
        single = search_relations(*arguments, goal=SearchGoal(2), boundary=accept)
        overlaid = search_relations(*arguments, goal=SearchGoal(2), boundary=overlay_boundaries({"map_a": accept, "map_b": accept}))
        self.assertEqual(single.nodes, overlaid.nodes)
        self.assertEqual(single.path_count, overlaid.path_count)
        self.assertEqual([(s.observation_id, s.source_node_id, s.proposed.node_id) for s in single.steps],
                         [(s.observation_id, s.source_node_id, s.proposed.node_id) for s in overlaid.steps])
        self.assertEqual(len(overlaid.steps[0].decision.evidence["overlays"]), 2)

    def test_observation_and_overlay_evidence_are_immutable_serializable(self):
        values = {"GR": 12.}; provenance = {"source": ["input", 3]}
        observed = ObservedRelation("id", "w", 0, 1, values, provenance)
        values["GR"] = 999.; provenance["source"][0] = "changed"
        self.assertEqual(observed.supported_values["GR"], 12.)
        self.assertEqual(observed.provenance["source"][0], "input")
        with self.assertRaises(TypeError): observed.supported_values["GR"] = 0.
        with self.assertRaises(ValueError): overlay_boundaries((("same", accept), ("same", accept)))

    def test_known_anchor_already_at_goal_needs_no_invented_relation(self):
        result = search_relations(state(2), {}, [], goal=SearchGoal(2), boundary=accept, budget=SearchBudget(max_checks=0))
        self.assertEqual((result.status, result.path_count, result.boundary_calls), ("found", 1, 0))


if __name__ == "__main__":
    unittest.main()
