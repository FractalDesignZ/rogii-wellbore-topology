"""Run a three-candidate TVT query with two quadratic maps on the same edges."""
import argparse
import cmath
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rwt.relation_search import (
    BoundaryDecision, CandidateState, ObservedRelation, SearchGoal,
    overlay_boundaries, search_relations,
)


def classify_orbit(start, parameter, reading):
    z = complex(start)
    outcome = "unresolved"
    for step in range(25):
        if abs(z) > 2:
            outcome = "rejected"
            break
        if abs(z) <= 0.25:
            outcome = "accepted"
            break
        if step < 24:
            z = z * z + parameter
    evidence = {"start": [start.real, start.imag], "step": step, "last": [z.real, z.imag]}
    return BoundaryDecision(outcome, evidence, {"observation": reading, "c": parameter})


def quadratic_view(center_key, scale_ft, parameter):
    """Use the normalized discrepancy as the orbit's initial coordinate."""
    def decide(observed, source, proposed):
        center = observed.supported_values.get(center_key)
        if center is None:
            return BoundaryDecision("unresolved", {"missing": center_key})
        start = (proposed.tvt_ft - source.tvt_ft - center) / scale_ft
        return classify_orbit(start, parameter, center_key)
    return decide


def periodic_view(period_ft, parameter):
    """Encode disagreement with a synthetic periodic reading in two coordinates."""
    reading = f"phase_{period_ft}_turns"

    def decide(observed, source, proposed):
        measured = observed.supported_values.get(reading)
        if measured is None:
            return BoundaryDecision("unresolved", {"missing": reading})
        turns = ((proposed.tvt_ft - source.tvt_ft) % period_ft) / period_ft
        predicted = cmath.exp(2j * math.pi * turns)
        measured = cmath.exp(2j * math.pi * measured)
        return classify_orbit(0.5 + 3 * (predicted - measured), parameter, reading)
    return decide


def phase_stack():
    """Return the complete searches for three synthetic sensors and a repeat control."""
    anchor = CandidateState("periodic", 0, 1000)
    candidates = {1: [CandidateState("periodic", 1, 1000 + offset) for offset in range(30)]}
    observed = ObservedRelation("periodic-readings", "periodic", 0, 1,
                                {"phase_2_turns": 1/2, "phase_3_turns": 2/3, "phase_5_turns": 2/5},
                                {"source": "constructed periodic sensors"})
    views = [("2 ft", periodic_view(2, 0)), ("3 ft", periodic_view(3, -1)), ("5 ft", periodic_view(5, 0))]
    arms = [("2 ft", views[:1]), ("2 + 3 ft", views[:2]), ("2 + 3 + 5 ft", views),
            ("2 ft repeated three times", [(str(i), views[0][1]) for i in range(3)])]
    return {name: search_relations(anchor, candidates, [observed], goal=SearchGoal(1),
                                   boundary=overlay_boundaries(stack)) for name, stack in arms}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-lift", action="store_true", help="measure shared endpoints using the local Network Engine")
    parser.add_argument("--phase-stack", action="store_true", help="also compare complementary and repeated periodic readings")
    args = parser.parse_args()
    anchor = CandidateState("demo", 0, 1000)
    candidates = {1: [CandidateState("demo", 1, tvt) for tvt in (996, 1000, 1004)]}
    observed = ObservedRelation("two-readings", "demo", 0, 1,
                                {"first_change_ft": -2, "second_change_ft": 2},
                                {"source": "constructed example"})
    views = {"first": quadratic_view("first_change_ft", 4, 0),
             "second": quadratic_view("second_change_ft", 2, -1)}
    for name, boundary in [*views.items(), ("both", overlay_boundaries(views))]:
        result = search_relations(anchor, candidates, [observed],
                                  goal=SearchGoal(1), boundary=boundary)
        depths = [state.tvt_ft for state in result.goal_states]
        print(f"{name:6} {result.status:10} reachable TVT (ft) {depths}")
        if name == "both":
            for crossing in result.steps:
                votes = crossing.decision.evidence["overlays"]
                details = ", ".join(f"{v['name']} {v['outcome']} at step {v['evidence']['step']}"
                                    for v in votes)
                print(f"  {crossing.proposed.tvt_ft:.0f} ft   {details}")
        if args.native_lift:
            from rwt.relation_lift import summarize_native_lift
            lifted = summarize_native_lift(result)
            print("  Native nodes by level", [level["nodes"] for level in lifted["levels"]])
    if args.phase_stack:
        print("\nPeriodic observations on the same 30 candidate depths")
        for name, result in phase_stack().items():
            print(f"  {name:26} {len(result.goal_states):2} reachable   {result.status}"
                  + (f"   {result.value_ft:g} ft" if result.unique_value else ""))


if __name__ == "__main__":
    main()
