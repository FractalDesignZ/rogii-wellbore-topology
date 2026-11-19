"""Run a three-candidate TVT query with two quadratic maps on the same edges."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rwt.relation_search import (
    BoundaryDecision, CandidateState, ObservedRelation, SearchGoal,
    overlay_boundaries, search_relations,
)


def quadratic_view(center_key, scale_ft, parameter):
    """Use the normalized discrepancy as the orbit's initial coordinate."""
    def decide(observed, source, proposed):
        center = observed.supported_values.get(center_key)
        if center is None:
            return BoundaryDecision("unresolved", {"missing": center_key})
        start = (proposed.tvt_ft - source.tvt_ft - center) / scale_ft
        z = start
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
        return BoundaryDecision(outcome, {"start": start, "step": step, "last": z},
                                {"observation": center_key, "c": parameter})
    return decide


def main():
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


if __name__ == "__main__":
    main()
