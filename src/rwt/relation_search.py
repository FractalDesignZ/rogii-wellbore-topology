"""Finite goal-directed wellbore traversal across an explicit observation boundary.

This is independent public control code, not a native Network Engine algorithm.
Boundary callbacks supply domain decisions; categorical verdicts are not calibrated
probabilities. Equal endpoints merge without discarding their relation provenance.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
import hashlib
import json
import math
from numbers import Integral
from types import MappingProxyType
from typing import Any, Literal

Outcome = Literal["accepted", "rejected", "unresolved"]


def _station(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError("station must be a nonnegative integer")
    return int(value)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(k, str) for k in value):
            raise ValueError("evidence/provenance keys must be strings")
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError("evidence/provenance must contain finite JSON values")


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        result = {f.name: _plain(getattr(value, f.name)) for f in fields(value)}
        if isinstance(value, CandidateState):
            if value.activity is None:
                result.pop("activity")
            result["node_id"] = value.node_id
        return result
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


@dataclass(frozen=True)
class CandidateState:
    well_id: str
    station: int
    tvt_ft: float
    layer: str = "placement"
    activity: str | None = None

    def __post_init__(self) -> None:
        if not self.well_id or not isinstance(self.well_id, str) or not self.layer or not isinstance(self.layer, str):
            raise ValueError("well_id and layer must be nonempty strings")
        if self.activity is not None and (not isinstance(self.activity, str) or not self.activity):
            raise ValueError("activity must be a nonempty state label when supplied")
        object.__setattr__(self, "station", _station(self.station))
        if isinstance(self.tvt_ft, bool) or not math.isfinite(float(self.tvt_ft)):
            raise ValueError("candidate TVT must be finite feet")
        object.__setattr__(self, "tvt_ft", float(self.tvt_ft) or 0.)

    @property
    def node_id(self) -> str:
        body = [self.well_id, self.layer, self.station, self.tvt_ft.hex()]
        if self.activity is not None:
            body.append(self.activity)
        return hashlib.sha256(json.dumps(body, separators=(",", ":")).encode()).hexdigest()

    @property
    def order_key(self) -> tuple[int, float, str]:
        return self.station, self.tvt_ft, self.activity or ""


@dataclass(frozen=True)
class ObservedRelation:
    observation_id: str
    well_id: str
    source_station: int
    target_station: int
    supported_values: Mapping[str, float]
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.observation_id, str) or not self.observation_id or not isinstance(self.well_id, str) or not self.well_id:
            raise ValueError("observation_id and well_id must be nonempty strings")
        object.__setattr__(self, "source_station", _station(self.source_station))
        object.__setattr__(self, "target_station", _station(self.target_station))
        if self.target_station <= self.source_station:
            raise ValueError("relations must advance station: this search is a DAG")
        values = {k: float(v) for k, v in self.supported_values.items()}
        if any(not isinstance(k, str) or not k or not math.isfinite(v) for k, v in values.items()):
            raise ValueError("supported values must be named finite readings; omit missing readings")
        object.__setattr__(self, "supported_values", _freeze(values))
        object.__setattr__(self, "provenance", _freeze(self.provenance))


@dataclass(frozen=True)
class BoundaryDecision:
    outcome: Outcome
    evidence: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.outcome not in ("accepted", "rejected", "unresolved"):
            raise ValueError("unknown boundary outcome")
        object.__setattr__(self, "evidence", _freeze(self.evidence))
        object.__setattr__(self, "provenance", _freeze(self.provenance))


Boundary = Callable[[ObservedRelation, CandidateState, CandidateState], BoundaryDecision]
RelationFilter = Callable[[ObservedRelation, CandidateState, CandidateState], bool]


def overlay_boundaries(overlays: Sequence[tuple[str, Boundary | None]] | Mapping[str, Boundary | None]) -> Boundary:
    """Overlay maps on the same edge; reject vetoes, all accept accepts, else unresolved.

    Every available callback runs. ``None`` records an unavailable map. Callback
    failures propagate instead of being converted into acceptance or omission.
    """
    maps = tuple(overlays.items() if isinstance(overlays, Mapping) else overlays)
    names = [name for name, _ in maps]
    if not maps or any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names):
        raise ValueError("overlay names must be nonempty and unique")
    if any(callback is not None and not callable(callback) for _, callback in maps):
        raise TypeError("overlay callbacks must be callable or explicitly unavailable")

    def combined(observed: ObservedRelation, source: CandidateState, proposed: CandidateState) -> BoundaryDecision:
        records = []
        for name, callback in maps:
            decision = (BoundaryDecision("unresolved", {"reason": "overlay_unavailable"}, {"map": name})
                        if callback is None else callback(observed, source, proposed))
            if not isinstance(decision, BoundaryDecision):
                raise TypeError("boundary must return BoundaryDecision")
            records.append({"name": name, **_plain(decision)})
        verdicts = {r["outcome"] for r in records}
        outcome = "rejected" if "rejected" in verdicts else "accepted" if verdicts == {"accepted"} else "unresolved"
        return BoundaryDecision(outcome, {"overlays": records, "disagreement": len(verdicts) > 1},
                                {"rule": "reject_if_any; accept_if_all; otherwise_unresolved"})

    return combined


@dataclass(frozen=True)
class SearchGoal:
    final_station: int
    tvt_interval_ft: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "final_station", _station(self.final_station))
        if self.tvt_interval_ft is not None:
            low, high = (float(v) for v in self.tvt_interval_ft)
            if not math.isfinite(low) or not math.isfinite(high) or low > high:
                raise ValueError("goal interval must be finite and ordered")
            object.__setattr__(self, "tvt_interval_ft", (low, high))

    def matches(self, state: CandidateState) -> bool:
        return state.station == self.final_station and (self.tvt_interval_ft is None or
               self.tvt_interval_ft[0] <= state.tvt_ft <= self.tvt_interval_ft[1])


@dataclass(frozen=True)
class SearchBudget:
    max_nodes: int = 10_000
    max_checks: int = 100_000
    max_boundary_calls: int = 100_000

    def __post_init__(self) -> None:
        for name in ("max_nodes", "max_checks", "max_boundary_calls"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < (1 if name == "max_nodes" else 0):
                raise ValueError("invalid finite search budget")


@dataclass(frozen=True)
class ParentLink:
    source_node_id: str
    observation_id: str


@dataclass(frozen=True)
class SearchNode:
    state: CandidateState
    parents: tuple[ParentLink, ...]
    path_count: int  # saturated at2; counts distinct observation-edge routes


@dataclass(frozen=True)
class TraversalStep:
    observation_id: str
    source_node_id: str
    proposed: CandidateState
    decision: BoundaryDecision
    admitted: bool
    reason: str


@dataclass(frozen=True)
class SearchResult:
    status: Literal["found", "ambiguous", "inconclusive", "exhausted"]
    complete: bool
    unique_value: bool
    unique_path: bool
    path_count: int
    goal_states: tuple[CandidateState, ...]
    nodes: tuple[SearchNode, ...]
    observations: tuple[ObservedRelation, ...]
    steps: tuple[TraversalStep, ...]
    incomplete_reasons: tuple[str, ...]
    checks: int
    boundary_calls: int

    @property
    def value_ft(self) -> float | None:
        return self.goal_states[0].tvt_ft if self.unique_value else None

    def to_dict(self) -> dict[str, Any]:
        return {**_plain(self), "value_ft": self.value_ft}

    def supporting_steps(self, goal_state: CandidateState | None = None) -> tuple[TraversalStep, ...]:
        """Trace accepted routes to reached goals, retaining their map evidence.

        Select a particular reachable terminal state to narrow the trace.
        Unresolved alternatives remain recorded in the full search result.
        """
        if goal_state is not None and goal_state not in self.goal_states:
            raise ValueError("the selected state was not reached at the goal")
        targets = self.goal_states if goal_state is None else (goal_state,)
        ancestors = {state.node_id for state in targets}
        support = []
        # The forward traversal processes earlier source stations first.
        for step in reversed(self.steps):
            if step.admitted and step.proposed.node_id in ancestors:
                ancestors.add(step.source_node_id)
                support.append(step)
        return tuple(reversed(support))


def search_relations(anchor: CandidateState, candidates_by_station: Mapping[int, Sequence[CandidateState]],
                     observations: Sequence[ObservedRelation], *, goal: SearchGoal, boundary: Boundary,
                     relation_filters: Sequence[tuple[str, RelationFilter]] = (),
                     budget: SearchBudget = SearchBudget()) -> SearchResult:
    """Search one well/layer completely enough to prove a value/path conclusion.

    Exact equal endpoints merge; no implicit numeric tolerance changes a value.
    An unresolved relevant edge or resource cutoff cannot produce ``found``.
    Accepted distinct goal values establish ambiguity even if other work remains.
    Filters and boundaries must be pure callbacks; exceptions fail the call.
    """
    if goal.final_station < anchor.station or not callable(boundary):
        raise ValueError("goal must not precede anchor and a boundary is required")
    candidates = {}
    for station, states in candidates_by_station.items():
        station = _station(station)
        unique = set(states)
        if any(not isinstance(s, CandidateState) or s.station != station or s.well_id != anchor.well_id or s.layer != anchor.layer for s in unique):
            raise ValueError("candidate well/station/layer scope mismatch")
        candidates[station] = tuple(sorted(unique, key=lambda s: s.order_key))
    filters = tuple(relation_filters)
    if len({n for n, _ in filters}) != len(filters) or any(not isinstance(n, str) or not n or not callable(f) for n, f in filters):
        raise ValueError("relation filters must have unique names and callable predicates")
    outgoing = defaultdict(list)
    ids = set()
    for observed in observations:
        if not isinstance(observed, ObservedRelation) or observed.well_id != anchor.well_id or observed.observation_id in ids:
            raise ValueError("observation well/identity scope mismatch")
        ids.add(observed.observation_id)
        outgoing[observed.source_station].append(observed)
    for values in outgoing.values():
        values.sort(key=lambda r: (r.target_station, r.observation_id))
    # Conservative potential reachability, without treating an unresolved edge as accepted.
    relevant = {goal.final_station} if any(goal.matches(s) for s in candidates.get(goal.final_station, ())) else set()
    if goal.matches(anchor):
        relevant.add(anchor.station)
    for station in sorted(outgoing, reverse=True):
        if any(r.target_station in relevant and candidates.get(r.target_station) for r in outgoing[station]):
            relevant.add(station)
    records = {anchor: {"parents": [], "count": 1}}
    at_station = defaultdict(set); at_station[anchor.station].add(anchor)
    steps, reasons = [], set()
    checks = calls = 0
    stop = False
    for station in sorted({anchor.station, *candidates}):
        if stop or station not in relevant or station >= goal.final_station:
            continue
        for source in sorted(at_station[station], key=lambda s: s.order_key):
            if stop: break
            for observed in outgoing[station]:
                if stop: break
                if observed.target_station not in relevant: continue
                for proposed in candidates.get(observed.target_station, ()):
                    if proposed.station == goal.final_station and not goal.matches(proposed): continue
                    if checks >= budget.max_checks:
                        reasons.add("max_checks"); stop = True; break
                    checks += 1
                    rejected_by = None
                    for name, predicate in filters:
                        verdict = predicate(observed, source, proposed)
                        if not isinstance(verdict, bool): raise TypeError("relation filter must return bool")
                        if not verdict: rejected_by = name; break
                    if rejected_by is not None:
                        decision = BoundaryDecision("rejected", {"filter": rejected_by}, {"kind": "relation_filter"})
                        steps.append(TraversalStep(observed.observation_id, source.node_id, proposed, decision, False, "relation_filter"))
                        continue
                    if calls >= budget.max_boundary_calls:
                        reasons.add("max_boundary_calls"); stop = True; break
                    calls += 1
                    decision = boundary(observed, source, proposed)
                    if not isinstance(decision, BoundaryDecision): raise TypeError("boundary must return BoundaryDecision")
                    admitted = False; reason = decision.outcome
                    if decision.outcome == "unresolved": reasons.add("unresolved_boundary")
                    if decision.outcome == "accepted":
                        if proposed not in records and len(records) >= budget.max_nodes:
                            reasons.add("max_nodes"); reason = "max_nodes"
                        else:
                            record = records.setdefault(proposed, {"parents": [], "count": 0})
                            link = ParentLink(source.node_id, observed.observation_id)
                            if link not in record["parents"]:
                                record["parents"].append(link)
                                record["count"] = min(2, record["count"] + records[source]["count"])
                            at_station[proposed.station].add(proposed); admitted = True
                    steps.append(TraversalStep(observed.observation_id, source.node_id, proposed, decision, admitted, reason))
    goals = tuple(sorted((s for s in records if goal.matches(s)), key=lambda s: s.order_key))
    depths = {state.tvt_ft for state in goals}
    complete = not reasons
    if len(depths) > 1:
        status = "ambiguous"
    elif not complete:
        status = "inconclusive"
    elif depths:
        status = "found"
    else:
        status = "exhausted"
    unique = status == "found"
    count = min(2, sum(records[s]["count"] for s in goals))
    nodes = tuple(SearchNode(s, tuple(records[s]["parents"]), records[s]["count"])
                  for s in sorted(records, key=lambda s: s.order_key))
    return SearchResult(status, complete, unique, unique and count == 1, count, goals, nodes,
                        tuple(sorted(observations, key=lambda r: r.observation_id)), tuple(steps), tuple(sorted(reasons)), checks, calls)
