"""Compose declared policy factors and paired held-well merit on an evidence DAG."""

from __future__ import annotations

from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from hashlib import sha256
import json
import math
from numbers import Real
from typing import Iterable
from unicodedata import normalize


@dataclass(frozen=True)
class WeightScope:
    scope_id: str
    calibration_wells: tuple[str, ...]
    query_wells: tuple[str, ...]


@dataclass(frozen=True)
class Footprint:
    count: int
    sha256: str  # Digest of the ordered native observation keys used in this loss.


@dataclass(frozen=True)
class PairedWellLoss:
    well_id: str
    candidate: float
    baseline: float
    candidate_footprint: Footprint
    baseline_footprint: Footprint


@dataclass(frozen=True)
class Given:
    """A supported source condition or a composition of supported parents."""


@dataclass(frozen=True)
class FixedWeight:
    factor_id: str
    value: float


@dataclass(frozen=True)
class HeldMerit:
    factor_id: str
    losses: tuple[PairedWellLoss, ...]
    minimum_wells: int = 8


@dataclass(frozen=True)
class WeightNode:
    node_id: str
    scope_id: str
    source_id: str
    evidence_id: str
    rule: Given | FixedWeight | HeldMerit
    parents: tuple[str, ...] = ()
    supported: bool = True


@dataclass(frozen=True)
class WeightDecision:
    node_id: str
    source_id: str
    evidence_id: str
    scope_id: str
    local_weight: float
    weight: float
    reason: str
    factors: tuple[str, ...]
    parent_hashes: tuple[tuple[str, str], ...]
    sha256: str

    @property
    def enabled(self) -> bool:
        return self.weight > 0.0


@dataclass(frozen=True)
class WeightGraph:
    decisions: tuple[WeightDecision, ...]
    scope_sha256: str
    sha256: str

    def decision(self, node_id: str) -> WeightDecision:
        for decision in self.decisions:
            if decision.node_id == node_id:
                return decision
        raise KeyError(node_id)


def _digest(value) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(data.encode("utf-8")).hexdigest()


def _key(value: str) -> str:
    if not isinstance(value, str) or not value or any(c.isspace() or not c.isprintable() for c in value):
        raise ValueError("identifiers must be nonempty and free of whitespace and control characters")
    if value != normalize('NFC', value):
        raise ValueError("identifiers must use canonical Unicode NFC spelling")
    return value


def _unique(keys: Iterable[str]) -> tuple[str, ...]:
    keys = tuple(_key(k) for k in keys)
    if len({k.casefold() for k in keys}) != len(keys):
        raise ValueError("duplicate identifier or case alias")
    return keys


def _number(value: Real) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError("expected a finite nonnegative number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError("number exceeds the finite range") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError("expected a finite nonnegative number")
    return result


def _merit(rule: HeldMerit, allowed: set[str]) -> tuple[float, str, str]:
    if type(rule.minimum_wells) is not int or rule.minimum_wells < 1:
        raise ValueError("minimum_wells must be a positive study parameter")
    if not isinstance(rule.losses, tuple):
        raise ValueError("losses must be a tuple in recorded evaluation order")
    if not all(isinstance(row, PairedWellLoss) for row in rule.losses):
        raise ValueError("expected one paired loss record per calibration well")
    _unique(row.well_id for row in rule.losses)
    candidate_sum = baseline_sum = 0.0
    evidence = []
    for row in rule.losses:
        if row.well_id not in allowed:
            raise ValueError("loss well is outside the permitted calibration bank")
        fp = row.candidate_footprint
        if (not isinstance(fp, Footprint) or not isinstance(row.baseline_footprint, Footprint)
                or type(row.baseline_footprint.count) is not int or fp != row.baseline_footprint):
            raise ValueError("paired losses have different observation footprints")
        if (type(fp.count) is not int or fp.count < 1 or not isinstance(fp.sha256, str)
                or len(fp.sha256) != 64 or any(c not in '0123456789abcdef' for c in fp.sha256)):
            raise ValueError("expected nonempty footprint and lowercase SHA-256")
        candidate, baseline = _number(row.candidate), _number(row.baseline)
        candidate_sum += candidate
        baseline_sum += baseline
        evidence.append((row.well_id, candidate, baseline, fp.count, fp.sha256))
    proof = _digest(["held_merit", rule.minimum_wells, evidence])
    if evidence and {row.well_id for row in rule.losses} != allowed:
        raise ValueError("paired losses must cover the complete declared calibration bank")
    if len(evidence) < rule.minimum_wells:
        return 0.0, "insufficient_wells", proof
    if not math.isfinite(candidate_sum) or not math.isfinite(baseline_sum):
        scale = max(max(row[1], row[2]) for row in evidence)
        candidate_sum = math.fsum(row[1] / scale for row in evidence)
        baseline_sum = math.fsum(row[2] / scale for row in evidence)
    if baseline_sum == 0:
        return 0.0, "zero_baseline_loss", proof
    if candidate_sum >= baseline_sum:
        return 0.0, "no_positive_merit", proof
    return 1.0 - candidate_sum / baseline_sum, "held_merit", proof


def resolve_weights(scope: WeightScope, nodes: Iterable[WeightNode]) -> WeightGraph:
    """Return an auditable product of unique ancestor factors for every policy.

    Supply mean losses per held calibration well in the original evaluation
    order. The caller binds how these held predictions and key digests were
    produced to the evidence ID. A supplied loss bank must cover every declared
    calibration well. An entirely absent bank switches the corresponding node
    off. An unsupported or zero-weight parent turns its descendants off.
    This calculation proposes action weights. Measured
    changes to a fixed prediction objective have a separate acceptance rule.
    """
    _key(scope.scope_id)
    if not isinstance(scope.calibration_wells, tuple) or not isinstance(scope.query_wells, tuple):
        raise ValueError("role lists must be tuples of well identifiers")
    _unique(scope.calibration_wells + scope.query_wells)
    permitted = set(scope.calibration_wells)
    scope_hash = _digest([scope.scope_id, sorted(permitted), sorted(scope.query_wells)])
    records = tuple(nodes)
    if not records:
        raise ValueError("weight graph must contain a node")
    _unique(n.node_id for n in records)
    by_id = {n.node_id: n for n in records}
    edges = {}
    sources: dict[str, str] = {}
    evidence: dict[str, str] = {}
    for node in records:
        if node.scope_id != scope.scope_id or type(node.supported) is not bool:
            raise ValueError("invalid node scope or support flag")
        for key, registry in ((node.source_id, sources), (node.evidence_id, evidence)):
            if registry.setdefault(_key(key).casefold(), key) != key:
                raise ValueError("source or evidence identifier case alias")
        if not isinstance(node.parents, tuple):
            raise ValueError("parents must be a tuple of node identifiers")
        edges[node.node_id] = sorted(_unique(node.parents))
        if any(p not in by_id for p in node.parents):
            raise ValueError("missing parent or noncanonical parent identifier")
    try:
        order = tuple(TopologicalSorter(edges).static_order())
    except CycleError as exc:
        raise ValueError("cycle in weight policies") from exc

    decisions: dict[str, WeightDecision] = {}
    factors: dict[str, tuple[str, float]] = {}
    claims: dict[tuple[str, str, str], str] = {}
    spellings: dict[str, str] = {}
    for name in order:
        node = by_id[name]
        parents = [decisions[p] for p in edges[name]]
        ancestors = {f for parent in parents for f in parent.factors}
        rule = node.rule
        if isinstance(rule, Given):
            local, reason, proof = 1.0, "given", _digest("given")
        else:
            if isinstance(rule, HeldMerit):
                local, reason, proof = _merit(rule, permitted)
            elif isinstance(rule, FixedWeight):
                local = _number(rule.value)
                if local > 1:
                    raise ValueError("fixed weight must be between zero and one")
                reason, proof = "fixed", _digest(["fixed", local])
            else:
                raise ValueError("unknown weight rule")
            fid = _key(rule.factor_id)
            if spellings.setdefault(fid.casefold(), fid) != fid:
                raise ValueError("factor identifier case alias")
            claim = (node.source_id, node.evidence_id, type(rule).__name__)
            if claims.setdefault(claim, fid) != fid:
                raise ValueError("one factor evidence claim has two identifiers")
            bound = (_digest([claim, proof]), local)
            if factors.setdefault(fid, bound) != bound:
                raise ValueError("factor identifier has conflicting evidence")
            ancestors.add(fid)
        ids = tuple(sorted(ancestors))
        parent_hashes = tuple((p.node_id, p.sha256) for p in parents)
        if not node.supported:
            local, weight, reason = 0.0, 0.0, "unsupported"
        elif any(not p.enabled for p in parents):
            weight, reason = 0.0, "parent_off"
        else:
            weight = math.prod((factors[f][1] for f in ids), start=1.0)
        local_factor = None if isinstance(rule, Given) else rule.factor_id
        digest = _digest([scope_hash, name, node.source_id, node.evidence_id, local_factor, proof,
                          node.supported, local, weight, reason, ids, parent_hashes])
        decisions[name] = WeightDecision(name, node.source_id, node.evidence_id, scope.scope_id,
                                         local, weight, reason, ids, parent_hashes, digest)
    resolved = tuple(decisions[k] for k in sorted(decisions))
    return WeightGraph(resolved, scope_hash, _digest([scope_hash, [(d.node_id, d.sha256) for d in resolved]]))
