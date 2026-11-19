"""Trace declared hidden-label ancestry through first-party evidence artifacts.

The caller supplies the direct label-well IDs, all parent edges and an evidence
digest for each artifact. These receipts bind those declarations. They do not
observe file reads or establish that the declarations are complete.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
import re

from .execution_graph import _digest
from .merit_weights import _key, _unique


def _ordered_ids(ids: Iterable[str]) -> tuple[str, ...]:
    if isinstance(ids, (str, bytes)):
        raise ValueError("expected a collection of identifiers")
    return tuple(sorted(_unique(ids)))


def _check_spellings(ids: Iterable[str], seen: dict[str, str]) -> None:
    for name in ids:
        previous = seen.setdefault(name.casefold(), name)
        if previous != name:
            raise ValueError(f"case alias for {previous!r}: {name!r}")


@dataclass(frozen=True)
class LabelArtifact:
    """A declaration with an externally supplied source or execution evidence hash."""

    artifact_id: str
    evidence_sha256: str
    parents: tuple[str, ...] = ()
    direct_label_wells: tuple[str, ...] = ()

    def __post_init__(self):
        _key(self.artifact_id)
        if not isinstance(self.evidence_sha256, str) or not re.fullmatch("[0-9a-f]{64}", self.evidence_sha256):
            raise ValueError("expected a lowercase SHA-256 evidence digest")
        object.__setattr__(self, "parents", _ordered_ids(self.parents))
        object.__setattr__(self, "direct_label_wells", _ordered_ids(self.direct_label_wells))


@dataclass(frozen=True)
class LabelDecision:
    artifact_id: str
    evidence_sha256: str
    parent_hashes: tuple[tuple[str, str], ...]
    direct_label_wells: tuple[str, ...]
    label_wells: tuple[str, ...]
    sha256: str


@dataclass(frozen=True)
class ExclusionProof:
    artifact_id: str
    graph_sha256: str
    decision_sha256: str
    held_wells: tuple[str, ...]
    overlap: tuple[str, ...]
    sha256: str


class LabelOverlapError(ValueError):
    """The selected artifact has a declared label ancestor in the requested holdout."""

    def __init__(self, proof: ExclusionProof):
        self.proof = proof
        super().__init__(f"{proof.artifact_id!r} uses held label wells {proof.overlap!r}")


@dataclass(frozen=True)
class LabelLineage:
    decisions: tuple[LabelDecision, ...]
    known_label_wells: tuple[str, ...]
    sha256: str

    def decision(self, artifact_id: str) -> LabelDecision:
        _key(artifact_id)
        for decision in self.decisions:
            if decision.artifact_id == artifact_id:
                return decision
        raise KeyError(artifact_id)

    def require_exclusion(self, artifact_id: str, held_wells: Iterable[str]) -> ExclusionProof:
        """Require the selected artifact's transitive hidden-label set to exclude the holdout."""
        decision = self.decision(artifact_id)
        held = _ordered_ids(held_wells)
        _check_spellings(held, {w.casefold(): w for w in self.known_label_wells})
        overlap = tuple(sorted(set(held).intersection(decision.label_wells)))
        digest = _digest(["rwt.label-exclusion.v1", self.sha256, artifact_id, decision.sha256, held, overlap])
        proof = ExclusionProof(artifact_id, self.sha256, decision.sha256, held, overlap, digest)
        if overlap:
            raise LabelOverlapError(proof)
        return proof


def build_label_lineage(artifacts: Iterable[LabelArtifact]) -> LabelLineage:
    """Build deterministic receipts over direct and inherited hidden-label declarations."""
    by_id = {}
    well_spellings: dict[str, str] = {}
    for artifact in artifacts:
        if not isinstance(artifact, LabelArtifact):
            raise ValueError("expected a LabelArtifact declaration")
        if artifact.artifact_id in by_id:
            raise ValueError(f"duplicate artifact {artifact.artifact_id!r}")
        by_id[artifact.artifact_id] = artifact
        _check_spellings(artifact.direct_label_wells, well_spellings)
    if not by_id:
        raise ValueError("declare at least one artifact")
    ordered = _ordered_ids(by_id)
    dependencies = {name: by_id[name].parents for name in ordered}
    for name, parents in dependencies.items():
        for parent in parents:
            if parent not in by_id:
                raise ValueError(f"unknown or noncanonical parent {parent!r} of {name!r}")
    try:
        order = tuple(TopologicalSorter(dependencies).static_order())
    except CycleError as exc:
        raise ValueError("cycle in label lineage") from exc

    decisions: dict[str, LabelDecision] = {}
    for name in order:
        artifact = by_id[name]
        wells = set(artifact.direct_label_wells)
        for parent in artifact.parents:
            wells.update(decisions[parent].label_wells)
        cumulative = tuple(sorted(wells))
        parents = tuple((parent, decisions[parent].sha256) for parent in artifact.parents)
        digest = _digest(["rwt.label-decision.v1", name, artifact.evidence_sha256,
                          parents, artifact.direct_label_wells, cumulative])
        decisions[name] = LabelDecision(name, artifact.evidence_sha256, parents,
                                        artifact.direct_label_wells, cumulative, digest)
    graph_hash = _digest(["rwt.label-lineage.v1", [(name, decisions[name].sha256) for name in order]])
    return LabelLineage(tuple(decisions.values()), tuple(sorted(well_spellings.values())), graph_hash)

