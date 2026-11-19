from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rwt.crossfit_roles import nested_well_roles
from rwt.label_lineage import LabelArtifact, LabelOverlapError, build_label_lineage


def artifact(name, parents=(), wells=()):
    evidence = sha256(name.encode("utf8")).hexdigest()
    return LabelArtifact(name, evidence, parents, wells)


def test_nested_fit_excludes_outer_and_bank_but_peer_filter_keeps_labels():
    outer = {"o": 0, "b": 1, "r1": 1, "r2": 2, "r3": 2}
    inner = {"b": 0, "r1": 1, "r2": 2, "r3": 1}
    matching = nested_well_roles(outer, inner, 0, 0)
    sibling = nested_well_roles(outer, inner, 0, 2)
    held = matching.query_wells + matching.residual_wells
    graph = build_label_lineage([
        artifact("matching_fit", wells=matching.fit_wells),
        artifact("matching_curve", ["matching_fit"]),
        artifact("same_outer_sibling", wells=sibling.fit_wells),
        artifact("other_outer_fit", wells=[w for w, fold in outer.items() if fold != 1]),
        artifact("filtered_sibling", ["same_outer_sibling"]),
        artifact("filtered_other_fold", ["other_outer_fit"]),
    ])
    proof = graph.require_exclusion("matching_curve", held)
    assert proof.held_wells == ("b", "o") and proof.overlap == ()
    assert proof.decision_sha256 == graph.decision("matching_curve").sha256
    for name, expected in (("filtered_sibling", ("b",)), ("filtered_other_fold", ("o",))):
        with pytest.raises(LabelOverlapError) as error:
            graph.require_exclusion(name, held)
        assert error.value.proof.overlap == expected
        assert error.value.proof.graph_sha256 == graph.sha256


def test_scalar_population_inherits_geometry_roster_despite_identical_case_names():
    scalar = artifact("visible-heel-q")
    geometry = artifact("registered-geometry", wells=("bank-well",))
    # Both roster calculations happened to return the same cases. The edge carries the provenance.
    old = artifact("old-roster", (geometry.artifact_id,))
    clean = artifact("observed-roster", (scalar.artifact_id,))
    graph = build_label_lineage([scalar, geometry, old, clean,
                                artifact("training-mean-old", ("visible-heel-q", "old-roster")),
                                artifact("training-mean-clean", ("visible-heel-q", "observed-roster"))])
    graph.require_exclusion("visible-heel-q", ["bank-well"])
    graph.require_exclusion("training-mean-clean", ["bank-well"])
    with pytest.raises(LabelOverlapError) as rejected:
        graph.require_exclusion("training-mean-old", ["bank-well"])
    assert rejected.value.proof.overlap == ("bank-well",)


def test_diamond_is_immutable_and_order_independent_and_binds_evidence():
    source_list = ["r2", "r1"]
    root = artifact("root", wells=source_list)
    source_list.append("later")
    assert root.direct_label_wells == ("r1", "r2")
    left = artifact("left", ["root"], ["r3"])
    right = artifact("right", ["root"], ["r3", "r4"])
    leaf = artifact("leaf", ["right", "left"])
    nodes = [root, left, right, leaf]
    graph = build_label_lineage(nodes)
    reordered = [replace(leaf, parents=("left", "right")), right, left,
                 replace(root, direct_label_wells=("r2", "r1"))]
    assert graph == build_label_lineage(reordered)
    receipt = graph.decision("leaf")
    assert receipt.label_wells == ("r1", "r2", "r3", "r4")
    assert receipt.direct_label_wells == ()
    assert receipt.parent_hashes == tuple((n, graph.decision(n).sha256) for n in ("left", "right"))
    assert [d.artifact_id for d in graph.decisions].index("left") < [d.artifact_id for d in graph.decisions].index("leaf")
    proof = graph.require_exclusion("leaf", ["x", "y"])
    assert graph.require_exclusion("leaf", ("y", "x")) == proof
    assert graph.require_exclusion("leaf", ("x",)).sha256 != proof.sha256
    changed = build_label_lineage([replace(root, evidence_sha256="a" * 64), *nodes[1:]])
    assert changed.sha256 != graph.sha256
    assert all(changed.decision(n.artifact_id).sha256 != graph.decision(n.artifact_id).sha256 for n in nodes)
    assert changed.require_exclusion("leaf", ["x", "y"]).sha256 != proof.sha256
    for obj, key in ((root, "parents"), (receipt, "label_wells"), (graph, "decisions"), (proof, "overlap")):
        with pytest.raises(FrozenInstanceError):
            setattr(obj, key, ())


@pytest.mark.parametrize("bad", ["ROOT", " root", "root\n", "Cafe\u0301"])
def test_artifact_alias_and_noncanonical_parent_denied(bad):
    with pytest.raises(ValueError):
        build_label_lineage([artifact("root"), artifact("child", parents=[bad])])


def test_bad_graph_and_malformed_digests_are_rejected():
    root = artifact("Usable")
    for nodes in ((root, root), (root, artifact("usable")), (artifact("solo", ["missing"]),),
                  (artifact("a", ["b"]), artifact("b", ["a"])), (artifact("self", ["self"]),), ()):
        with pytest.raises(ValueError):
            build_label_lineage(nodes)
    for parents in (["parent", "parent"], ["parent", "PARENT"], "parent"):
        with pytest.raises(ValueError):
            artifact("node", parents)
    for digest in ("", "A" * 64, "0" * 63, "x" * 64, "0" * 64 + "\n", None):
        with pytest.raises(ValueError):
            replace(root, evidence_sha256=digest)
    with pytest.raises(KeyError):
        build_label_lineage([root]).require_exclusion("unknown", ("x",))


def test_well_aliases_cannot_hide_in_parents_or_holdout():
    root = artifact("source", wells=("Straße", "Café"))
    graph = build_label_lineage([root, artifact("clean")])
    for wells in (["strasse"], ["Cafe\u0301"], ["x", "X"], ["x", "x"], ["X\t"], "Straße"):
        with pytest.raises(ValueError):
            graph.require_exclusion("clean", wells)
    with pytest.raises(ValueError):
        build_label_lineage([root, artifact("other", wells=["STRASSE"])])
    for vals in (["A", "a"], ["Cafe\u0301"], ["well\x00"], ["a", "a"]):
        with pytest.raises(ValueError):
            artifact("bad-source", wells=vals)
    with pytest.raises(LabelOverlapError):
        graph.require_exclusion("source", ["Straße"])

