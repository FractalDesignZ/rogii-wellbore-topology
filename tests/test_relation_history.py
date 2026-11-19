import pytest

from rwt.relation_history import ComponentMap, GraphRevision, component_history


def snapshot(key, nodes, edges=()):
    return GraphRevision("well-1/uv-grid-feet", key, tuple(nodes), tuple(edges))


def test_component_maps_preserve_shared_vertices_when_members_change():
    left = snapshot("s0", "abcx", [("b", "a"), ("b", "c")])
    right = snapshot("s1", "bcdz", [("d", "c"), ("b", "c")])
    history = component_history([left, right])
    span, = history.spans
    assert history.components == ((("a", "b", "c"), ("x",)), (("b", "c", "d"), ("z",)))
    assert span.maps == (ComponentMap(("b", "c"), ("a", "b", "c"), ("b", "c", "d")),)
    assert span.intersection_edges == (("b", "c"),)
    assert span.appearing == (("z",),) and span.departing == (("x",),)
    assert span.splits == span.merges == ()


def test_bridge_cut_and_restoration_are_a_split_then_merge():
    full = (("a", "b"), ("b", "c"), ("c", "d"))
    cut = [full[0], full[2]]
    rows = [snapshot("a", "dcba", full), snapshot("b", "abcd", cut), snapshot("c", "abcd", full)]
    h = component_history(rows)
    split, merge = h.spans
    assert split.intersection_nodes == merge.intersection_nodes == ("a", "b", "c", "d")
    assert split.intersection_edges == merge.intersection_edges == tuple(cut)
    assert split.splits == merge.merges == (("a", "b", "c", "d"),)
    assert split.appearing == merge.departing == split.merges == merge.splits == ()
    assert [(m.intersection, m.right) for m in split.maps] == [(("a", "b"), ("a", "b")), (("c", "d"), ("c", "d"))]
    assert [(m.intersection, m.left) for m in merge.maps] == [(("a", "b"), ("a", "b")), (("c", "d"), ("c", "d"))]


def test_intersection_removes_nonshared_edges_even_if_both_graphs_are_connected():
    a = snapshot("a", "abc", [("a", "b"), ("b", "c")])
    b = snapshot("b", "abc", [("a", "c"), ("b", "c")])
    span, = component_history([a, b]).spans
    assert span.intersection_edges == (("b", "c"),)
    assert [m.intersection for m in span.maps] == [("a",), ("b", "c")]
    assert all(link.left == link.right == ("a", "b", "c") for link in span.maps)
    assert span.splits == span.merges == ()


def test_empty_graphs_and_source_order():
    a = snapshot("later-name", [], [])
    b = snapshot("earlier-name", ["y", "x"], [])
    h = component_history([a, b, snapshot("empty", [])])
    assert [r.key for r in h.revisions] == ["later-name", "earlier-name", "empty"]
    assert h.spans[0].appearing == h.spans[1].departing == (("x",), ("y",))
    assert h.spans[0].maps == h.spans[1].maps == ()
    assert component_history([b]).spans == ()
    assert snapshot("v", "abc", [("b", "a")]) == snapshot("v", "cba", [("a", "b")])


@pytest.mark.parametrize("nodes,edges", [
    ([""], []), (["a", "a"], []), (["x", 1], []), (["a"], [("a", "b")]),
    (["a"], [("a", "a")]), (["a", "b"], [("a", "b"), ("b", "a")]), (["a", "b"], ["ab"]),
])
def test_malformed_vertex_or_edge_rejected(nodes, edges):
    with pytest.raises(ValueError):
        snapshot("k", nodes, edges)


def test_separate_domains_and_reused_revision_keys_rejected():
    a = snapshot("k", "a")
    with pytest.raises(ValueError):
        component_history([a, GraphRevision("other-well", "k2", ("a",), ())])
    with pytest.raises(ValueError):
        component_history([a, a])
    with pytest.raises(ValueError):
        component_history([])
