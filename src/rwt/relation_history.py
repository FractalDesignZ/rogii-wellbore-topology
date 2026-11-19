"""Connected components across ordered revisions of a keyed, undirected graph.

For each adjacent pair this computes the two inclusion maps on connected
components from their intersection graph. The output is local to that pair.
It does not assign permanent component identities or compute barcode intervals.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


Component = tuple[str, ...]
Edge = tuple[str, str]


def _key(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


@dataclass(frozen=True)
class GraphRevision:
    """One simple graph. A node key must denote the same vertex across its domain.

    `domain` identifies the well or other source and the node-key convention.
    `key` identifies this revision. Nodes include all active isolates. Edges
    are endpoint pairs among those nodes. Input order has no significance.
    """

    domain: str
    key: str
    nodes: tuple[str, ...]
    edges: tuple[Edge, ...]

    def __post_init__(self):
        if not _key(self.domain) or not _key(self.key):
            raise ValueError("domain and revision key must be nonempty strings")
        if isinstance(self.nodes, str) or isinstance(self.edges, str):
            raise ValueError("nodes and edges must be collections")
        nodes = tuple(self.nodes)
        if not all(_key(node) for node in nodes) or len(set(nodes)) != len(nodes):
            raise ValueError("node keys must be distinct nonempty strings")
        active = set(nodes)
        edges = set()
        for edge in self.edges:
            if not isinstance(edge, (tuple, list)) or len(edge) != 2:
                raise ValueError("each edge must be a pair of node keys")
            a, b = edge
            if not all(_key(v) and v in active for v in (a, b)) or a == b:
                raise ValueError("edge endpoints must be distinct declared nodes")
            pair = (a, b) if a < b else (b, a)
            if pair in edges:
                raise ValueError("duplicate undirected edge")
            edges.add(pair)
        object.__setattr__(self, "nodes", tuple(sorted(nodes)))
        object.__setattr__(self, "edges", tuple(sorted(edges)))


@dataclass(frozen=True)
class ComponentMap:
    """Both images of one connected component of the intersection."""

    intersection: Component
    left: Component
    right: Component


@dataclass(frozen=True)
class ComponentSpan:
    """The intersection and its maps for one pair of adjacent revisions.

    An appearing/departing component has no vertex in the other revision.
    A split/merge lists a left/right component linked to multiple distinct
    components on the other side. Read each link's members via `maps`.
    """

    left_key: str
    right_key: str
    intersection_nodes: tuple[str, ...]
    intersection_edges: tuple[Edge, ...]
    maps: tuple[ComponentMap, ...]
    appearing: tuple[Component, ...]
    departing: tuple[Component, ...]
    splits: tuple[Component, ...]
    merges: tuple[Component, ...]


@dataclass(frozen=True)
class ComponentHistory:
    revisions: tuple[GraphRevision, ...]
    components: tuple[tuple[Component, ...], ...]
    spans: tuple[ComponentSpan, ...]


def _components(nodes, edges):
    adjacency = {node: [] for node in nodes}
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    unseen = set(nodes)
    components = []
    for node in nodes:
        if node not in unseen:
            continue
        unseen.remove(node)
        stack = [node]
        members = []
        while stack:
            here = stack.pop()
            members.append(here)
            for neighbor in adjacency[here]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        components.append(tuple(sorted(members)))
    return tuple(components)


def _span(left, right, left_components, right_components):
    nodes = tuple(sorted(set(left.nodes).intersection(right.nodes)))
    edges = tuple(sorted(set(left.edges).intersection(right.edges)))
    left_by_node = {v: c for c in left_components for v in c}
    right_by_node = {v: c for c in right_components for v in c}
    links = tuple(ComponentMap(c, left_by_node[c[0]], right_by_node[c[0]])
                  for c in _components(nodes, edges))
    images = {c: set() for c in left_components}
    preimages = {c: set() for c in right_components}
    for link in links:
        images[link.left].add(link.right)
        preimages[link.right].add(link.left)
    return ComponentSpan(
        left.key, right.key, nodes, edges, links,
        tuple(c for c in right_components if not preimages[c]),
        tuple(c for c in left_components if not images[c]),
        tuple(c for c in left_components if len(images[c]) > 1),
        tuple(c for c in right_components if len(preimages[c]) > 1),
    )


def component_history(revisions: Iterable[GraphRevision]) -> ComponentHistory:
    """Compute H0 components and adjacent spans in supplied revision order.

    All snapshots must share a domain and have unique revision keys. Store
    source keys unchanged across revisions. An empty graph is a valid snapshot.
    The result and all member/edge orders are deterministic.
    """
    rows = tuple(revisions)
    if not rows or not all(isinstance(row, GraphRevision) for row in rows):
        raise ValueError("at least one GraphRevision is required")
    if len({row.domain for row in rows}) != 1 or len({row.key for row in rows}) != len(rows):
        raise ValueError("revisions must share one domain and have distinct keys")
    components = tuple(_components(row.nodes, row.edges) for row in rows)
    spans = tuple(_span(left, right, a, b) for left, right, a, b in
                  zip(rows, rows[1:], components, components[1:]))
    return ComponentHistory(rows, components, spans)
