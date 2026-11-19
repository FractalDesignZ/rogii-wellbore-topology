"""Shared observation identities, ordered relations and model views for one well."""
from __future__ import annotations

import numpy as np
import pandas as pd

from rwt.block_model import FEATURE_NAMES, INPUT_COLUMNS, OFFSETS, build_block_features
from rwt.offset_path import minimum_cost_offset_path
from rwt.registered_views import inverse_projection, project_trajectory


REFERENCE_CHANNELS = ("TVT", "GR")
HOPS = 4
MOVE_COST = 0.25
RELATION_NAMES = (
    "draft_along_ft", "draft_cross_ft", "incoming_continuity", "outgoing_continuity",
    "incoming_revision", "outgoing_revision", "forward_support_hops", "backward_support_hops",
    "graph_offset_ft",
)
MODEL_COLUMNS = (*FEATURE_NAMES, *RELATION_NAMES, *(f"path_cost_{int(x):+d}" for x in OFFSETS))


def _neighbor_min(costs: np.ndarray) -> np.ndarray:
    """Transport a message across stay and one-bin moves without wrapping."""
    best = costs.copy()
    best[:, 1:] = np.minimum(best[:, 1:], costs[:, :-1] + MOVE_COST)
    best[:, :-1] = np.minimum(best[:, :-1], costs[:, 1:] + MOVE_COST)
    return best - best.min(axis=1, keepdims=True)


def window_messages(local: np.ndarray, continuity: np.ndarray, hops: int = HOPS):
    """Return separate messages and actual connected spans on both sides."""
    local = np.asarray(local, dtype=float)
    continuity = np.asarray(continuity, dtype=float)
    if (local.ndim != 2 or not all(local.shape) or continuity.shape != (len(local) - 1,)
            or not np.isfinite(local).all() or not np.isfinite(continuity).all()
            or (continuity < 0).any() or (continuity > 1).any()):
        raise ValueError("require finite station-by-bin costs and one continuity weight in [0,1] per edge")
    if not isinstance(hops, int) or isinstance(hops, bool) or hops < 0:
        raise ValueError("hops must be a nonnegative integer")
    forward = np.zeros_like(local)
    backward = np.zeros_like(local)
    left = np.zeros(len(local), dtype=np.int32)
    right = left.copy()
    for _ in range(min(hops, len(local) - 1)):
        forward[1:] = continuity[:, None] * _neighbor_min(local[:-1] + forward[:-1])
        backward[:-1] = continuity[:, None] * _neighbor_min(local[1:] + backward[1:])
        left[1:] = np.where(continuity > 0, left[:-1] + 1, 0)
        right[:-1] = np.where(continuity > 0, right[1:] + 1, 0)
    return forward, backward, left, right


def _continuity(base: np.ndarray) -> np.ndarray:
    """Gate an edge at a change in GR or a change of survey direction."""
    c = {k: base[:, FEATURE_NAMES.index(k)].astype(float) for k in
         ("block_dx", "block_dy", "block_dz", "gr_mean", "gr_noise")}
    d = np.column_stack([c[key] for key in ("block_dx", "block_dy", "block_dz")])
    length = np.linalg.norm(d, axis=1)
    denom = length[:-1] * length[1:]
    cosine = np.divide(np.sum(d[:-1] * d[1:], axis=1), denom,
                       out=np.full(denom.shape, np.nan), where=denom > 0)
    turn = 1 - np.clip(cosine, -1, 1)
    gr = np.abs(np.diff(c["gr_mean"])) / np.maximum(1., c["gr_noise"][1:])
    observed = np.isfinite(turn) | np.isfinite(gr)
    disagreement = np.where(np.isfinite(turn), turn, 0.) + np.where(np.isfinite(gr), gr, 0.)
    return np.where(observed & (disagreement <= 1.), 1. / (1. + disagreement), 0.)


def revise_edge_weights(local, continuity, hops=HOPS):
    """Use observations on opposite sides of each edge without traversing it.

    A pair of independent optima may require an illegal jump. The cost of
    reconciling them with a legal stay/one-bin move discounts the initial
    continuity. A later observation can thus revise an earlier edge weight.
    The two contexts may share a typewell and are not independent sensors.
    """
    forward, backward, _, _ = window_messages(local, continuity, hops)
    left = local[:-1] + forward[:-1]
    right = local[1:] + backward[1:]
    left = left - left.min(axis=1, keepdims=True)
    right = right - right.min(axis=1, keepdims=True)
    feasible_left = left.copy()
    feasible_left[:, 1:] = np.minimum(feasible_left[:, 1:], left[:, :-1])
    feasible_left[:, :-1] = np.minimum(feasible_left[:, :-1], left[:, 1:])
    reconciliation = np.min(feasible_left + right, axis=1)
    return continuity / (1. + reconciliation), forward, backward


def _decode(local, forward, backward):
    joint = local + forward + backward
    joint -= joint.min(axis=1, keepdims=True)
    node_cost = joint.copy()
    node_cost[0] += 0.5 * np.square(OFFSETS / 5.)
    edges = np.broadcast_to(np.array([MOVE_COST, 0., MOVE_COST])[None, :, None],
                            (len(joint) - 1, 3, len(OFFSETS)))
    return OFFSETS[minimum_cost_offset_path(node_cost, edges)], joint


def build_well_bundle(lateral: pd.DataFrame, typewell: pd.DataFrame) -> dict[str, np.ndarray]:
    """Build one sample without reading target columns. Source rows remain in order."""
    raw = lateral.loc[:, list(INPUT_COLUMNS)].to_numpy(dtype=np.float64, copy=True)
    reference = typewell.loc[:, list(REFERENCE_CHANNELS)].to_numpy(dtype=np.float64, copy=True)
    base, meta = build_block_features(lateral, typewell)
    first, center = meta["first_station"], meta["center_station"]
    anchor = int(first[0] - 1)
    # Keep both coordinates of this derived drafting view and the source XYZ.
    # A projection carries the ancestry of the survey. It adds no second vote.
    xyz = raw[:, [INPUT_COLUMNS.index(axis) for axis in ("X", "Y", "Z")]]
    coords = np.full((len(center), 2), np.nan)
    if np.isfinite(xyz[:, :2]).all(axis=1).any():
        view = project_trajectory(*xyz.T, azimuth_deg=45.)
        restored = inverse_projection(view["u"], view["cross"], view["z"], view["origin_xy"], 45.)
        supported = np.isfinite(xyz).all(axis=1)
        if not np.allclose(restored[supported], xyz[supported], rtol=0., atol=1e-7):
            raise ValueError("drafting projection failed its survey round trip")
        coords = np.column_stack([view[k][center] - view[k][anchor] for k in ("u", "cross")])
    signal = base[:, -len(OFFSETS):].astype(float)
    support = np.isfinite(signal)
    row_center = np.ma.median(np.ma.array(signal, mask=~support), axis=1).filled(0.)
    local = np.where(support, np.clip(signal - row_center[:, None], -4., 4.), 0.)
    initial = _continuity(base)
    continuity, first_fwd, first_back = revise_edge_weights(local, initial)
    fwd, back, left, right = window_messages(local, continuity)
    path, joint = _decode(local, fwd, back)
    control, _ = _decode(local, first_fwd, first_back)
    revision = continuity - initial
    features = np.column_stack((base, coords, np.r_[0., continuity], np.r_[continuity, 0.],
                                np.r_[0., revision], np.r_[revision, 0.], left, right, path, joint)).astype(np.float32)
    return {"lateral": raw, "typewell": reference, "features": features,
            "first_station": first, "center_station": center, "row_count": meta["row_count"],
            "last_known": meta["last_known"], "graph_prediction": meta["last_known"] + path,
            "graph_control": meta["last_known"] + control,
            "continuity": continuity, "initial_continuity": initial, "left_hops": left, "right_hops": right}


def trace_node(bundle, well_id: str, first_station: int) -> dict:
    """Resolve a model row and its incoming/outgoing edges to original local row keys."""
    wells = bundle["well_ids"].tolist()
    wi = wells.index(well_id)
    a, b = bundle["block_ptr"][wi:wi+2]
    starts = bundle["first_station"]
    hits = np.flatnonzero(starts[a:b] == first_station)
    if len(hits) != 1:
        raise ValueError("no unique window begins at that original lateral station")
    i = int(a + hits[0])
    center = int(bundle["center_station"][i])
    count = int(bundle["row_count"][i])
    raw = bundle["lateral_values"]
    lateral_origin = int(bundle["lateral_ptr"][wi])

    def reading(row, channel):
        value = raw[lateral_origin + row, INPUT_COLUMNS.index(channel)]
        return {"key":[well_id, "lateral", row, channel],
                "parsed_value":float(value) if np.isfinite(value) else None}

    sources, destinations = bundle["edge_source"], bundle["edge_target"]
    initial, revised = bundle["edge_initial_weight"], bundle["edge_weight"]
    edges = []
    for e in np.flatnonzero((sources == i) | (destinations == i)):
        edges.append({"from":[well_id, int(starts[sources[e]])], "to":[well_id, int(starts[destinations[e]])],
                      "initial_weight":float(initial[e]), "revised_weight":float(revised[e])})
    return {"node":[well_id, first_station], "block_rows_half_open":[first_station,first_station+count],
            "center_xyz":[reading(center, ch) for ch in ("X","Y","Z")],
            "center_gr":reading(center, "GR"), "heel_tvt":reading(int(starts[a])-1,"TVT_input"),
            "typewell_row_range_half_open":[0, int(np.diff(bundle["typewell_ptr"][wi:wi+2])[0])],
            "left_message_hops":int(bundle["left_hops"][i]), "right_message_hops":int(bundle["right_hops"][i]),
            "path_scope":"whole-well offline decode", "incident_edges":edges}
