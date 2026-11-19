"""
Spatial neighbor graph for public sheaf-lite TVT transfer.

Public layer only: k-NN on lateral XY centroids, azimuth gate, nearest-XY TVT
restriction. Proprietary sheaf Laplacians stay in private_kernel.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from rwt.load import normalize_curve_nulls


SpatialLabelPolicy = Literal["observed_only", "full_tvt"]


@dataclass
class WellSpatialNode:
    well_id: str
    cx: float
    cy: float
    azimuth_x: float
    azimuth_y: float
    md: np.ndarray
    x: np.ndarray
    y: np.ndarray
    tvt: np.ndarray  # visible donor observations; NaN where unknown
    label_policy: SpatialLabelPolicy = "observed_only"


@dataclass
class SpatialIndex:
    nodes: dict[str, WellSpatialNode] = field(default_factory=dict)
    max_dist: float = 600.0
    close_dist: float = 150.0
    label_policy: SpatialLabelPolicy = "observed_only"

    def add(self, node: WellSpatialNode) -> None:
        self.nodes[node.well_id] = node

    def neighbors(
        self,
        well_id: str,
        *,
        k: int = 5,
        max_dist: float | None = None,
        min_cos_azimuth: float = -0.25,
        query_node: WellSpatialNode | None = None,
    ) -> list[tuple[str, float]]:
        """Find donors for indexed or unseen query geometry without changing the index."""
        well_id = str(well_id)
        a = query_node if query_node is not None else self.nodes.get(well_id)
        if a is None or k <= 0:
            return []
        max_dist = float(self.max_dist if max_dist is None else max_dist)
        if not (np.isfinite(a.cx) and np.isfinite(a.cy)):
            return []
        hits: list[tuple[str, float]] = []
        for wid, b in self.nodes.items():
            if wid == well_id:
                continue
            if not (np.isfinite(b.cx) and np.isfinite(b.cy)):
                continue
            d = float(np.hypot(a.cx - b.cx, a.cy - b.cy))
            if not np.isfinite(d) or d > max_dist:
                continue
            cos = float(a.azimuth_x * b.azimuth_x + a.azimuth_y * b.azimuth_y)
            na = float(np.hypot(a.azimuth_x, a.azimuth_y))
            nb = float(np.hypot(b.azimuth_x, b.azimuth_y))
            if na > 1e-9 and nb > 1e-9:
                cos = cos / (na * nb)
            if cos < min_cos_azimuth:
                continue
            hits.append((wid, d))
        hits.sort(key=lambda t: (t[1], t[0]))
        return hits[:k]


def well_spatial_node_from_lateral(
    well_id: str,
    df: pd.DataFrame,
    *,
    tvt_col: str = "TVT",
    tvt_input_col: str = "TVT_input",
    label_policy: SpatialLabelPolicy = "observed_only",
) -> WellSpatialNode:
    """Snapshot geometry and explicitly visible observations.

    The default never reads full TVT, even if TVT_input is absent or empty.
    ``full_tvt`` is an explicit opt-in for authorized training donors only;
    callers must exclude held-out labels before selecting that policy.
    """
    if label_policy not in ("observed_only", "full_tvt"):
        raise ValueError(f"unknown spatial label policy: {label_policy!r}")
    x = df["X"].to_numpy(dtype=float).copy()
    y = df["Y"].to_numpy(dtype=float).copy()
    md = df["MD"].to_numpy(dtype=float) if "MD" in df.columns else np.arange(len(df), dtype=float)
    n = len(x)
    if n == 0:
        return WellSpatialNode(
            well_id=str(well_id),
            cx=float("nan"),
            cy=float("nan"),
            azimuth_x=0.0,
            azimuth_y=0.0,
            md=md,
            x=x,
            y=y,
            tvt=np.zeros(0),
            label_policy=label_policy,
        )
    # Prefer the last 60% of the path. Only paired finite XY observations
    # define position; missing coordinates must not become an origin node.
    cut = min(max(0, int(0.4 * n)), n - 1)
    valid_xy = np.flatnonzero(np.isfinite(x) & np.isfinite(y))
    body = valid_xy[valid_xy >= cut]
    if not len(body):
        body = valid_xy
    cx = float(np.mean(x[body])) if len(body) else float("nan")
    cy = float(np.mean(y[body])) if len(body) else float("nan")
    ax = ay = 0.0
    if len(body) >= 2:
        dx = float(x[body[-1]] - x[body[0]])
        dy = float(y[body[-1]] - y[body[0]])
        hn = float(np.hypot(dx, dy))
        if hn > 1e-15:
            ax, ay = dx / hn, dy / hn
    source = tvt_col if label_policy == "full_tvt" else tvt_input_col
    tvt = normalize_curve_nulls(df[source]) if source in df.columns else np.full(n, np.nan)
    tvt[~np.isfinite(tvt)] = np.nan
    return WellSpatialNode(
        well_id=str(well_id),
        cx=cx,
        cy=cy,
        azimuth_x=ax,
        azimuth_y=ay,
        md=md,
        x=x,
        y=y,
        tvt=tvt,
        label_policy=label_policy,
    )


def neighbor_tvt_transfer(
    target: pd.DataFrame,
    neighbor: WellSpatialNode,
    *,
    close_dist: float = 150.0,
) -> dict[str, np.ndarray]:
    """Nearest neighbor XY sample's TVT (vectorized)."""
    tx = target["X"].to_numpy(dtype=float)
    ty = target["Y"].to_numpy(dtype=float)
    n = len(tx)
    out_tvt = np.full(n, np.nan)
    out_d = np.full(n, np.nan)
    # Public nodes can also be constructed directly. Decode label nulls at
    # consumption without changing the node's observations or coordinates.
    nx, ny, ntvt = neighbor.x, neighbor.y, normalize_curve_nulls(neighbor.tvt)
    finite = np.isfinite(ntvt) & np.isfinite(nx) & np.isfinite(ny)
    target_rows = np.flatnonzero(np.isfinite(tx) & np.isfinite(ty))
    if not finite.any() or not len(target_rows):
        return {
            "neighbor_tvt": out_tvt,
            "neighbor_xy_dist": out_d,
            "neighbor_transfer_ok": np.zeros(n, dtype=float),
        }
    nx, ny, ntvt = nx[finite], ny[finite], ntvt[finite]
    # (n, m) distances via broadcasting in chunks if huge
    d2 = (nx[None, :] - tx[target_rows, None]) ** 2 + (ny[None, :] - ty[target_rows, None]) ** 2
    j = np.argmin(d2, axis=1)
    out_tvt[target_rows] = ntvt[j]
    out_d[target_rows] = np.sqrt(d2[np.arange(len(target_rows)), j])
    ok = (out_d <= close_dist).astype(float)
    return {
        "neighbor_tvt": out_tvt,
        "neighbor_xy_dist": out_d,
        "neighbor_transfer_ok": ok,
    }


def attach_spatial_features(
    stations: pd.DataFrame,
    lateral: pd.DataFrame,
    index: SpatialIndex | None,
    well_id: str,
    *,
    max_dist: float | None = None,
    close_dist: float | None = None,
    k_neighbors: int = 5,
) -> pd.DataFrame:
    """Add neighbor transfer columns (multi-neighbor weighted + best).

    Public sheaf-lite upgrade:
    - Best-neighbor columns kept for backward compatibility.
    - New: neighbor_tvt_wmean, neighbor_weight_sum, neighbor_count_close
      (distance-inverse weighted average of up to k close transfers, heel-offset corrected).
    """
    out = stations.copy()
    out["neighbor_tvt"] = np.nan
    out["neighbor_xy_dist"] = np.nan
    out["neighbor_transfer_ok"] = 0.0
    out["neighbor_count"] = 0.0
    out["neighbor_min_dist"] = np.nan
    out["neighbor_offset"] = 0.0
    out["neighbor_tvt_wmean"] = np.nan
    out["neighbor_weight_sum"] = 0.0
    out["neighbor_count_close"] = 0.0
    if index is None:
        return out
    max_dist = float(index.max_dist if max_dist is None else max_dist)
    close_dist = float(index.close_dist if close_dist is None else close_dist)
    query = well_spatial_node_from_lateral(well_id, lateral)
    neigh = index.neighbors(well_id, k=k_neighbors, max_dist=max_dist, query_node=query)
    out["neighbor_count"] = float(len(neigh))
    if not neigh:
        return out
    out["neighbor_min_dist"] = float(neigh[0][1])

    n_lat = len(lateral)
    tx = lateral["X"].to_numpy(dtype=float)
    ty = lateral["Y"].to_numpy(dtype=float)
    n = len(tx)

    # Collect per-neighbor transfers
    transfers: list[dict[str, np.ndarray]] = []
    offsets: list[float] = []
    dists: list[float] = []
    for nid, nd in neigh:
        node = index.nodes[nid]
        tr = neighbor_tvt_transfer(lateral, node, close_dist=close_dist)
        # Heel-only affine offset (TVT_input only)
        offset = 0.0
        if "TVT_input" in lateral.columns:
            heel_tvt = normalize_curve_nulls(lateral["TVT_input"])
            tr_full = tr["neighbor_tvt"]
            m = (
                np.isfinite(heel_tvt)
                & np.isfinite(tr_full)
                & (tr["neighbor_transfer_ok"] > 0.5)
            )
            if m.sum() >= 8:
                offset = float(np.median(heel_tvt[m] - tr_full[m]))
                if abs(offset) > 40.0:
                    offset = 0.0
        transfers.append(tr)
        offsets.append(offset)
        dists.append(float(nd))

    # Best neighbor (legacy columns)
    best_tr, best_off, best_d = transfers[0], offsets[0], dists[0]
    if "station" in out.columns:
        st = out["station"].to_numpy(dtype=int)
        st = np.clip(st, 0, max(n_lat - 1, 0))
        out["neighbor_tvt"] = best_tr["neighbor_tvt"][st] + best_off
        out["neighbor_xy_dist"] = best_tr["neighbor_xy_dist"][st]
        ok = best_tr["neighbor_transfer_ok"][st]
        if best_d > close_dist:
            ok = ok * 0.25
        out["neighbor_transfer_ok"] = ok
        out["neighbor_offset"] = float(best_off)

        # Multi-neighbor inverse-distance weighted mean (only close transfers)
        wmean = np.zeros(n, dtype=float)
        wsum = np.zeros(n, dtype=float)
        n_close = np.zeros(n, dtype=float)
        for tr, off, d in zip(transfers, offsets, dists):
            if d > max_dist:
                continue
            ok_arr = tr["neighbor_transfer_ok"]
            tvt_arr = tr["neighbor_tvt"] + off
            # weight = 1 / (d + eps) * ok
            w = (1.0 / (d + 15.0)) * ok_arr
            fin = np.isfinite(tvt_arr) & (ok_arr > 0.1)
            wmean = np.where(fin, wmean + w * tvt_arr, wmean)
            wsum = np.where(fin, wsum + w, wsum)
            n_close = np.where(fin & (d <= close_dist * 1.5), n_close + 1.0, n_close)
        with np.errstate(invalid="ignore", divide="ignore"):
            wmean = np.where(wsum > 1e-9, wmean / wsum, np.nan)
        out["neighbor_tvt_wmean"] = wmean[st]
        out["neighbor_weight_sum"] = wsum[st]
        out["neighbor_count_close"] = n_close[st]
    return out


def build_spatial_index(
    wells: list[tuple[str, pd.DataFrame]],
    *,
    max_dist: float | None = None,
    close_dist: float | None = None,
    label_policy: SpatialLabelPolicy = "observed_only",
) -> SpatialIndex:
    """
    Build an observed-only donor index by default. Full training labels require
    explicit ``label_policy="full_tvt"`` and caller-controlled fold visibility.
    If radii unset, use field quantiles of centroid distances
    so sparse pads still form a useful graph (coherence on real ROGII scales).
    """
    if label_policy not in ("observed_only", "full_tvt"):
        raise ValueError(f"unknown spatial label policy: {label_policy!r}")
    idx = SpatialIndex(label_policy=label_policy)
    for wid, df in wells:
        idx.add(well_spatial_node_from_lateral(wid, df, label_policy=label_policy))
    nodes = [n for n in idx.nodes.values() if np.isfinite(n.cx) and np.isfinite(n.cy)]
    if len(nodes) >= 2 and (max_dist is None or close_dist is None):
        pts = np.array([[n.cx, n.cy] for n in nodes], dtype=float)
        # vectorized pairwise upper triangle
        d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2))
        iu = np.triu_indices(len(pts), k=1)
        darr = d[iu]
        darr = darr[np.isfinite(darr)]
        if len(darr):
            q05 = float(np.percentile(darr, 5))
            q15 = float(np.percentile(darr, 15))
            if close_dist is None:
                close_dist = max(150.0, q05)
            if max_dist is None:
                max_dist = max(600.0, q15, close_dist * 1.5)
    idx.close_dist = float(close_dist if close_dist is not None else 150.0)
    idx.max_dist = float(max_dist if max_dist is not None else 600.0)
    return idx
