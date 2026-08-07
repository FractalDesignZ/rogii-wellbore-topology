"""
Spatial neighbor graph for public sheaf-lite TVT transfer.

Public layer only: k-NN on lateral XY centroids, azimuth gate, affine MD→TVT
restriction. Proprietary sheaf Laplacians stay in private_kernel.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


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
    tvt: np.ndarray  # known / train labels; NaN where unknown


@dataclass
class SpatialIndex:
    nodes: dict[str, WellSpatialNode] = field(default_factory=dict)
    # Field-scale adaptive radii (ft); filled by build_spatial_index
    max_dist: float = 600.0
    close_dist: float = 150.0

    def add(self, node: WellSpatialNode) -> None:
        self.nodes[node.well_id] = node

    def neighbors(
        self,
        well_id: str,
        *,
        k: int = 5,
        max_dist: float | None = None,
        min_cos_azimuth: float = -0.25,
    ) -> list[tuple[str, float]]:
        """Return [(neighbor_id, xy_dist)] within max_dist, azimuth-compatible."""
        if well_id not in self.nodes:
            return []
        max_dist = float(self.max_dist if max_dist is None else max_dist)
        a = self.nodes[well_id]
        hits: list[tuple[str, float]] = []
        for wid, b in self.nodes.items():
            if wid == well_id:
                continue
            d = float(np.hypot(a.cx - b.cx, a.cy - b.cy))
            if d > max_dist:
                continue
            # directional partition: skip strongly opposite-azimuth laterals
            cos = float(a.azimuth_x * b.azimuth_x + a.azimuth_y * b.azimuth_y)
            na = float(np.hypot(a.azimuth_x, a.azimuth_y))
            nb = float(np.hypot(b.azimuth_x, b.azimuth_y))
            if na > 1e-9 and nb > 1e-9:
                cos = cos / (na * nb)
            if cos < min_cos_azimuth:
                continue
            hits.append((wid, d))
        hits.sort(key=lambda t: t[1])
        return hits[:k]


def well_spatial_node_from_lateral(
    well_id: str,
    df: pd.DataFrame,
    *,
    tvt_col: str = "TVT",
    tvt_input_col: str = "TVT_input",
) -> WellSpatialNode:
    x = df["X"].to_numpy(dtype=float)
    y = df["Y"].to_numpy(dtype=float)
    md = df["MD"].to_numpy(dtype=float) if "MD" in df.columns else np.arange(len(df), dtype=float)
    # centroid of lateral body (prefer last 60% / higher MD)
    cut = max(1, int(0.4 * len(x)))
    cx = float(np.nanmean(x[cut:]))
    cy = float(np.nanmean(y[cut:]))
    dx = float(x[-1] - x[cut]) if len(x) > cut else float(x[-1] - x[0])
    dy = float(y[-1] - y[cut]) if len(y) > cut else float(y[-1] - y[0])
    hn = float(np.hypot(dx, dy)) + 1e-15
    ax, ay = dx / hn, dy / hn
    # TVT profile for transfer: prefer full train TVT, else TVT_input
    if tvt_col in df.columns and df[tvt_col].notna().any():
        tvt = df[tvt_col].to_numpy(dtype=float)
    elif tvt_input_col in df.columns:
        tvt = df[tvt_input_col].to_numpy(dtype=float)
    else:
        tvt = np.full(len(df), np.nan)
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
    )


def neighbor_tvt_transfer(
    target: pd.DataFrame,
    neighbor: WellSpatialNode,
    *,
    close_dist: float = 150.0,
) -> dict[str, np.ndarray]:
    """
    Restriction-map lite: for each target station, nearest neighbor XY sample's TVT.

    If lateral separation of centroids exceeds close_dist, still transfers but
    marks weak confidence. Affine MD alignment is implicit via XY nearest.
    """
    tx = target["X"].to_numpy(dtype=float)
    ty = target["Y"].to_numpy(dtype=float)
    n = len(tx)
    out_tvt = np.full(n, np.nan)
    out_d = np.full(n, np.nan)
    nx, ny, ntvt = neighbor.x, neighbor.y, neighbor.tvt
    finite = np.isfinite(ntvt)
    if not finite.any():
        return {
            "neighbor_tvt": out_tvt,
            "neighbor_xy_dist": out_d,
            "neighbor_transfer_ok": np.zeros(n, dtype=float),
        }
    nx, ny, ntvt = nx[finite], ny[finite], ntvt[finite]
    for i in range(n):
        d2 = (nx - tx[i]) ** 2 + (ny - ty[i]) ** 2
        j = int(np.argmin(d2))
        out_tvt[i] = float(ntvt[j])
        out_d[i] = float(np.sqrt(d2[j]))
    ok = (out_d <= close_dist).astype(float)
    # fill NaN with nanmedian of neighbor
    med = float(np.nanmedian(ntvt))
    out_tvt = np.where(np.isfinite(out_tvt), out_tvt, med)
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
) -> pd.DataFrame:
    """Add neighbor transfer columns to a station frame (no-op if no index)."""
    out = stations.copy()
    out["neighbor_tvt"] = np.nan
    out["neighbor_xy_dist"] = np.nan
    out["neighbor_transfer_ok"] = 0.0
    out["neighbor_count"] = 0.0
    out["neighbor_min_dist"] = np.nan
    if index is None or well_id not in index.nodes:
        return out
    max_dist = float(index.max_dist if max_dist is None else max_dist)
    close_dist = float(index.close_dist if close_dist is None else close_dist)
    neigh = index.neighbors(well_id, k=3, max_dist=max_dist)
    out["neighbor_count"] = float(len(neigh))
    if not neigh:
        return out
    out["neighbor_min_dist"] = float(neigh[0][1])
    best_id, best_d = neigh[0]
    node = index.nodes[best_id]
    tr = neighbor_tvt_transfer(lateral, node, close_dist=close_dist)
    if "station" in out.columns:
        st = out["station"].to_numpy(dtype=int)
        n_lat = len(lateral)
        st = np.clip(st, 0, max(n_lat - 1, 0))
        out["neighbor_tvt"] = tr["neighbor_tvt"][st]
        out["neighbor_xy_dist"] = tr["neighbor_xy_dist"][st]
        out["neighbor_transfer_ok"] = tr["neighbor_transfer_ok"][st]
        if best_d > close_dist:
            out["neighbor_transfer_ok"] = out["neighbor_transfer_ok"] * 0.25
    return out


def build_spatial_index(
    wells: list[tuple[str, pd.DataFrame]],
    *,
    max_dist: float | None = None,
    close_dist: float | None = None,
) -> SpatialIndex:
    """
    Build index; if radii unset, use field quantiles of centroid distances
    so sparse pads still form a useful graph (coherence on real ROGII scales).
    """
    idx = SpatialIndex()
    for wid, df in wells:
        idx.add(well_spatial_node_from_lateral(wid, df))
    nodes = list(idx.nodes.values())
    if len(nodes) >= 2 and (max_dist is None or close_dist is None):
        pts = np.array([[n.cx, n.cy] for n in nodes], dtype=float)
        # sample pairwise for scale
        dlist: list[float] = []
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                dlist.append(float(np.hypot(pts[i, 0] - pts[j, 0], pts[i, 1] - pts[j, 1])))
        darr = np.asarray(dlist, dtype=float)
        # close ≈ 5th pct (strong), max ≈ 15th pct (graph) with floors
        q05 = float(np.percentile(darr, 5))
        q15 = float(np.percentile(darr, 15))
        if close_dist is None:
            close_dist = max(150.0, q05)
        if max_dist is None:
            max_dist = max(600.0, q15, close_dist * 1.5)
    idx.close_dist = float(close_dist if close_dist is not None else 150.0)
    idx.max_dist = float(max_dist if max_dist is not None else 600.0)
    return idx
