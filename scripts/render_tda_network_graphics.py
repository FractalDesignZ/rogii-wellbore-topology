#!/usr/bin/env python3
"""
Render network + TDA-relevant graphics for rwt (methods testing / docs).

Outputs PNG panels under artifacts/graphics/:
  01_neighbor_network.png   — k-NN spatial well graph (sheaf-lite F_u→e)
  02_persistence_barcode.png — H0 barcode of well XY Vietoris–Rips style
  03_open_curve_holonomy.png — trajectory with torsion / E_ref coloring
  04_dual_transport.png     — forward / backward / mid prior along MD
  05_tda_dashboard.png      — multi-panel combined dashboard
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rwt.geometry import e_ref_joint_strain, station_local_geometry, trajectory_geometry_features
from rwt.load import iter_well_frames
from rwt.middle_out import dual_transport_prior, resolve_middle_out
from rwt.spatial import build_spatial_index

OUT = ROOT / "artifacts" / "graphics"
DATA = ROOT / "data" / "raw"

# dark scientific aesthetic
BG = "#0b1020"
FG = "#e8eefc"
ACCENT = "#5ce1ff"
ACCENT2 = "#ff6bcb"
ACCENT3 = "#a78bfa"
GRID = "#1c2744"
EDGE = "#3d5a80"


def _style_ax(ax, title: str = "") -> None:
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.title.set_color(FG)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)


def load_wells(max_wells: int = 120) -> list[tuple[str, pd.DataFrame]]:
    wells = []
    for i, (wid, df) in enumerate(iter_well_frames(DATA, "train")):
        wells.append((wid, df))
        if max_wells and i + 1 >= max_wells:
            break
    return wells


def h0_persistence_intervals(points: np.ndarray, max_eps: float | None = None) -> list[tuple[float, float]]:
    """
    Single-linkage H0 barcode (Kruskal): birth=0, death=merge distance.
    Classic Vietoris–Rips H0 skeleton — real computation, not decoration.
    """
    n = len(points)
    if n < 2:
        return []
    # pairwise distances
    dmat = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((float(dmat[i, j]), i, j))
    edges.sort()
    parent = list(range(n))
    rank = [0] * n

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1
        return True

    deaths: list[float] = []
    components = n
    for dist, i, j in edges:
        if max_eps is not None and dist > max_eps:
            break
        if union(i, j):
            deaths.append(dist)
            components -= 1
            if components == 1:
                break
    # one infinite bar for final component; finite bars birth 0 death merge
    intervals = [(0.0, d) for d in deaths]
    intervals.append((0.0, float("inf")))
    return intervals


def render_neighbor_network(wells: list[tuple[str, pd.DataFrame]], path: Path) -> nx.Graph:
    idx = build_spatial_index(wells)
    G = nx.Graph()
    pos = {}
    colors = []
    for wid, node in idx.nodes.items():
        G.add_node(wid)
        pos[wid] = (node.cx, node.cy)
        # color by azimuth angle
        ang = np.arctan2(node.azimuth_y, node.azimuth_x)
        colors.append(ang)

    for wid in list(idx.nodes.keys()):
        for nid, dist in idx.neighbors(wid, k=4, max_dist=600.0, min_cos_azimuth=-0.2):
            if not G.has_edge(wid, nid):
                G.add_edge(wid, nid, weight=dist, close=dist < 150.0)

    fig, ax = plt.subplots(figsize=(12, 10), facecolor=BG)
    _style_ax(ax, "Spatial neighbor graph  ·  k-NN XY  ·  d<600 ft  ·  sheaf-lite Fᵤ→ₑ")

    # edges: close vs far
    close_e = [(u, v) for u, v, d in G.edges(data=True) if d.get("close")]
    far_e = [(u, v) for u, v, d in G.edges(data=True) if not d.get("close")]
    nx.draw_networkx_edges(G, pos, edgelist=far_e, ax=ax, edge_color=EDGE, width=0.6, alpha=0.45)
    nx.draw_networkx_edges(G, pos, edgelist=close_e, ax=ax, edge_color=ACCENT2, width=1.4, alpha=0.85)

    node_list = list(G.nodes())
    ang = np.array([np.arctan2(idx.nodes[n].azimuth_y, idx.nodes[n].azimuth_x) for n in node_list])
    sc = ax.scatter(
        [pos[n][0] for n in node_list],
        [pos[n][1] for n in node_list],
        c=ang,
        cmap="cool",
        s=55,
        edgecolors="white",
        linewidths=0.3,
        zorder=5,
    )
    cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("lateral azimuth", color=FG)
    cb.ax.yaxis.set_tick_params(color=FG)
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color=FG)

    ax.set_xlabel("Easting X (ft)")
    ax.set_ylabel("Northing Y (ft)")
    ax.set_aspect("equal", adjustable="datalim")
    leg = [
        mpatches.Patch(color=ACCENT2, label="edge d < 150 ft (strong transfer)"),
        mpatches.Patch(color=EDGE, label="edge d < 600 ft"),
    ]
    ax.legend(handles=leg, facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=8, loc="upper right")
    ax.text(
        0.02,
        0.02,
        f"nodes={G.number_of_nodes()}  edges={G.number_of_edges()}  ·  open-core public spatial layer",
        transform=ax.transAxes,
        color=ACCENT,
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=BG)
    plt.close(fig)
    return G


def render_persistence_barcode(wells: list[tuple[str, pd.DataFrame]], path: Path) -> None:
    idx = build_spatial_index(wells)
    pts = np.array([[n.cx, n.cy] for n in idx.nodes.values()], dtype=float)
    # scale to feet-ish already
    intervals = h0_persistence_intervals(pts)
    finite = [(b, d) for b, d in intervals if np.isfinite(d)]
    infinite = [(b, d) for b, d in intervals if not np.isfinite(d)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor=BG)
    ax, ax2 = axes
    _style_ax(ax, "H₀ persistence barcode  ·  well XY point cloud")
    deaths = sorted([d for _, d in finite])
    for i, (b, d) in enumerate(sorted(finite, key=lambda t: t[1])):
        ax.hlines(i, b, d, colors=ACCENT, linewidth=2.2)
        ax.plot([b, d], [i, i], "o", color=ACCENT2, markersize=3)
    if infinite:
        y0 = len(finite)
        xmax = max(deaths) * 1.05 if deaths else 1.0
        ax.hlines(y0, 0, xmax, colors=ACCENT3, linewidth=3.0)
        ax.text(xmax * 0.7, y0 + 0.3, "∞  final component", color=ACCENT3, fontsize=9)
    ax.set_xlabel("filtration scale ε (ft)")
    ax.set_ylabel("H₀ bars (merge death)")
    ax.set_ylim(-1, len(finite) + 2)

    _style_ax(ax2, "Merge dendrogram scale  ·  connected components vs ε")
    if deaths:
        eps = np.linspace(0, max(deaths) * 1.05, 200)
        n0 = len(pts)
        # n_components = n - number of merges with death <= eps
        comps = [n0 - sum(1 for d in deaths if d <= e) for e in eps]
        ax2.fill_between(eps, comps, alpha=0.35, color=ACCENT)
        ax2.plot(eps, comps, color=ACCENT, linewidth=2)
        ax2.axvline(150, color=ACCENT2, linestyle="--", linewidth=1.2, label="150 ft transfer")
        ax2.axvline(600, color=ACCENT3, linestyle="--", linewidth=1.2, label="600 ft graph")
        ax2.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=8)
    ax2.set_xlabel("ε (ft)")
    ax2.set_ylabel("# H₀ components")
    fig.suptitle(
        "Vietoris–Rips H₀ on well centroids  ·  topology of the field network",
        color=FG,
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def render_open_curve_holonomy(wells: list[tuple[str, pd.DataFrame]], path: Path) -> None:
    # pick a well with interesting length
    wid, df = max(wells, key=lambda t: len(t[1]))
    x = df["X"].to_numpy(float)
    y = df["Y"].to_numpy(float)
    z = df["Z"].to_numpy(float)
    md = df["MD"].to_numpy(float)
    local = station_local_geometry(x, y, z)
    geom = trajectory_geometry_features(md, x, y, z)
    tors = local["local_torsion"]
    eref = local["e_ref"]

    fig = plt.figure(figsize=(13, 9), facecolor=BG)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1.0], hspace=0.28, wspace=0.22)

    # 3D-ish XY plan view colored by torsion
    ax = fig.add_subplot(gs[0, 0])
    _style_ax(ax, f"Open curve γ  ·  well {wid}  ·  color = discrete torsion τ")
    pts = np.column_stack([x, y])
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap="magma", linewidths=2.2)
    lc.set_array(tors[1:])
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    cb = fig.colorbar(lc, ax=ax, fraction=0.046)
    cb.set_label("τ (rad)", color=FG)
    cb.ax.yaxis.set_tick_params(color=FG)
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color=FG)

    ax2 = fig.add_subplot(gs[0, 1])
    _style_ax(ax2, "Vertical section  ·  color = SE(3) E_ref joint strain")
    pts2 = np.column_stack([md, z])
    segs2 = np.stack([pts2[:-1], pts2[1:]], axis=1)
    lc2 = LineCollection(segs2, cmap="viridis", linewidths=2.2)
    lc2.set_array(eref[1:])
    ax2.add_collection(lc2)
    ax2.autoscale()
    ax2.set_xlabel("MD (ft)")
    ax2.set_ylabel("Z (ft)")
    ax2.invert_yaxis()
    cb2 = fig.colorbar(lc2, ax=ax2, fraction=0.046)
    cb2.set_label("E_ref", color=FG)
    cb2.ax.yaxis.set_tick_params(color=FG)
    plt.setp(plt.getp(cb2.ax.axes, "yticklabels"), color=FG)

    ax3 = fig.add_subplot(gs[1, :])
    _style_ax(ax3, "Holonomy-rate series along open MD  ·  τ(s) and E_ref(s)  ·  no false closure")
    ax3.plot(md, tors, color=ACCENT2, linewidth=1.4, label="discrete torsion τ")
    ax3b = ax3.twinx()
    ax3b.plot(md, eref, color=ACCENT, linewidth=1.2, alpha=0.85, label="E_ref")
    ax3b.tick_params(colors=FG)
    ax3b.spines["right"].set_color(GRID)
    ax3.set_xlabel("MD (ft)")
    ax3.set_ylabel("τ (rad)", color=ACCENT2)
    ax3b.set_ylabel("E_ref", color=ACCENT)
    lines, labels = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3b.get_legend_handles_labels()
    ax3.legend(lines + lines2, labels + labels2, facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=8)
    ax3.text(
        0.01,
        0.95,
        f"cum_twist={geom['cum_twist']:.3f}  mean|τ|={geom['mean_abs_torsion']:.4f}  mean E_ref={geom['mean_e_ref']:.4f}",
        transform=ax3.transAxes,
        color=ACCENT3,
        fontsize=9,
        va="top",
    )
    fig.savefig(path, dpi=160, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def render_dual_transport(wells: list[tuple[str, pd.DataFrame]], path: Path) -> None:
    wid, df = wells[0]
    sp = resolve_middle_out(df, well_id=wid)
    dual = dual_transport_prior(df, sp)
    md = df["MD"].to_numpy(float)
    tvt = df["TVT"].to_numpy(float) if "TVT" in df.columns else np.full(len(df), np.nan)

    fig, ax = plt.subplots(figsize=(12, 5.5), facecolor=BG)
    _style_ax(ax, f"Dual-transport middle-out  ·  well {wid}")
    ax.plot(md, dual["prior_forward"], color=ACCENT, linewidth=1.5, alpha=0.9, label="prior_forward")
    ax.plot(md, dual["prior_backward"], color=ACCENT3, linewidth=1.5, alpha=0.9, label="prior_backward (structural)")
    ax.plot(md, dual["prior_mid"], color=ACCENT2, linewidth=2.4, label="prior_mid (fused)")
    if np.isfinite(tvt).any():
        ax.plot(md, tvt, color=FG, linewidth=1.0, alpha=0.35, label="TVT (train label)")
    if len(sp.known_idx):
        k0, k1 = int(sp.known_idx[0]), int(sp.known_idx[-1])
        ax.axvspan(md[k0], md[k1], color=ACCENT, alpha=0.08, label="heel known")
    if len(sp.predict_idx):
        p0, p1 = int(sp.predict_idx[0]), int(sp.predict_idx[-1])
        ax.axvspan(md[p0], md[p1], color=ACCENT2, alpha=0.08, label="toe predict")
    ax.set_xlabel("MD (ft)")
    ax.set_ylabel("TVT (ft)")
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=8, ncol=2, loc="best")
    ax.text(
        0.01,
        0.02,
        "prior_mid = w_f·forward + w_b·backward   ·   residual learner targets TVT − prior_mid on toe",
        transform=ax.transAxes,
        color=ACCENT,
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=BG)
    plt.close(fig)


def render_dashboard(
    wells: list[tuple[str, pd.DataFrame]],
    path: Path,
) -> None:
    """Compact 2×2 scientific dashboard."""
    idx = build_spatial_index(wells)
    pts = np.array([[n.cx, n.cy] for n in idx.nodes.values()])
    intervals = h0_persistence_intervals(pts)
    finite = sorted([d for _, d in intervals if np.isfinite(d)])

    fig = plt.figure(figsize=(14, 11), facecolor=BG)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    # network mini
    ax = fig.add_subplot(gs[0, 0])
    _style_ax(ax, "Network · well neighbor graph")
    G = nx.Graph()
    pos = {w: (n.cx, n.cy) for w, n in idx.nodes.items()}
    for w in idx.nodes:
        G.add_node(w)
        for nid, dist in idx.neighbors(w, k=3, max_dist=600.0):
            G.add_edge(w, nid, w=dist)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=EDGE, width=0.5, alpha=0.5)
    ax.scatter([p[0] for p in pos.values()], [p[1] for p in pos.values()], c=ACCENT, s=18, zorder=3)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])

    # barcode mini
    ax2 = fig.add_subplot(gs[0, 1])
    _style_ax(ax2, "TDA · H₀ barcode (XY Rips)")
    for i, d in enumerate(finite[:80]):
        ax2.hlines(i, 0, d, colors=ACCENT, linewidth=1.6)
    if finite:
        ax2.axvline(150, color=ACCENT2, ls="--", lw=1)
        ax2.axvline(600, color=ACCENT3, ls="--", lw=1)
    ax2.set_xlabel("ε (ft)")

    # holonomy mini
    wid, df = max(wells[:20], key=lambda t: len(t[1]))
    ax3 = fig.add_subplot(gs[1, 0])
    _style_ax(ax3, f"SE(3) · torsion along {wid}")
    local = station_local_geometry(df["X"].to_numpy(float), df["Y"].to_numpy(float), df["Z"].to_numpy(float))
    ax3.plot(df["MD"], local["local_torsion"], color=ACCENT2, lw=1.2)
    ax3.plot(df["MD"], local["e_ref"], color=ACCENT, lw=1.0, alpha=0.7)
    ax3.set_xlabel("MD")

    # dual transport mini
    ax4 = fig.add_subplot(gs[1, 1])
    _style_ax(ax4, f"Middle-out · dual prior {wid}")
    sp = resolve_middle_out(df, well_id=wid)
    dual = dual_transport_prior(df, sp)
    ax4.plot(df["MD"], dual["prior_mid"], color=ACCENT2, lw=2)
    ax4.plot(df["MD"], dual["prior_forward"], color=ACCENT, lw=1, alpha=0.7)
    ax4.plot(df["MD"], dual["prior_backward"], color=ACCENT3, lw=1, alpha=0.7)
    ax4.set_xlabel("MD")

    fig.suptitle(
        "rwt · topology-sees-sequence methods dashboard\n"
        "network · H₀ · open-curve holonomy · dual-transport middle-out",
        color=FG,
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.savefig(path, dpi=170, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    if not DATA.is_dir() or not (DATA / "train").is_dir():
        print("NEED_INFO: data/raw/train missing", file=sys.stderr)
        return 2
    OUT.mkdir(parents=True, exist_ok=True)
    print("Loading wells…", flush=True)
    wells = load_wells(120)
    print(f"n_wells={len(wells)}", flush=True)

    print("01 neighbor network", flush=True)
    render_neighbor_network(wells, OUT / "01_neighbor_network.png")
    print("02 persistence barcode", flush=True)
    render_persistence_barcode(wells, OUT / "02_persistence_barcode.png")
    print("03 open curve holonomy", flush=True)
    render_open_curve_holonomy(wells, OUT / "03_open_curve_holonomy.png")
    print("04 dual transport", flush=True)
    render_dual_transport(wells, OUT / "04_dual_transport.png")
    print("05 dashboard", flush=True)
    render_dashboard(wells, OUT / "05_tda_dashboard.png")

    # index
    lines = [
        "# rwt graphics index",
        "",
        "Generated by `scripts/render_tda_network_graphics.py` from real competition laterals.",
        "",
        "| File | Content |",
        "|------|---------|",
        "| `01_neighbor_network.png` | Spatial k-NN well graph (600/150 ft), azimuth-colored |",
        "| `02_persistence_barcode.png` | H₀ Vietoris–Rips barcode + component decay |",
        "| `03_open_curve_holonomy.png` | Trajectory τ / E_ref holonomy-rate series |",
        "| `04_dual_transport.png` | Forward / backward / mid middle-out prior |",
        "| `05_tda_dashboard.png` | Combined methods dashboard |",
        "",
        "Not a Kaggle submission — methods visualization only.",
        "",
    ]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"OK wrote graphics → {OUT}")
    for p in sorted(OUT.glob("*.png")):
        print(f"  {p.name}  ({p.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
