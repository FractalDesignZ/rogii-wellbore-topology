#!/usr/bin/env python3
"""
Geometry-first GR heatmap visuals.

Perception model:
  MD is dependent on wellbore geometry (arc-length along the path).
  We represent the trajectory in space, then heatmap gamma along it.

  python scripts/render_geometry_gr_heatmap.py --out artifacts/geometry_gr_visuals

Sources: synthetic pad (default) or --stations path to real stations.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.facecolor": "#0d1117",
            "axes.facecolor": "#161b22",
            "axes.edgecolor": "#30363d",
            "axes.labelcolor": "#e6edf3",
            "text.color": "#e6edf3",
            "xtick.color": "#8b949e",
            "ytick.color": "#8b949e",
            "grid.color": "#21262d",
            "legend.facecolor": "#161b22",
            "legend.edgecolor": "#30363d",
            "font.size": 10,
        }
    )


def _path_metrics(md, x, y, z):
    """MD as process index; also true arc-length from XYZ for geometry dependence."""
    import numpy as np

    md = np.asarray(md, dtype=float)
    x, y, z = map(lambda a: np.asarray(a, dtype=float), (x, y, z))
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    dz = np.diff(z, prepend=z[0])
    ds = np.sqrt(dx * dx + dy * dy + dz * dz)
    ds[0] = 0.0
    arc = np.cumsum(ds)
    # horizontal displacement
    dhoriz = np.sqrt(np.diff(x, prepend=x[0]) ** 2 + np.diff(y, prepend=y[0]) ** 2)
    dhoriz[0] = 0.0
    # inclination proxy from dZ/dS
    with np.errstate(invalid="ignore", divide="ignore"):
        incl = np.degrees(np.arccos(np.clip(np.abs(dz) / np.maximum(ds, 1e-9), 0, 1)))
    incl[0] = incl[1] if len(incl) > 1 else 90.0
    return {
        "arc_length": arc,
        "ds": ds,
        "dhoriz_cum": np.cumsum(dhoriz),
        "incl_proxy_deg": incl,
        "tvd_proxy": -z,  # if Z is SS-like negative down, flip for display depth
    }


def _load_synth(n_wells: int, n_stations: int, seed: int):
    from rwt.load import apply_curve_nulls_frame
    from rwt.pad_synth import decode_gr_nulls, make_pad_laterals

    pad = make_pad_laterals(
        n_wells=n_wells, n_stations=n_stations, seed=seed, gr_noise=2.5
    )
    wells = []
    for w in pad["wells"]:
        w = apply_curve_nulls_frame(w)
        gr = decode_gr_nulls(w["GR"].to_numpy(dtype=float))
        wells.append(
            {
                "well_id": str(w["well_id"].iloc[0]),
                "role": str(w["pad_role"].iloc[0]),
                "MD": w["MD"].to_numpy(dtype=float),
                "X": w["X"].to_numpy(dtype=float),
                "Y": w["Y"].to_numpy(dtype=float),
                "Z": w["Z"].to_numpy(dtype=float),
                "GR": gr,
                "TVT": w["TVT_true"].to_numpy(dtype=float)
                if "TVT_true" in w.columns
                else None,
                "is_predict": w["is_predict"].to_numpy(dtype=bool)
                if "is_predict" in w.columns
                else None,
            }
        )
    return wells, "synthetic_pad"


def _load_stations(path: Path, max_wells: int = 6):
    import numpy as np
    import pandas as pd

    from rwt.load import normalize_curve_nulls, GR_ABSURD_LE

    df = pd.read_csv(path)
    if "GR" in df.columns:
        df["GR"] = normalize_curve_nulls(df["GR"], absurd_le=GR_ABSURD_LE)
    wells = []
    for i, (wid, g) in enumerate(df.groupby("well_id")):
        if max_wells and i >= max_wells:
            break
        g = g.sort_values("MD" if "MD" in g.columns else "station")
        wells.append(
            {
                "well_id": str(wid),
                "role": "real",
                "MD": g["MD"].to_numpy(dtype=float)
                if "MD" in g.columns
                else np.arange(len(g), dtype=float),
                "X": g["X"].to_numpy(dtype=float),
                "Y": g["Y"].to_numpy(dtype=float),
                "Z": g["Z"].to_numpy(dtype=float),
                "GR": g["GR"].to_numpy(dtype=float)
                if "GR" in g.columns
                else np.full(len(g), np.nan),
                "TVT": g["TVT"].to_numpy(dtype=float)
                if "TVT" in g.columns
                else None,
                "is_predict": g["is_predict"].to_numpy(dtype=bool)
                if "is_predict" in g.columns
                else None,
            }
        )
    return wells, f"stations:{path.name}"


def main(argv: list[str] | None = None) -> int:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import cm
    from matplotlib.collections import LineCollection
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    ap = argparse.ArgumentParser()
    ap.add_argument("--stations", default="", help="optional stations.csv")
    ap.add_argument("--max-wells", type=int, default=6)
    ap.add_argument("--n-wells", type=int, default=4)
    ap.add_argument("--n-stations", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--out", default=str(ROOT / "artifacts" / "geometry_gr_visuals")
    )
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    _style()

    if a.stations:
        wells, src = _load_stations(Path(a.stations), max_wells=int(a.max_wells))
    else:
        wells, src = _load_synth(int(a.n_wells), int(a.n_stations), int(a.seed))

    # global GR scale
    gr_all = np.concatenate([w["GR"] for w in wells])
    gr_all = gr_all[np.isfinite(gr_all)]
    if len(gr_all) == 0:
        print("no finite GR", flush=True)
        return 1
    gmin, gmax = np.percentile(gr_all, [2, 98])
    if gmax <= gmin:
        gmax = gmin + 1.0
    norm = plt.Normalize(gmin, gmax)
    cmap = cm.inferno

    def colored_line_2d(ax, x, y, c, lw=2.5):
        pts = np.array([x, y]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        # segment colors = mid of endpoints
        cc = 0.5 * (c[:-1] + c[1:])
        lc = LineCollection(segs, cmap=cmap, norm=norm, linewidths=lw)
        lc.set_array(cc)
        ax.add_collection(lc)
        ax.autoscale()
        return lc

    def colored_line_3d(ax, x, y, z, c, lw=2.0):
        pts = np.array([x, y, z]).T.reshape(-1, 1, 3)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        cc = 0.5 * (c[:-1] + c[1:])
        lc = Line3DCollection(segs, cmap=cmap, norm=norm, linewidths=lw)
        lc.set_array(cc)
        ax.add_collection(lc)
        return lc

    # ── pick primary well for focus panels ──
    primary = wells[0]
    for w in wells:
        if w.get("role") == "test":
            primary = w
            break
    pm = _path_metrics(primary["MD"], primary["X"], primary["Y"], primary["Z"])

    # ═══════════════════════════════════════════════════════════
    # 1. Geometry scaffold only (no GR) — pure path representation
    # ═══════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(11, 4.5), dpi=140)
    fig.suptitle(
        "1 · Wellbore geometry representation (MD is arc-length along this path)",
        fontweight="bold",
        y=1.02,
    )
    ax1 = fig.add_subplot(131)
    for w in wells:
        ax1.plot(w["X"], w["Y"], lw=1.5, alpha=0.85, label=w["well_id"])
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_title("Plan view (map)")
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=7, loc="best")

    ax2 = fig.add_subplot(132)
    for w in wells:
        ax2.plot(w["X"], w["Z"], lw=1.5, alpha=0.85)
    ax2.set_xlabel("X")
    ax2.set_ylabel("Z")
    ax2.set_title("Section (vertical)")
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()

    ax3 = fig.add_subplot(133, projection="3d")
    ax3.set_facecolor("#0d1117")
    for w in wells:
        ax3.plot(w["X"], w["Y"], w["Z"], lw=1.8, alpha=0.9)
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Z")
    ax3.set_title("3D trajectory")
    ax3.view_init(elev=18, azim=-60)
    fig.tight_layout()
    p1 = out / "01_geometry_scaffold.png"
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)

    # ═══════════════════════════════════════════════════════════
    # 2. Same geometry, GR heatmap (the money shot)
    # ═══════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(12, 5), dpi=150)
    fig.suptitle(
        "2 · Geometry scaffold + GR heatmap  (reality: path first, sensor colors the path)",
        fontweight="bold",
        y=1.02,
    )
    ax1 = fig.add_subplot(131)
    for w in wells:
        lc = colored_line_2d(ax1, w["X"], w["Y"], w["GR"], lw=3.0)
    cb = fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax1, fraction=0.046, pad=0.04
    )
    cb.set_label("GR")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_title("Plan · GR on geometry")
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.grid(True, alpha=0.25)

    ax2 = fig.add_subplot(132)
    for w in wells:
        colored_line_2d(ax2, w["X"], w["Z"], w["GR"], lw=3.0)
    ax2.set_xlabel("X")
    ax2.set_ylabel("Z")
    ax2.set_title("Section · GR on geometry")
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.25)

    ax3 = fig.add_subplot(133, projection="3d")
    ax3.set_facecolor("#0d1117")
    for w in wells:
        colored_line_3d(ax3, w["X"], w["Y"], w["Z"], w["GR"], lw=2.5)
    # set limits
    ax3.set_xlim(min(w["X"].min() for w in wells), max(w["X"].max() for w in wells))
    ax3.set_ylim(min(w["Y"].min() for w in wells), max(w["Y"].max() for w in wells))
    ax3.set_zlim(min(w["Z"].min() for w in wells), max(w["Z"].max() for w in wells))
    fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax3, shrink=0.55, label="GR"
    )
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Z")
    ax3.set_title("3D · GR heatmapped path")
    ax3.view_init(elev=18, azim=-60)
    fig.tight_layout()
    p2 = out / "02_geometry_gr_heatmap.png"
    fig.savefig(p2, bbox_inches="tight")
    plt.close(fig)

    # ═══════════════════════════════════════════════════════════
    # 3. MD as geometry-dependent: arc-length vs reported MD + GR ribbon
    # ═══════════════════════════════════════════════════════════
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), dpi=140, sharex=False)
    fig.suptitle(
        f"3 · MD depends on geometry — well {primary['well_id']}",
        fontweight="bold",
    )
    ax = axes[0]
    ax.plot(primary["MD"], pm["arc_length"], color="#58a6ff", lw=2)
    ax.plot(
        primary["MD"],
        primary["MD"] - primary["MD"][0],
        color="#8b949e",
        ls="--",
        lw=1,
        label="MD−MD0 (index)",
    )
    ax.set_ylabel("arc length from XYZ")
    ax.set_title("Reported MD vs true path length (geometry-derived)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    # GR vs arc-length (geometry order)
    ax.scatter(
        pm["arc_length"],
        primary["GR"],
        c=primary["GR"],
        cmap=cmap,
        norm=norm,
        s=8,
        alpha=0.9,
    )
    ax.set_ylabel("GR")
    ax.set_title("GR along geometry arc-length (sensor on the path)")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    # unwrapped path: horizontal departure vs Z, colored by GR
    colored_line_2d(
        ax,
        pm["dhoriz_cum"],
        primary["Z"],
        primary["GR"],
        lw=3.5,
    )
    ax.set_xlabel("horizontal departure (from XYZ geometry)")
    ax.set_ylabel("Z")
    ax.set_title("Section in path coordinates · GR heatmap")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes,
        fraction=0.02,
        pad=0.02,
        label="GR",
    )
    fig.tight_layout()
    p3 = out / "03_md_geometry_gr_ribbon.png"
    fig.savefig(p3, bbox_inches="tight")
    plt.close(fig)

    # ═══════════════════════════════════════════════════════════
    # 4. "Heatmap matrix" — stations as image: rows=wells, cols=MD index, value=GR
    #    + geometry signature row (inclination / dS)
    # ═══════════════════════════════════════════════════════════
    # resample each well to fixed n cols by MD index
    n_cols = 160
    gr_mat = []
    labels = []
    for w in wells:
        gr = w["GR"].astype(float)
        idx = np.linspace(0, len(gr) - 1, n_cols)
        base = np.arange(len(gr), dtype=float)
        gr_i = np.interp(idx, base, np.where(np.isfinite(gr), gr, np.nanmedian(gr)))
        gr_mat.append(gr_i)
        labels.append(w["well_id"])
    gr_mat = np.vstack(gr_mat)

    fig, axes = plt.subplots(
        2, 1, figsize=(11, 5.5), dpi=140, gridspec_kw={"height_ratios": [3, 1.2]}
    )
    fig.suptitle(
        "4 · GR heatmap on discrete path index (geometry samples as columns)",
        fontweight="bold",
    )
    im = axes[0].imshow(
        gr_mat,
        aspect="auto",
        cmap=cmap,
        vmin=gmin,
        vmax=gmax,
        interpolation="nearest",
    )
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_yticklabels(labels)
    axes[0].set_xlabel("station index along wellbore (∝ MD / path samples)")
    axes[0].set_title("Gamma heatmapped on geometry samples")
    fig.colorbar(im, ax=axes[0], fraction=0.02, pad=0.02, label="GR")

    # primary geometry signature
    m = _path_metrics(primary["MD"], primary["X"], primary["Y"], primary["Z"])
    axes[1].plot(
        np.linspace(0, n_cols - 1, len(m["ds"])),
        m["ds"],
        color="#58a6ff",
        lw=1.5,
        label="|dXYZ| step",
    )
    axes[1].set_xlabel("station index")
    axes[1].set_ylabel("step length")
    axes[1].set_title(f"Geometry energy along {primary['well_id']} (why MD is path-dependent)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    p4 = out / "04_gr_matrix_on_path_index.png"
    fig.savefig(p4, bbox_inches="tight")
    plt.close(fig)

    # ═══════════════════════════════════════════════════════════
    # 5. Optional: if TVT exists, second heatmap for strat page vs GR on path
    # ═══════════════════════════════════════════════════════════
    p5 = None
    if primary.get("TVT") is not None and np.isfinite(primary["TVT"]).any():
        fig = plt.figure(figsize=(11, 4.8), dpi=140)
        fig.suptitle(
            "5 · Same geometry — dual heatmaps: GR (sensor) vs TVT (strat page)",
            fontweight="bold",
            y=1.02,
        )
        ax1 = fig.add_subplot(121, projection="3d")
        ax1.set_facecolor("#0d1117")
        colored_line_3d(
            ax1,
            primary["X"],
            primary["Y"],
            primary["Z"],
            primary["GR"],
            lw=3.0,
        )
        ax1.set_xlim(primary["X"].min(), primary["X"].max())
        ax1.set_ylim(primary["Y"].min(), primary["Y"].max())
        ax1.set_zlim(primary["Z"].min(), primary["Z"].max())
        ax1.set_title("Path colored by GR")
        ax1.view_init(elev=20, azim=-55)
        fig.colorbar(
            cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=ax1,
            shrink=0.5,
            label="GR",
        )

        ax2 = fig.add_subplot(122, projection="3d")
        ax2.set_facecolor("#0d1117")
        tv = primary["TVT"]
        tnorm = plt.Normalize(np.nanpercentile(tv, 2), np.nanpercentile(tv, 98))
        pts = np.array([primary["X"], primary["Y"], primary["Z"]]).T.reshape(-1, 1, 3)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        cc = 0.5 * (tv[:-1] + tv[1:])
        lc = Line3DCollection(segs, cmap="viridis", norm=tnorm, linewidths=3.0)
        lc.set_array(cc)
        ax2.add_collection(lc)
        ax2.set_xlim(primary["X"].min(), primary["X"].max())
        ax2.set_ylim(primary["Y"].min(), primary["Y"].max())
        ax2.set_zlim(primary["Z"].min(), primary["Z"].max())
        ax2.set_title("Same path colored by TVT (page on book)")
        ax2.view_init(elev=20, azim=-55)
        fig.colorbar(
            cm.ScalarMappable(norm=tnorm, cmap="viridis"),
            ax=ax2,
            shrink=0.5,
            label="TVT",
        )
        fig.tight_layout()
        p5 = out / "05_geometry_gr_vs_tvt_color.png"
        fig.savefig(p5, bbox_inches="tight")
        plt.close(fig)

    # ═══════════════════════════════════════════════════════════
    # Dashboard
    # ═══════════════════════════════════════════════════════════
    from matplotlib import image as mpimg

    panels = [p1, p2, p3, p4] + ([p5] if p5 else [])
    n = len(panels)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(13, 4.2 * rows), dpi=120)
    fig.suptitle(
        f"Geometry → GR heatmap  |  source={src}",
        fontweight="bold",
        color="#f0f6fc",
        y=0.995,
    )
    for i, path in enumerate(panels):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(mpimg.imread(path))
        ax.axis("off")
        ax.set_title(path.name, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    p0 = out / "00_dashboard_geometry_gr.png"
    fig.savefig(p0, bbox_inches="tight")
    plt.close(fig)

    (out / "README.md").write_text(
        "\n".join(
            [
                "# Geometry-first GR heatmap",
                "",
                "Model: **MD is dependent on wellbore geometry** — we represent the path,",
                "then heatmap gamma as a field on that path (sensor coloring the curve).",
                "",
                f"Source: `{src}`",
                "",
                "| File | Content |",
                "|------|---------|",
                "| `00_dashboard_geometry_gr.png` | collage |",
                "| `01_geometry_scaffold.png` | path only (plan / section / 3D) |",
                "| `02_geometry_gr_heatmap.png` | same path, GR heatmapped |",
                "| `03_md_geometry_gr_ribbon.png` | arc-length vs MD, GR on path coords |",
                "| `04_gr_matrix_on_path_index.png` | multi-well GR image on station index |",
                "| `05_geometry_gr_vs_tvt_color.png` | GR vs TVT coloring same geometry |",
                "",
                "Law: RESTRICT offline visuals.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Wrote → {out}  (source={src})")
    for p in [p0] + panels:
        if p and p.exists():
            print(f"  {p.name}  ({p.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
