#!/usr/bin/env python3
"""
Visualize the GR@TVT pad manifold (4D construct: X, Y, TVT, GR).

  python scripts/render_manifold_gr_visual.py --out artifacts/manifold_gr_synth/visuals

Panels written as PNGs + one multi-panel dashboard.
RESTRICT: synthetic offline only — no Kaggle submit.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _style():
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
            "axes.titlesize": 12,
            "figure.titlesize": 14,
        }
    )


def main(argv: list[str] | None = None) -> int:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    from rwt.load import apply_curve_nulls_frame
    from rwt.pad_synth import (
        decode_gr_nulls,
        last_known_series,
        make_pad_laterals,
        neighbor_idw_tvt,
        tvt_field_xy,
    )

    ap = argparse.ArgumentParser(description="Render GR@TVT pad manifold visuals")
    ap.add_argument("--n-wells", type=int, default=6)
    ap.add_argument("--n-stations", type=int, default=180)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--out",
        default=str(ROOT / "artifacts" / "manifold_gr_synth" / "visuals"),
    )
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    _style()

    pad = make_pad_laterals(
        n_wells=int(a.n_wells),
        n_stations=int(a.n_stations),
        seed=int(a.seed),
        gr_noise=2.5,
    )
    tw = pad["typewell"]
    wells = [apply_curve_nulls_frame(w) for w in pad["wells"]]
    train_ids = set(pad["train_ids"])
    donors = [w for w in wells if str(w["well_id"].iloc[0]) in train_ids]
    fp = pad["field_params"]

    # shared color maps
    all_tvt = np.concatenate([w["TVT_true"].to_numpy(dtype=float) for w in wells])
    all_gr = np.concatenate(
        [decode_gr_nulls(w["GR"].to_numpy(dtype=float)) for w in wells]
    )
    tvt_lo, tvt_hi = float(np.nanpercentile(all_tvt, 2)), float(
        np.nanpercentile(all_tvt, 98)
    )
    gr_lo, gr_hi = float(np.nanpercentile(all_gr, 2)), float(
        np.nanpercentile(all_gr, 98)
    )
    colors = cm.turbo(np.linspace(0.1, 0.9, len(wells)))

    # ── 1. Typewell book ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5), dpi=140)
    ax.plot(tw["TVT"], tw["GR"], color="#58a6ff", lw=2.0, label=r"$G_{\mathrm{tw}}(\mathrm{TVT})$")
    ax.fill_between(tw["TVT"], tw["GR"], alpha=0.15, color="#58a6ff")
    ax.set_xlabel("TVT (stratigraphic page)")
    ax.set_ylabel("GR (API)")
    ax.set_title("1 · Formation book — GR @ TVT tensor")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    p1 = out / "01_typewell_book.png"
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)

    # ── 2. TVT field on XY + lateral tracks ───────────────────────────
    xs = np.linspace(0, 3600, 120)
    ys = np.linspace(-50, 200.0 * (len(wells) - 1) + 80, 100)
    XX, YY = np.meshgrid(xs, ys)
    ZZ = tvt_field_xy(
        XX, YY, t0=float(fp["t0"]), gx=float(fp["gx"]), gy=float(fp["gy"])
    )
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=140)
    im = ax.pcolormesh(XX, YY, ZZ, shading="auto", cmap="viridis", alpha=0.95)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("TVT field")
    for i, w in enumerate(wells):
        role = str(w["pad_role"].iloc[0])
        ls = "-" if role == "train" else "--"
        ax.plot(
            w["X"],
            w["Y"],
            color=colors[i],
            lw=2.2,
            ls=ls,
            label=f"{w['well_id'].iloc[0]} ({role})",
        )
        # heel marker
        heel = ~w["is_predict"].to_numpy()
        if heel.any():
            ax.scatter(
                w.loc[heel, "X"].iloc[-1],
                w.loc[heel, "Y"].iloc[-1],
                s=40,
                c=[colors[i]],
                edgecolors="white",
                zorder=5,
            )
    ax.set_xlabel("X (ft)")
    ax.set_ylabel("Y (ft)")
    ax.set_title("2 · Pad manifold — TVT(X,Y) + laterals (solid=train, dashed=test)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    p2 = out / "02_pad_tvt_field.png"
    fig.savefig(p2, bbox_inches="tight")
    plt.close(fig)

    # ── 3. 3D: XYZ paths colored by TVT (space + stratigraphy) ───────
    fig = plt.figure(figsize=(9, 7), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0d1117")
    for i, w in enumerate(wells):
        tv = w["TVT_true"].to_numpy(dtype=float)
        sc = ax.scatter(
            w["X"],
            w["Y"],
            w["Z"],
            c=tv,
            cmap="viridis",
            s=6,
            vmin=tvt_lo,
            vmax=tvt_hi,
            alpha=0.9,
        )
        ax.plot(w["X"], w["Y"], w["Z"], color=colors[i], lw=0.8, alpha=0.5)
    fig.colorbar(sc, ax=ax, shrink=0.55, label="TVT_true")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3 · Laterals in space — color = stratigraphic TVT")
    ax.view_init(elev=22, azim=-55)
    fig.tight_layout()
    p3 = out / "03_laterals_xyz_tvt.png"
    fig.savefig(p3, bbox_inches="tight")
    plt.close(fig)

    # ── 4. 4D projection: GR vs TVT (book vs observations) ───────────
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=140)
    ax.plot(
        tw["TVT"],
        tw["GR"],
        color="#f0f6fc",
        lw=2.5,
        alpha=0.9,
        label="book G_tw",
        zorder=10,
    )
    for i, w in enumerate(wells):
        gr = decode_gr_nulls(w["GR"].to_numpy(dtype=float))
        tv = w["TVT_true"].to_numpy(dtype=float)
        m = np.isfinite(gr)
        ax.scatter(
            tv[m],
            gr[m],
            s=8,
            alpha=0.35,
            c=[colors[i]],
            label=str(w["well_id"].iloc[0]),
        )
    ax.set_xlabel("TVT_true (page)")
    ax.set_ylabel("GR observed")
    ax.set_title("4 · 4D slice: observations live on the GR@TVT book")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=7, ncol=2)
    fig.tight_layout()
    p4 = out / "04_gr_vs_tvt_book.png"
    fig.savefig(p4, bbox_inches="tight")
    plt.close(fig)

    # ── 5. Along-MD: TVT true vs last vs neighbor (one test well) ────
    test_wells = [w for w in wells if str(w["pad_role"].iloc[0]) == "test"]
    w = test_wells[0] if test_wells else wells[-1]
    wid = str(w["well_id"].iloc[0])
    md = w["MD"].to_numpy(dtype=float)
    y = w["TVT_true"].to_numpy(dtype=float)
    last = last_known_series(w)
    nb = neighbor_idw_tvt(w, donors)
    nb_f = np.where(np.isfinite(nb), nb, last)
    ps = int(fp["ps"])

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), dpi=140, sharex=True)
    ax0, ax1 = axes
    ax0.plot(md, y, color="#3fb950", lw=2, label="TVT_true")
    ax0.plot(md, last, color="#d29922", lw=1.5, ls="--", label="last_known")
    ax0.plot(md, nb_f, color="#58a6ff", lw=1.5, label="neighbor IDW")
    ax0.axvline(ps, color="#f85149", ls=":", lw=1.2, label="PS (predict start)")
    ax0.set_ylabel("TVT")
    ax0.set_title(f"5a · Placement on {wid} (test map — toe labels withheld)")
    ax0.legend(loc="best", fontsize=8)
    ax0.grid(True, alpha=0.3)

    gr = decode_gr_nulls(w["GR"].to_numpy(dtype=float))
    # book GR at true vs last
    from rwt.gr_match import typewell_gr_at_tvt

    heel = ~w["is_predict"].to_numpy()
    known = np.where(heel, w["TVT_input"].to_numpy(dtype=float), np.nan)
    gr_true = typewell_gr_at_tvt(
        tw["GR"].to_numpy(),
        tw["TVT"].to_numpy(),
        y,
        lateral_gr=gr,
        known_tvt=known,
    )
    gr_last = typewell_gr_at_tvt(
        tw["GR"].to_numpy(),
        tw["TVT"].to_numpy(),
        last,
        lateral_gr=gr,
        known_tvt=known,
    )
    ax1.plot(md, gr, color="#e6edf3", lw=1.2, alpha=0.85, label="GR lateral")
    ax1.plot(md, gr_true, color="#3fb950", lw=1.4, label="book@TVT_true")
    ax1.plot(md, gr_last, color="#d29922", lw=1.2, ls="--", label="book@last_known")
    ax1.axvline(ps, color="#f85149", ls=":", lw=1.2)
    ax1.set_xlabel("MD (ft)")
    ax1.set_ylabel("GR")
    ax1.set_title("5b · GR corroboration — book agrees at true page, less at last_known")
    ax1.legend(loc="best", fontsize=8)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    p5 = out / "05_placement_and_gr_corroboration.png"
    fig.savefig(p5, bbox_inches="tight")
    plt.close(fig)

    # ── 6. 3D surface TVT(X,Y) with wells projected ──────────────────
    fig = plt.figure(figsize=(9, 7), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0d1117")
    # downsample surface
    step = 3
    surf = ax.plot_surface(
        XX[::step, ::step],
        YY[::step, ::step],
        ZZ[::step, ::step],
        cmap="viridis",
        alpha=0.55,
        linewidth=0,
        antialiased=True,
    )
    for i, w in enumerate(wells):
        tv = w["TVT_true"].to_numpy(dtype=float)
        ax.plot(w["X"], w["Y"], tv, color=colors[i], lw=2.0, label=str(w["well_id"].iloc[0]))
    fig.colorbar(surf, ax=ax, shrink=0.5, label="TVT field")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("TVT")
    ax.set_title("6 · Stratigraphic surface TVT(X,Y) — laterals ride the manifold")
    ax.view_init(elev=28, azim=-60)
    fig.tight_layout()
    p6 = out / "06_tvt_surface_3d.png"
    fig.savefig(p6, bbox_inches="tight")
    plt.close(fig)

    # ── 7. Dashboard collage (2×3) ───────────────────────────────────
    fig = plt.figure(figsize=(14, 10), dpi=130)
    fig.suptitle(
        "GR@TVT pad manifold — 4D construct (X, Y, TVT, GR)",
        color="#f0f6fc",
        fontweight="bold",
        y=0.98,
    )

    # reuse images
    from matplotlib import image as mpimg

    panels = [p1, p2, p3, p4, p5, p6]
    titles = [
        "Book G_tw",
        "Pad TVT field",
        "XYZ + TVT color",
        "GR vs TVT",
        "Placement + GR",
        "TVT surface",
    ]
    for i, (path, title) in enumerate(zip(panels, titles)):
        ax = fig.add_subplot(2, 3, i + 1)
        ax.imshow(mpimg.imread(path))
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p7 = out / "00_dashboard_4d_manifold.png"
    fig.savefig(p7, bbox_inches="tight")
    plt.close(fig)

    # index markdown
    readme = out / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# GR@TVT pad manifold visuals",
                "",
                "4D construct: **(X, Y, TVT, GR)** — geology as book, pad as field, laterals as paths.",
                "",
                "| # | File | What you see |",
                "|---|------|--------------|",
                "| 0 | `00_dashboard_4d_manifold.png` | All panels |",
                "| 1 | `01_typewell_book.png` | Formation book GR@TVT |",
                "| 2 | `02_pad_tvt_field.png` | TVT(X,Y) + train/test laterals |",
                "| 3 | `03_laterals_xyz_tvt.png` | 3D wellbores colored by TVT |",
                "| 4 | `04_gr_vs_tvt_book.png` | Observations on the book |",
                "| 5 | `05_placement_and_gr_corroboration.png` | last vs neighbor vs true + GR |",
                "| 6 | `06_tvt_surface_3d.png` | Strat surface with laterals |",
                "",
                f"Seed={a.seed} wells={a.n_wells} stations={a.n_stations}",
                "",
                "Law: RESTRICT synthetic offline only.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Wrote visuals → {out}")
    for p in [p7, p1, p2, p3, p4, p5, p6]:
        print(f"  {p.name}  ({p.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
