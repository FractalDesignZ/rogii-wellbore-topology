#!/usr/bin/env python3
"""
Slice the real train matrix into dimensional views + heatmaps.

  python scripts/slice_real_field.py --data "C:/Users/.../train" --max-wells 24 --out artifacts/field_slices

STEP 1 geometry (X,Y,Z) + MD index
STEP 2 GR on path + TVT labels + typewell book + tops (train-only EDA)

RESTRICT: offline measure/visuals only.
"""
from __future__ import annotations

import argparse
import json
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
            "font.size": 9,
        }
    )


def _iter_laterals(data: Path, max_wells: int):
    paths = sorted(data.glob("*horizontal*.csv")) + sorted(
        data.glob("*horizontal_well*.csv")
    )
    # de-dupe
    seen = set()
    out = []
    for p in paths:
        if p.name in seen:
            continue
        seen.add(p.name)
        out.append(p)
    if max_wells:
        out = out[: int(max_wells)]
    return out


def _wid(path: Path) -> str:
    stem = path.stem
    for suf in ("__horizontal_well_TRAIN", "__horizontal_well", "_TRAIN"):
        if stem.endswith(suf):
            return stem[: -len(suf)]
    return stem.split("__")[0]


def _typewell_path(data: Path, wid: str) -> Path | None:
    for name in (
        f"{wid}__typewell_TRAIN.csv",
        f"{wid}__typewell.csv",
        f"{wid}_typewell.csv",
    ):
        p = data / name
        if p.is_file():
            return p
    return None


def main(argv: list[str] | None = None) -> int:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib import cm
    from matplotlib.collections import LineCollection

    from rwt.load import GR_ABSURD_LE, normalize_curve_nulls

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        default=str(
            Path.home()
            / "Documents"
            / "rogii-wellbore-geology-prediction-original"
            / "train"
        ),
    )
    ap.add_argument("--max-wells", type=int, default=24)
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "field_slices"))
    a = ap.parse_args(argv)

    data = Path(a.data)
    if not data.is_dir():
        # fallback repo data
        alt = ROOT / "data" / "raw" / "train"
        if alt.is_dir():
            data = alt
        else:
            print(f"no data dir: {a.data}", flush=True)
            return 2

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    _style()

    laterals = _iter_laterals(data, a.max_wells)
    wells = []
    for p in laterals:
        df = pd.read_csv(p)
        wid = _wid(p)
        gr = (
            normalize_curve_nulls(df["GR"].to_numpy(dtype=float), absurd_le=GR_ABSURD_LE)
            if "GR" in df.columns
            else np.full(len(df), np.nan)
        )
        wells.append(
            {
                "well_id": wid,
                "MD": df["MD"].to_numpy(dtype=float),
                "X": df["X"].to_numpy(dtype=float),
                "Y": df["Y"].to_numpy(dtype=float),
                "Z": df["Z"].to_numpy(dtype=float),
                "GR": gr,
                "TVT": df["TVT"].to_numpy(dtype=float)
                if "TVT" in df.columns
                else np.full(len(df), np.nan),
                "TVT_input": df["TVT_input"].to_numpy(dtype=float)
                if "TVT_input" in df.columns
                else np.full(len(df), np.nan),
                "raw": df,
            }
        )

    # global scales
    gr_all = np.concatenate([w["GR"] for w in wells])
    gr_f = gr_all[np.isfinite(gr_all)]
    gmin, gmax = (
        (float(np.percentile(gr_f, 2)), float(np.percentile(gr_f, 98)))
        if len(gr_f)
        else (0.0, 100.0)
    )
    if gmax <= gmin:
        gmax = gmin + 1.0
    gnorm = plt.Normalize(gmin, gmax)
    gcmap = cm.inferno

    tvt_all = np.concatenate([w["TVT"] for w in wells])
    tvt_f = tvt_all[np.isfinite(tvt_all)]
    tmin, tmax = (
        (float(np.percentile(tvt_f, 2)), float(np.percentile(tvt_f, 98)))
        if len(tvt_f)
        else (0.0, 1.0)
    )
    tnorm = plt.Normalize(tmin, tmax)

    def color_line(ax, x, y, c, cmap, norm, lw=2.0):
        m = np.isfinite(x) & np.isfinite(y) & np.isfinite(c)
        if m.sum() < 2:
            return None
        x, y, c = x[m], y[m], c[m]
        pts = np.column_stack([x, y]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        cc = 0.5 * (c[:-1] + c[1:])
        lc = LineCollection(segs, cmap=cmap, norm=norm, linewidths=lw)
        lc.set_array(cc)
        ax.add_collection(lc)
        return lc

    # ── SLICE A: pad map geometry ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8), dpi=140)
    for w in wells:
        ax.plot(w["X"], w["Y"], color="#30363d", lw=0.8, alpha=0.5, zorder=1)
        color_line(ax, w["X"], w["Y"], w["GR"], gcmap, gnorm, lw=2.2)
    # heel points
    for w in wells:
        hi = np.isfinite(w["TVT_input"])
        if hi.any():
            i0 = int(np.flatnonzero(hi)[0])
            ax.scatter(w["X"][i0], w["Y"][i0], s=12, c="#58a6ff", zorder=5)
    sm = cm.ScalarMappable(norm=gnorm, cmap=gcmap)
    fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, label="GR")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(
        f"SLICE A · Pad map — geometry paths heatmapped by GR  (n={len(wells)} wells)"
    )
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    pa = out / "A_pad_xy_gr.png"
    fig.savefig(pa, bbox_inches="tight")
    plt.close(fig)

    # ── SLICE B: section X–Z GR ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6), dpi=140)
    for w in wells:
        color_line(ax, w["X"], w["Z"], w["GR"], gcmap, gnorm, lw=1.8)
    fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, label="GR")
    ax.set_xlabel("X")
    ax.set_ylabel("Z (vertical / SS-like)")
    ax.set_title("SLICE B · Vertical section — path geometry + GR")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    pb = out / "B_section_xz_gr.png"
    fig.savefig(pb, bbox_inches="tight")
    plt.close(fig)

    # ── SLICE C: multi-well GR matrix (path index) ───────────────────
    n_cols = 200
    mat = []
    labs = []
    for w in wells:
        gr = w["GR"].astype(float)
        if not np.isfinite(gr).any():
            continue
        fill = float(np.nanmedian(gr))
        g = np.where(np.isfinite(gr), gr, fill)
        idx = np.linspace(0, len(g) - 1, n_cols)
        mat.append(np.interp(idx, np.arange(len(g)), g))
        labs.append(w["well_id"][:10])
    mat = np.vstack(mat) if mat else np.zeros((1, n_cols))
    fig, ax = plt.subplots(figsize=(12, max(4, 0.28 * len(labs) + 2)), dpi=140)
    im = ax.imshow(mat, aspect="auto", cmap=gcmap, vmin=gmin, vmax=gmax)
    ax.set_yticks(range(len(labs)))
    ax.set_yticklabels(labs, fontsize=7)
    ax.set_xlabel("station index along well (∝ MD / path sample)")
    ax.set_title("SLICE C · GR field as matrix  [well × path index]")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="GR")
    fig.tight_layout()
    pc = out / "C_gr_matrix.png"
    fig.savefig(pc, bbox_inches="tight")
    plt.close(fig)

    # ── SLICE D: TVT matrix (labels) ─────────────────────────────────
    mat_t = []
    labs_t = []
    for w in wells:
        tv = w["TVT"].astype(float)
        if not np.isfinite(tv).any():
            continue
        fill = float(np.nanmedian(tv))
        t = np.where(np.isfinite(tv), tv, fill)
        idx = np.linspace(0, len(t) - 1, n_cols)
        mat_t.append(np.interp(idx, np.arange(len(t)), t))
        labs_t.append(w["well_id"][:10])
    mat_t = np.vstack(mat_t) if mat_t else np.zeros((1, n_cols))
    fig, ax = plt.subplots(figsize=(12, max(4, 0.28 * len(labs_t) + 2)), dpi=140)
    im = ax.imshow(mat_t, aspect="auto", cmap="viridis", vmin=tmin, vmax=tmax)
    ax.set_yticks(range(len(labs_t)))
    ax.set_yticklabels(labs_t, fontsize=7)
    ax.set_xlabel("station index along well")
    ax.set_title("SLICE D · TVT labels matrix  [well × path]  (train answers)")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="TVT")
    fig.tight_layout()
    pd_ = out / "D_tvt_matrix.png"
    fig.savefig(pd_, bbox_inches="tight")
    plt.close(fig)

    # ── SLICE E: heel vs toe mask ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=140)
    for w in wells:
        heel = np.isfinite(w["TVT_input"])
        toe = ~heel
        axes[0].plot(w["X"][heel], w["Y"][heel], color="#3fb950", lw=1.2, alpha=0.7)
        axes[1].plot(w["X"][toe], w["Y"][toe], color="#f85149", lw=1.2, alpha=0.7)
    axes[0].set_title("SLICE E1 · HEEL (TVT_input known)")
    axes[1].set_title("SLICE E2 · TOE (predict zone — answers hard)")
    for ax in axes:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.25)
    fig.suptitle("Heel vs toe slices on the pad", fontweight="bold")
    fig.tight_layout()
    pe = out / "E_heel_vs_toe.png"
    fig.savefig(pe, bbox_inches="tight")
    plt.close(fig)

    # ── SLICE F: typewell books (first 8) ────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), dpi=130)
    axes = axes.ravel()
    n_tw = 0
    for w in wells:
        if n_tw >= 8:
            break
        tp = _typewell_path(data, w["well_id"])
        if tp is None:
            continue
        tw = pd.read_csv(tp)
        if "TVT" not in tw.columns or "GR" not in tw.columns:
            continue
        ax = axes[n_tw]
        ax.plot(tw["TVT"], tw["GR"], color="#58a6ff", lw=1.2)
        # overlay lateral GR vs TVT where both finite
        m = np.isfinite(w["GR"]) & np.isfinite(w["TVT"])
        if m.sum() > 20:
            ax.scatter(
                w["TVT"][m][:: max(1, m.sum() // 80)],
                w["GR"][m][:: max(1, m.sum() // 80)],
                s=4,
                c="#d29922",
                alpha=0.5,
            )
        ax.set_title(w["well_id"][:12], fontsize=8)
        ax.grid(True, alpha=0.25)
        n_tw += 1
    for j in range(n_tw, 8):
        axes[j].axis("off")
    fig.suptitle(
        "SLICE F · Typewell book G_tw(TVT)  |  orange = lateral GR@TVT (train)",
        fontweight="bold",
    )
    fig.tight_layout()
    pf = out / "F_typewell_books.png"
    fig.savefig(pf, bbox_inches="tight")
    plt.close(fig)

    # ── SLICE G: tops (train-only) mean surface depth per well ───────
    tops = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
    top_rows = []
    for w in wells:
        df = w["raw"]
        row = {"well_id": w["well_id"], "X": float(np.nanmean(w["X"])), "Y": float(np.nanmean(w["Y"]))}
        for t in tops:
            if t in df.columns:
                arr = df[t].to_numpy(dtype=float)
                row[t] = float(np.nanmedian(arr)) if np.isfinite(arr).any() else float("nan")
        top_rows.append(row)
    T = pd.DataFrame(top_rows)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), dpi=130)
    for ax, t in zip(axes.ravel(), tops):
        if t not in T.columns:
            ax.axis("off")
            continue
        sc = ax.scatter(T["X"], T["Y"], c=T[t], cmap="coolwarm", s=40)
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(t)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True, alpha=0.25)
    fig.suptitle(
        "SLICE G · Formation tops (train-only) — pad map of median top Z  [NOT serve features]",
        fontweight="bold",
    )
    fig.tight_layout()
    pg = out / "G_tops_pad.png"
    fig.savefig(pg, bbox_inches="tight")
    plt.close(fig)

    # ── SLICE H: single-well full strip ──────────────────────────────
    w0 = wells[0]
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), dpi=140, sharex=True)
    md = w0["MD"]
    axes[0].plot(md, w0["Z"], color="#8b949e", lw=1.2)
    axes[0].set_ylabel("Z")
    axes[0].set_title(f"SLICE H · Single well strip {w0['well_id']} — geometry + fields")
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(md, w0["GR"], c=w0["GR"], cmap=gcmap, norm=gnorm, s=6)
    axes[1].set_ylabel("GR")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(md, w0["TVT"], color="#3fb950", lw=1.2, label="TVT")
    ti = w0["TVT_input"]
    axes[2].plot(md, ti, color="#58a6ff", lw=1.0, ls="--", label="TVT_input")
    axes[2].legend(fontsize=8)
    axes[2].set_ylabel("TVT")
    axes[2].grid(True, alpha=0.3)

    df0 = w0["raw"]
    for t, col in zip(tops, cm.tab10(np.linspace(0, 1, len(tops)))):
        if t in df0.columns:
            axes[3].plot(md, df0[t].to_numpy(dtype=float), lw=1.0, label=t, color=col)
    axes[3].legend(fontsize=7, ncol=3)
    axes[3].set_ylabel("top Z")
    axes[3].set_xlabel("MD")
    axes[3].invert_yaxis()
    axes[3].grid(True, alpha=0.3)
    fig.tight_layout()
    ph = out / "H_single_well_strip.png"
    fig.savefig(ph, bbox_inches="tight")
    plt.close(fig)

    # ── Dashboard ────────────────────────────────────────────────────
    from matplotlib import image as mpimg

    panels = [pa, pb, pc, pd_, pe, pf, pg, ph]
    titles = [
        "A pad XY+GR",
        "B section XZ+GR",
        "C GR matrix",
        "D TVT matrix",
        "E heel vs toe",
        "F typewell books",
        "G tops (train)",
        "H single strip",
    ]
    fig = plt.figure(figsize=(16, 14), dpi=120)
    fig.suptitle(
        f"REAL FIELD SLICED  ·  {data}  ·  n_wells={len(wells)}",
        fontweight="bold",
        color="#f0f6fc",
        y=0.995,
    )
    for i, (path, title) in enumerate(zip(panels, titles)):
        ax = fig.add_subplot(4, 2, i + 1)
        ax.imshow(mpimg.imread(path))
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    p0 = out / "00_DASHBOARD_slices.png"
    fig.savefig(p0, bbox_inches="tight")
    plt.close(fig)

    # inventory JSON
    inv = {
        "data": str(data),
        "n_wells_sliced": len(wells),
        "well_ids": [w["well_id"] for w in wells],
        "gr_finite_frac": float(np.isfinite(gr_all).mean()),
        "tvt_input_finite_frac": float(
            np.isfinite(np.concatenate([w["TVT_input"] for w in wells])).mean()
        ),
        "slices": [p.name for p in panels],
        "law": "RESTRICT offline; tops train-only EDA; core matrix (X,Y,Z):GR + TVT labels",
        "two_step": "1 geometry path  2 GR paint + TVT place on book",
    }
    (out / "inventory.json").write_text(json.dumps(inv, indent=2) + "\n", encoding="utf-8")
    (out / "README.md").write_text(
        "\n".join(
            [
                "# Real field slices",
                "",
                "Dimensional slices of the **real** train matrix.",
                "",
                "| Slice | File | Axis |",
                "|-------|------|------|",
                "| A | `A_pad_xy_gr.png` | pad map XY + GR heatmap |",
                "| B | `B_section_xz_gr.png` | vertical section XZ + GR |",
                "| C | `C_gr_matrix.png` | well × path index · GR |",
                "| D | `D_tvt_matrix.png` | well × path · TVT labels |",
                "| E | `E_heel_vs_toe.png` | heel known vs toe predict |",
                "| F | `F_typewell_books.png` | G_tw books + lateral overlay |",
                "| G | `G_tops_pad.png` | tops (train-only) |",
                "| H | `H_single_well_strip.png` | one well full strip |",
                "| 00 | `00_DASHBOARD_slices.png` | all |",
                "",
                f"n_wells={len(wells)} data=`{data}`",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(json.dumps(inv, indent=2), flush=True)
    print(f"SLICED → {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
