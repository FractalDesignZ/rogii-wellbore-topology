#!/usr/bin/env python3
"""
Verification & validation harness — no Kaggle data required.

Builds a synthetic pad-style cohort with known ground truth, splits it into
train wells and *truly held-out* wells, then answers three questions the
repo's headline numbers depend on:

  V1  Does the pipeline run end-to-end (assemble -> GroupKFold -> submit)?
  V2  Is the reported OOF an honest estimate of held-out error?
  V3  What does picking the best pure prior by argmin actually cost?

V2 is the one that matters: every claim of the form "residual 14.54 beats
last_known 15.91" is an OOF claim, and an OOF number is only worth as much as
its agreement with data the model never saw.

    python scripts/vv_harness.py --out artifacts/vv
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FLOOR_COLS = {
    "last_known": "prior_last_tvt",
    "local_dip_ps": "prior_local_dip",
    "flat_formation": "prior_flat_tvt",
    "linear_md_extrap": "prior_extrap_tvt",
    "dual_transport_mid": "prior_mid",
    "heel_dip_stack": "prior_dip_stack",
    "particle_filter": "prior_pf_tvt",
    "pf_blend_last_known": "prior_pf_blend",
    "pf_bag": "prior_pf_bag",
    "prior_stack": "prior_stack",
    "prior_best": "prior_best",
}


def make_cohort(n_wells: int, n: int, ps_frac: float, seed: int):
    """Laterals on a pad grid, so the spatial neighbour graph is non-trivial."""
    rng = np.random.default_rng(seed)
    tw_tvt = np.linspace(50, 200, 300)
    tw_gr = 40 + 30 * np.sin(tw_tvt / 15) + rng.normal(0, 2, 300)
    typewell = pd.DataFrame({"TVT": tw_tvt, "GR": tw_gr})
    wells: dict[str, pd.DataFrame] = {}
    per_row = 6
    for w in range(n_wells):
        gx, gy = w % per_row, w // per_row
        x0, y0 = gx * 300.0, gy * 900.0
        md = np.arange(n, dtype=float) * 10.0
        z = np.concatenate(
            [np.linspace(0, -800, n // 3), np.linspace(-800, -820, n - n // 3)]
        )
        x = x0 + np.linspace(0, 4000, n) + rng.normal(0, 2, n)
        y = y0 + np.linspace(0, 120, n) + rng.normal(0, 2, n)
        # shared field-wide structure + per-well steering drift
        tvt = (
            100
            + 0.0015 * (x - x0)
            + 0.02 * gy
            + np.cumsum(rng.normal(0, 0.08, n))
            + rng.normal(0, 0.3)
        )
        gr = np.interp(tvt, tw_tvt, tw_gr) + rng.normal(0, 3, n)
        is_pred = np.zeros(n, dtype=bool)
        is_pred[int(n * ps_frac) :] = True
        wells[f"W{w:03d}"] = pd.DataFrame(
            {
                "well_id": f"W{w:03d}",
                "MD": md, "X": x, "Y": y, "Z": z, "GR": gr,
                "TVT": tvt,
                "TVT_input": np.where(is_pred, np.nan, tvt),
                "is_predict": is_pred,
            }
        )
    return wells, typewell


def write_cohort(root: Path, wells, typewell, hold_every: int) -> tuple[int, int]:
    """Kaggle shape: train wells fully labeled, held-out wells heel-only."""
    shutil.rmtree(root, ignore_errors=True)
    (root / "train").mkdir(parents=True)
    (root / "test").mkdir(parents=True)
    ids = sorted(wells)
    test_ids = {w for i, w in enumerate(ids) if i % hold_every == hold_every - 1}
    truth = []
    for w in ids:
        df = wells[w]
        split = "test" if w in test_ids else "train"
        out = df.copy()
        if split == "test":
            toe = df["is_predict"].to_numpy()
            truth.append(
                pd.DataFrame(
                    {
                        "well_id": w,
                        "station": np.arange(len(df))[toe],
                        "TVT_true": df["TVT"].to_numpy()[toe],
                    }
                )
            )
            out = out.drop(columns=["TVT"])  # toe truth hidden, heel kept
        out.to_csv(root / split / f"{w}__horizontal_well.csv", index=False)
        typewell.to_csv(root / split / f"{w}__typewell.csv", index=False)
    pd.concat(truth, ignore_index=True).to_csv(root / "holdout_truth.csv", index=False)
    return len(ids) - len(test_ids), len(test_ids)


def rmse(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2))) if m.any() else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description="RWT verification & validation harness")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "vv"))
    ap.add_argument("--wells", type=int, default=24)
    ap.add_argument("--stations", type=int, default=240)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--hold-every", type=int, default=3, help="1 in N wells held out")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = out / "cohort"

    wells, tw = make_cohort(args.wells, args.stations, 0.6, args.seed)
    n_tr, n_te = write_cohort(data, wells, tw, args.hold_every)
    print(f"[V1] cohort: {n_tr} train wells, {n_te} held-out wells -> {data}")

    run = out / "run"
    for stage, cmd in (
        ("train", [sys.executable, "-m", "rwt.train", "--data", str(data),
                   "--out", str(run), "--folds", str(args.folds)]),
        ("submit", [sys.executable, "-m", "rwt.submit", "--model", str(run / "model.joblib"),
                    "--data", str(data), "--split", "test", "--out", str(run / "sub.csv")]),
    ):
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[V1] FAIL {stage}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
            return 1
        for line in r.stdout.splitlines():
            if line.startswith(("[GATE", "[SELECTION")):
                print(f"     {line}")
    print("[V1] PASS — assemble -> GroupKFold -> Gate3 -> submit ran clean")

    report = json.loads((run / "cv_report.json").read_text())
    truth = pd.read_csv(data / "holdout_truth.csv")
    truth["id"] = truth["well_id"].astype(str) + "_" + truth["station"].astype(int).astype(str)
    sub = pd.read_csv(run / "sub.csv").merge(truth, on="id", how="inner")

    oof = float(report["oof_rmse_selected"])
    held = rmse(sub["TVT_true"], sub["tvt"])
    print(f"\n[V2] reported OOF (Gate3-selected) : {oof:.4f}")
    print(f"[V2] true held-out RMSE            : {held:.4f}")
    print(f"[V2] optimism                      : {held - oof:+.4f} ft ({held / oof:.2f}x)")
    print(f"[V2] {'PASS — OOF is not optimistic' if held <= oof else 'WARN — OOF understates held-out error'}")

    # V3: what argmin-over-floors costs, measured on wells never seen
    from rwt.submit import assemble_split_stations

    st = assemble_split_stations(data, "test")
    st["id"] = st["well_id"].astype(str) + "_" + st["station"].astype(int).astype(str)
    m = st.merge(truth[["id", "TVT_true"]], on="id", how="inner")
    y = m["TVT_true"].to_numpy(float)
    base = report.get("baseline_rmse", {})
    rows = [
        (n, base[n], rmse(y, m[c].to_numpy(float)))
        for n, c in FLOOR_COLS.items()
        if n in base and c in m.columns
    ]
    rows.sort(key=lambda r: r[1])
    print(f"\n[V3] {'floor':<22}{'OOF (selected on)':>19}{'held-out':>12}{'gap':>9}")
    for n, o, t in rows:
        print(f"     {n:<22}{o:>19.4f}{t:>12.4f}{t - o:>9.4f}")
    if rows:
        win = rows[0]
        best = min(rows, key=lambda r: r[2])
        print(f"\n[V3] argmin-by-OOF pick : {win[0]} — reported {win[1]:.4f}, true {win[2]:.4f}")
        print(f"[V3] actually best      : {best[0]} at {best[2]:.4f}")
        print(f"[V3] selection regret   : {win[2] - best[2]:+.4f} ft")
    audit = report.get("selection_audit", {})
    if audit.get("ok"):
        print(f"[V3] in-run nested check: naive {audit['oof_rmse_naive_argmin']:.4f} "
              f"-> nested {audit['oof_rmse_nested']:.4f} "
              f"(optimism {audit['selection_optimism']:+.4f}, "
              f"stable_pick={audit['stable_pick']})")

    summary = {
        "cohort": {"train_wells": n_tr, "holdout_wells": n_te, "seed": args.seed},
        "v2_oof_vs_holdout": {
            "oof_rmse_selected": oof,
            "holdout_rmse": held,
            "optimism_ft": held - oof,
            "oof_is_honest": bool(held <= oof),
        },
        "v3_floor_table": [
            {"floor": n, "oof": o, "holdout": t, "gap": t - o} for n, o, t in rows
        ],
        "v3_selection_audit": audit,
        "gate3": report["gates"][2],
    }
    (out / "vv_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote {out / 'vv_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
