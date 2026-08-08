#!/usr/bin/env python3
"""
Private lattice pilot MEASURE — Viterbi TVT prior RMSE vs last_known / public floors.

  set RWT_KERNEL=private
  python scripts/measure_private_lattice_pilot.py --data data/raw --max-wells 40

RESTRICT: no Kaggle submit. Local artifacts only.
private_kernel/* is gitignored; public CI fails closed without it.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

os.environ.setdefault("RWT_KERNEL", "private")


def main() -> int:
    import argparse

    import numpy as np
    import pandas as pd

    from rwt.baselines import rmse
    from rwt.kernel_api import resolve_kernel
    from rwt.load import iter_well_frames, load_typewell, typewell_path_for_well
    from rwt.middle_out import resolve_middle_out
    from rwt.priors import blend_stack_priors, compute_heel_fitted_dip_prior

    ap = argparse.ArgumentParser(description="Private lattice pure-prior pilot")
    ap.add_argument("--data", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--max-wells", type=int, default=40)
    ap.add_argument("--n-states", type=int, default=96)
    ap.add_argument(
        "--out", default=str(ROOT / "artifacts" / "private_lattice_pilot")
    )
    a = ap.parse_args()

    try:
        from private_kernel.beam_match import beam_blend_last, beam_prior_for_well
        from private_kernel.lattice_solver import lattice_prior_for_well
    except ImportError as e:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": f"private_kernel import failed: {e}",
                    "hint": "Local only: implement private_kernel/ (gitignored)",
                },
                indent=2,
            ),
            flush=True,
        )
        return 2

    k = resolve_kernel()
    print(
        json.dumps({"kernel_id": k.kernel_id(), "private": k.is_private()}),
        flush=True,
    )
    if not k.is_private():
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "kernel not private (RWT_KERNEL / import fail-closed)",
                    "kernel_id": k.kernel_id(),
                },
                indent=2,
            ),
            flush=True,
        )
        return 2

    root = Path(a.data)
    rows = []
    for i, (wid, df) in enumerate(iter_well_frames(root, "train")):
        sp = resolve_middle_out(df, well_id=wid)
        if sp.n_predict < 20:
            continue
        tw_path = typewell_path_for_well(root, wid, "train")
        tw = load_typewell(tw_path) if tw_path else None
        lat = lattice_prior_for_well(
            df, tw, sp, n_states=int(a.n_states), window=32
        )
        beam = beam_prior_for_well(df, tw, sp, beam_width=max(24, a.n_states // 2))
        pred = np.asarray(sp.predict_idx, dtype=int)
        y = (
            df["TVT"].to_numpy(dtype=float)
            if "TVT" in df.columns
            else np.full(len(df), np.nan)
        )
        # last-known series
        last = np.full(len(df), np.nan)
        if "TVT_input" in df.columns:
            ti = df["TVT_input"].to_numpy(dtype=float)
            fill = np.nan
            for j in range(len(df)):
                if np.isfinite(ti[j]):
                    fill = float(ti[j])
                last[j] = fill
        blend = beam_blend_last(
            beam["prior_beam_tvt"], last, beam["prior_beam_conf"]
        )
        m = pred[np.isfinite(y[pred]) & np.isfinite(blend[pred])]
        if len(m) < 20:
            continue
        # public stack floor (cheap, no PF) for comparison
        try:
            dip = compute_heel_fitted_dip_prior(df, sp)
            last_f = (
                float(last[np.isfinite(last)][-1]) if np.isfinite(last).any() else 0.0
            )
            stack = blend_stack_priors(
                last_f,
                dip["prior_dip_stack"],
                np.full(len(df), last_f, dtype=float),
                np.zeros(len(df), dtype=float),
            )
            r_stack = rmse(y[m], np.asarray(stack, dtype=float)[m])
        except Exception:
            r_stack = float("nan")

        lat_track = np.asarray(lat["prior_lattice_tvt"], dtype=float)
        lat_conf = np.asarray(
            lat.get("lattice_conf", np.zeros(len(lat_track))), dtype=float
        )
        r_lat = rmse(y[m], lat_track[m])
        r_blend = rmse(y[m], blend[m])
        r_lk = rmse(y[m], last[m])
        # Adaptive lattice: conf-blend (serve-like) + well-level pick (measure only)
        lat_conf_blend = beam_blend_last(lat_track, last, lat_conf)
        r_lat_conf = rmse(y[m], lat_conf_blend[m])
        # well-adaptive: pure lattice only if it beats last_known on this well
        use_lat_well = bool(np.isfinite(r_lat) and np.isfinite(r_lk) and r_lat < r_lk)
        r_lat_well = r_lat if use_lat_well else r_lk

        rows.append(
            {
                "well_id": wid,
                "n_pred": int(len(m)),
                "rmse_lattice": r_lat,
                "rmse_lattice_conf_blend": r_lat_conf,
                "rmse_lattice_well_adaptive": r_lat_well,
                "lattice_beats_last_known": use_lat_well,
                "rmse_beam_blend": r_blend,
                "rmse_last_known": r_lk,
                "rmse_public_stack_approx": r_stack,
                "lattice_ok": bool(lat.get("ok", False)),
                "n_states": int(lat.get("n_states", 0)),
                "mean_lattice_conf": float(np.nanmean(lat_conf[m])),
            }
        )
        if a.max_wells and len(rows) >= a.max_wells:
            break
        if len(rows) % 5 == 0:
            print(f"scored {len(rows)} wells…", flush=True)

    R = pd.DataFrame(rows)
    summary = {
        "ok": len(R) > 0,
        "kernel_id": k.kernel_id(),
        "engine": "viterbi_lattice",
        "n_wells": int(len(R)),
        "mean_rmse_lattice": float(R["rmse_lattice"].mean()) if len(R) else None,
        "mean_rmse_lattice_conf_blend": (
            float(R["rmse_lattice_conf_blend"].mean()) if len(R) else None
        ),
        "mean_rmse_lattice_well_adaptive": (
            float(R["rmse_lattice_well_adaptive"].mean()) if len(R) else None
        ),
        "n_wells_lattice_beats_last_known": (
            int(R["lattice_beats_last_known"].sum()) if len(R) else 0
        ),
        "mean_rmse_beam_blend": float(R["rmse_beam_blend"].mean()) if len(R) else None,
        "mean_rmse_last_known": float(R["rmse_last_known"].mean()) if len(R) else None,
        "mean_rmse_public_stack_approx": (
            float(R["rmse_public_stack_approx"].mean()) if len(R) else None
        ),
        "median_rmse_lattice": float(R["rmse_lattice"].median()) if len(R) else None,
        "beats_last_known": (
            float(R["rmse_lattice"].mean()) < float(R["rmse_last_known"].mean())
            if len(R)
            else False
        ),
        "adaptive_conf_blend_beats_last_known": (
            float(R["rmse_lattice_conf_blend"].mean())
            < float(R["rmse_last_known"].mean())
            if len(R)
            else False
        ),
        "adaptive_well_beats_last_known": (
            float(R["rmse_lattice_well_adaptive"].mean())
            < float(R["rmse_last_known"].mean())
            if len(R)
            else False
        ),
        "beats_public_stack_approx": (
            float(R["rmse_lattice"].mean())
            < float(R["rmse_public_stack_approx"].mean())
            if len(R) and np.isfinite(R["rmse_public_stack_approx"]).any()
            else False
        ),
        "public_12w_residual_band": [10.8, 11.1],
        "public_773w_residual": 15.38,
        "crown_band": [6.0, 7.0],
        "hunt_ft": 5.0,
        "note": (
            "Per-well predict-zone lattice prior RMSE — not full GroupKFold residual. "
            "Adaptive axes: conf-blend (serve-like) and well-level pick (needs y_true). "
            "Gate3 still owns residual accept on public path. Adaptive is the game."
        ),
        "law": "RESTRICT: local MEASURE only; never Kaggle submit; private_kernel gitignored",
    }
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if len(R):
        R.to_csv(out / "per_well_lattice.csv", index=False)
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
