#!/usr/bin/env python3
"""
Private kernel pilot MEASURE — beam prior RMSE vs public floors (local only).

  set RWT_KERNEL=private
  python scripts/measure_private_beam_pilot.py --max-wells 40

Never ships private_kernel. Gate3 still owns residual accept on public path;
this script reports beam prior OOF-style holdout by well for gap diagnosis.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))  # private_kernel package root

os.environ["RWT_KERNEL"] = "private"


def main() -> int:
    import argparse
    import numpy as np
    import pandas as pd

    from rwt.baselines import rmse
    from rwt.load import iter_well_frames, load_typewell, typewell_path_for_well
    from rwt.middle_out import resolve_middle_out
    from rwt.kernel_api import resolve_kernel

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--max-wells", type=int, default=40)
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "private_beam_pilot"))
    a = ap.parse_args()

    # Fail-closed imports: public clones have no private_kernel package body
    try:
        from private_kernel.beam_match import beam_blend_last, beam_prior_for_well
    except ImportError as e:
        summary = {
            "ok": False,
            "error": f"private_kernel import failed: {e}",
            "hint": "Local only: private_kernel/* is gitignored; not required for public CI",
        }
        print(json.dumps(summary, indent=2), flush=True)
        return 2

    k = resolve_kernel()
    print(json.dumps({"kernel_id": k.kernel_id(), "private": k.is_private()}), flush=True)
    if not k.is_private():
        summary = {
            "ok": False,
            "error": "private_kernel not loaded (expected if gitignored / RWT_KERNEL!=private)",
            "kernel_id": k.kernel_id(),
            "hint": "Local only: implement private_kernel/ and set RWT_KERNEL=private",
        }
        print(json.dumps(summary, indent=2), flush=True)
        return 2

    root = Path(a.data)
    rows = []
    for i, (wid, df) in enumerate(iter_well_frames(root, "train")):
        sp = resolve_middle_out(df, well_id=wid)
        if sp.n_predict < 10:
            continue
        tw_path = typewell_path_for_well(root, wid, "train")
        tw = load_typewell(tw_path) if tw_path else None
        beam = beam_prior_for_well(df, tw, sp, beam_width=48)
        pred = np.asarray(sp.predict_idx, dtype=int)
        y = df["TVT"].to_numpy(float) if "TVT" in df.columns else np.full(len(df), np.nan)
        # last known series
        last = np.full(len(df), np.nan)
        if "TVT_input" in df.columns:
            ti = df["TVT_input"].to_numpy(float)
            fill = np.nan
            for j in range(len(df)):
                if np.isfinite(ti[j]):
                    fill = ti[j]
                last[j] = fill
        blend = beam_blend_last(beam["prior_beam_tvt"], last, beam["prior_beam_conf"])
        # One comparable cohort: finite truth, blend, and last-known on every scored station.
        # Non-comparable wells are skipped so win-rate / delta denominators stay honest.
        m = pred[
            np.isfinite(y[pred])
            & np.isfinite(blend[pred])
            & np.isfinite(last[pred])
            & np.isfinite(beam["prior_beam_tvt"][pred])
        ]
        if len(m) < 20:
            continue
        r_beam = rmse(y[m], beam["prior_beam_tvt"][m])
        r_blend = rmse(y[m], blend[m])
        r_last = rmse(y[m], last[m])
        conf_m = beam["prior_beam_conf"][m]
        rows.append(
            {
                "well_id": wid,
                "n_pred": len(m),
                "rmse_beam": r_beam,
                "rmse_beam_blend": r_blend,
                "rmse_last_known": r_last,
                "mean_conf": float(np.nanmean(conf_m)),
                "frac_conf_ge_0_5": float(np.mean(conf_m >= 0.5)),
                "beats_last_known": bool(r_blend < r_last),
            }
        )
        if a.max_wells and len(rows) >= a.max_wells:
            break
        if len(rows) % 10 == 0:
            print(f"scored {len(rows)} wells…", flush=True)

    R = pd.DataFrame(rows)
    # Rows already require finite last-known; keep a named cohort for clarity / future filters.
    C = R.dropna(subset=["rmse_last_known", "rmse_beam_blend", "rmse_beam"]) if len(R) else R
    summary = {
        "ok": len(C) > 0,
        "kernel_id": k.kernel_id(),
        "n_wells": len(C),
        "n_wells_scored_raw": len(R),
        "comparable_cohort": "finite y + blend + last + beam on predict mask",
        "mean_rmse_beam": float(C["rmse_beam"].mean()) if len(C) else None,
        "mean_rmse_beam_blend": float(C["rmse_beam_blend"].mean()) if len(C) else None,
        "mean_rmse_last_known": float(C["rmse_last_known"].mean()) if len(C) else None,
        "median_rmse_beam_blend": float(C["rmse_beam_blend"].median()) if len(C) else None,
        "beats_last_known_mean": (
            float(C["rmse_beam_blend"].mean()) < float(C["rmse_last_known"].mean())
            if len(C)
            else False
        ),
        "frac_wells_beat_last_known": (
            float(C["beats_last_known"].mean()) if len(C) and "beats_last_known" in C else None
        ),
        "mean_conf": float(C["mean_conf"].mean()) if len(C) and "mean_conf" in C else None,
        "delta_blend_vs_last_known": (
            float(C["rmse_beam_blend"].mean() - C["rmse_last_known"].mean()) if len(C) else None
        ),
        "prior_pilot_mean_blend": 11.923837695178506,
        "prior_pilot_mean_last_known": 10.788491868712727,
        "public_full_cohort_selected": 14.67,
        "crown_band": [6.0, 7.0],
        "hunt_ft": 5.0,
        "stalk": "v2 freeze clamp_origin + GR-rel conf + NCC reseed + reversion blend",
        "note": "Per-well predict-zone RMSE of pure beam prior (not full GroupKFold residual stack)",
        "law": "private offline scaffold; export predictions only; Gate3 still owns residual accept",
    }
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    C.to_csv(out / "per_well_beam.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
