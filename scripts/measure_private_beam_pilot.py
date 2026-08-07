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
    from rwt.priors import compute_local_dip_prior
    from private_kernel.beam_match import beam_blend_last, beam_prior_for_well
    from rwt.kernel_api import resolve_kernel

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--max-wells", type=int, default=40)
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "private_beam_pilot"))
    a = ap.parse_args()

    k = resolve_kernel()
    print(json.dumps({"kernel_id": k.kernel_id(), "private": k.is_private()}), flush=True)
    if not k.is_private():
        # Fail-closed: public CI / clones without private_kernel stay green-path honest
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
        m = pred[np.isfinite(y[pred]) & np.isfinite(blend[pred])]
        if len(m) < 20:
            continue
        r_beam = rmse(y[m], beam["prior_beam_tvt"][m])
        r_blend = rmse(y[m], blend[m])
        r_last = rmse(y[m], last[m]) if np.isfinite(last[m]).all() else float("nan")
        rows.append(
            {
                "well_id": wid,
                "n_pred": int(len(m)),
                "rmse_beam": r_beam,
                "rmse_beam_blend": r_blend,
                "rmse_last_known": r_last,
            }
        )
        if a.max_wells and len(rows) >= a.max_wells:
            break
        if len(rows) % 10 == 0:
            print(f"scored {len(rows)} wells…", flush=True)

    R = pd.DataFrame(rows)
    summary = {
        "ok": len(R) > 0,
        "kernel_id": k.kernel_id(),
        "n_wells": int(len(R)),
        "mean_rmse_beam": float(R["rmse_beam"].mean()) if len(R) else None,
        "mean_rmse_beam_blend": float(R["rmse_beam_blend"].mean()) if len(R) else None,
        "mean_rmse_last_known": float(R["rmse_last_known"].mean()) if len(R) else None,
        "median_rmse_beam_blend": float(R["rmse_beam_blend"].median()) if len(R) else None,
        "beats_last_known_mean": (
            float(R["rmse_beam_blend"].mean()) < float(R["rmse_last_known"].mean())
            if len(R)
            else False
        ),
        "public_full_cohort_selected": 14.67,
        "crown_band": [6.0, 7.0],
        "hunt_ft": 5.0,
        "note": "Per-well predict-zone RMSE of pure beam prior (not full GroupKFold residual stack)",
        "law": "private offline scaffold; export predictions only; Gate3 still owns residual accept",
    }
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    R.to_csv(out / "per_well_beam.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
