#!/usr/bin/env python3
"""
Private kernel pilot MEASURE — beam + lattice prior RMSE vs public floors (local only).

  set RWT_KERNEL=private
  python scripts/measure_private_beam_pilot.py --max-wells 40

Never ships private_kernel. Gate3 still owns residual accept on public path;
this script reports beam/lattice prior OOF-style holdout by well for gap diagnosis.
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
        from private_kernel.beam_match import (
            _calibrate_typewell_gr,
            _prep_typewell,
            beam_blend_last,
            beam_prior_for_well,
        )
        from private_kernel.lattice_match import (
            conf_gate_beam_lattice,
            fuse_beam_lattice,
            gr_emission_gate,
            lattice_blend_last,
            lattice_prior_for_well,
        )
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
    n_wells_scanned = 0  # raw train wells seen (pre comparable filter)
    # Sealed v6 conf_gate knobs (explicit; denser-129 / beam-led-v8 REJECT on 40w)
    conf_gate_params = {
        "margin": 0.08,
        "agree_ft": 2.0,
        "soft_temp": 0.05,
        "lat_floor": 0.18,
        "pure_lat_boost": 0.0,
    }
    lattice_n_nodes = 97  # denser 129 REJECT: pure lat 10.801 vs v6 10.686
    min_comparable_stations = 20
    for wid, df in iter_well_frames(root, "train"):
        n_wells_scanned += 1
        sp = resolve_middle_out(df, well_id=wid)
        if sp.n_predict < 10:
            continue
        tw_path = typewell_path_for_well(root, wid, "train")
        tw = load_typewell(tw_path) if tw_path else None
        # v6 seal: heel GR affine + adaptive sigma; v2 beam knobs; n_nodes=97
        beam = beam_prior_for_well(df, tw, sp, beam_width=48, heel_affine=True)
        lat = lattice_prior_for_well(
            df, tw, sp, n_nodes=lattice_n_nodes, heel_affine=True
        )
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
        lat_blend = lattice_blend_last(
            lat["prior_lattice_tvt"], last, lat["prior_lattice_conf"]
        )
        fused, fused_conf = fuse_beam_lattice(
            beam["prior_beam_tvt"],
            beam["prior_beam_conf"],
            lat["prior_lattice_tvt"],
            lat["prior_lattice_conf"],
            last,
        )
        # Label-free conf gate (sealed v6 soft params; pure_lattice off)
        gated, gated_conf = conf_gate_beam_lattice(
            blend,
            beam["prior_beam_conf"],
            lat_blend,
            lat["prior_lattice_conf"],
            pure_lattice=None,
            **conf_gate_params,
        )
        # Cycle-8 lever: GR emission gate (label-free stalk pick by typewell GR)
        emit = gated.copy()
        emit_conf = gated_conf.copy()
        if tw is not None and "GR" in df.columns:
            tw_tvt, tw_gr, _ = _prep_typewell(tw)
            known = np.asarray(sp.known_idx, dtype=int)
            tw_gr_cal, gr_sig = _calibrate_typewell_gr(
                tw_tvt, tw_gr, df, known, base_gr_sigma=10.0
            )
            emit, emit_conf = gr_emission_gate(
                blend,
                lat_blend,
                df["GR"].to_numpy(float),
                tw_tvt,
                tw_gr_cal,
                conf_a=beam["prior_beam_conf"],
                conf_b=lat["prior_lattice_conf"],
                gr_sigma=gr_sig,
                last_known=last,
            )
        # One comparable cohort: finite truth + all stalk preds + last-known.
        # Include raw lattice prior so rmse_lattice shares the same stations as blend/gate.
        m = pred[
            np.isfinite(y[pred])
            & np.isfinite(blend[pred])
            & np.isfinite(last[pred])
            & np.isfinite(beam["prior_beam_tvt"][pred])
            & np.isfinite(lat["prior_lattice_tvt"][pred])
            & np.isfinite(lat_blend[pred])
            & np.isfinite(fused[pred])
            & np.isfinite(gated[pred])
            & np.isfinite(emit[pred])
        ]
        if len(m) < min_comparable_stations:
            continue
        r_beam = rmse(y[m], beam["prior_beam_tvt"][m])
        r_blend = rmse(y[m], blend[m])
        r_lat = rmse(y[m], lat["prior_lattice_tvt"][m])
        r_lat_blend = rmse(y[m], lat_blend[m])
        r_fuse = rmse(y[m], fused[m])
        r_gated = rmse(y[m], gated[m])
        r_emit = rmse(y[m], emit[m])
        r_last = rmse(y[m], last[m])
        conf_m = beam["prior_beam_conf"][m]
        lat_conf_m = lat["prior_lattice_conf"][m]
        # oracle well pick is diagnostic only (label leakage) — not a deploy prior
        candidates = {
            "beam_blend": r_blend,
            "lattice_blend": r_lat_blend,
            "fuse_blend": r_fuse,
            "conf_gate": r_gated,
            "emit_gate": r_emit,
        }
        best_name = min(candidates, key=candidates.get)
        best_r = candidates[best_name]
        rows.append(
            {
                "well_id": wid,
                "n_pred": len(m),
                "rmse_beam": r_beam,
                "rmse_beam_blend": r_blend,
                "rmse_lattice": r_lat,
                "rmse_lattice_blend": r_lat_blend,
                "rmse_fuse_blend": r_fuse,
                "rmse_conf_gate": r_gated,
                "rmse_emit_gate": r_emit,
                "rmse_last_known": r_last,
                "rmse_oracle_stalk": best_r,
                "oracle_stalk": best_name,
                "mean_conf": float(np.nanmean(conf_m)),
                "mean_lat_conf": float(np.nanmean(lat_conf_m)),
                "mean_fuse_conf": float(np.nanmean(fused_conf[m])),
                "mean_gate_conf": float(np.nanmean(gated_conf[m])),
                "frac_conf_ge_0_5": float(np.mean(conf_m >= 0.5)),
                "beats_last_known": bool(r_blend < r_last),
                "beats_last_known_lat": bool(r_lat_blend < r_last),
                "beats_last_known_fuse": bool(r_fuse < r_last),
                "beats_last_known_gate": bool(r_gated < r_last),
                "beats_last_known_emit": bool(r_emit < r_last),
                "beats_conf_gate_emit": bool(r_emit < r_gated),
            }
        )
        if a.max_wells and len(rows) >= a.max_wells:
            break
        if len(rows) % 10 == 0:
            print(f"scored {len(rows)} wells…", flush=True)

    R = pd.DataFrame(rows)
    # Rows already require finite last-known; keep a named cohort for clarity / future filters.
    need = [
        "rmse_last_known",
        "rmse_beam_blend",
        "rmse_beam",
        "rmse_lattice_blend",
        "rmse_fuse_blend",
        "rmse_conf_gate",
        "rmse_emit_gate",
    ]
    C = R.dropna(subset=need) if len(R) else R

    def _mean(col: str) -> float | None:
        return float(C[col].mean()) if len(C) and col in C else None

    # Deploy prior for this pilot: conf_gate (label-free). Beam remains seal baseline.
    # emit_gate is trial-only until it beats conf_gate on the comparable cohort.
    deploy_col = "rmse_conf_gate"
    summary = {
        "ok": len(C) > 0,
        "kernel_id": k.kernel_id(),
        "n_wells": len(C),
        "n_wells_eligible": len(R),  # post finite-mask + min-stations (not raw scan)
        "n_wells_scanned": int(n_wells_scanned),
        "min_comparable_stations": min_comparable_stations,
        "comparable_cohort": (
            "finite y + beam/lattice/fuse/gate/emit + last on predict mask; "
            f"min_comparable_stations={min_comparable_stations}"
        ),
        "conf_gate_params": conf_gate_params,
        "lattice_n_nodes": lattice_n_nodes,
        "mean_rmse_beam": _mean("rmse_beam"),
        "mean_rmse_beam_blend": _mean("rmse_beam_blend"),
        "mean_rmse_lattice": _mean("rmse_lattice"),
        "mean_rmse_lattice_blend": _mean("rmse_lattice_blend"),
        "mean_rmse_fuse_blend": _mean("rmse_fuse_blend"),
        "mean_rmse_conf_gate": _mean("rmse_conf_gate"),
        "mean_rmse_emit_gate": _mean("rmse_emit_gate"),
        "mean_rmse_oracle_stalk": _mean("rmse_oracle_stalk"),
        "mean_rmse_last_known": _mean("rmse_last_known"),
        "median_rmse_beam_blend": float(C["rmse_beam_blend"].median()) if len(C) else None,
        "median_rmse_lattice_blend": float(C["rmse_lattice_blend"].median()) if len(C) else None,
        "median_rmse_conf_gate": float(C["rmse_conf_gate"].median()) if len(C) else None,
        "median_rmse_emit_gate": float(C["rmse_emit_gate"].median()) if len(C) else None,
        "beats_last_known_mean": (
            float(C["rmse_beam_blend"].mean()) < float(C["rmse_last_known"].mean())
            if len(C)
            else False
        ),
        "beats_last_known_lat_mean": (
            float(C["rmse_lattice_blend"].mean()) < float(C["rmse_last_known"].mean())
            if len(C)
            else False
        ),
        "beats_last_known_fuse_mean": (
            float(C["rmse_fuse_blend"].mean()) < float(C["rmse_last_known"].mean())
            if len(C)
            else False
        ),
        "beats_last_known_gate_mean": (
            float(C[deploy_col].mean()) < float(C["rmse_last_known"].mean()) if len(C) else False
        ),
        "beats_beam_blend_gate_mean": (
            float(C[deploy_col].mean()) < float(C["rmse_beam_blend"].mean()) if len(C) else False
        ),
        "frac_wells_beat_last_known": (
            float(C["beats_last_known"].mean()) if len(C) and "beats_last_known" in C else None
        ),
        "frac_wells_beat_last_known_lat": (
            float(C["beats_last_known_lat"].mean())
            if len(C) and "beats_last_known_lat" in C
            else None
        ),
        "frac_wells_beat_last_known_gate": (
            float(C["beats_last_known_gate"].mean())
            if len(C) and "beats_last_known_gate" in C
            else None
        ),
        "mean_conf": float(C["mean_conf"].mean()) if len(C) and "mean_conf" in C else None,
        "mean_lat_conf": (
            float(C["mean_lat_conf"].mean()) if len(C) and "mean_lat_conf" in C else None
        ),
        "delta_blend_vs_last_known": (
            float(C["rmse_beam_blend"].mean() - C["rmse_last_known"].mean()) if len(C) else None
        ),
        "delta_lattice_blend_vs_last_known": (
            float(C["rmse_lattice_blend"].mean() - C["rmse_last_known"].mean()) if len(C) else None
        ),
        "delta_fuse_vs_last_known": (
            float(C["rmse_fuse_blend"].mean() - C["rmse_last_known"].mean()) if len(C) else None
        ),
        "delta_conf_gate_vs_last_known": (
            float(C[deploy_col].mean() - C["rmse_last_known"].mean()) if len(C) else None
        ),
        "delta_conf_gate_vs_beam_blend": (
            float(C[deploy_col].mean() - C["rmse_beam_blend"].mean()) if len(C) else None
        ),
        "delta_emit_vs_last_known": (
            float(C["rmse_emit_gate"].mean() - C["rmse_last_known"].mean()) if len(C) else None
        ),
        "delta_emit_vs_conf_gate": (
            float(C["rmse_emit_gate"].mean() - C[deploy_col].mean()) if len(C) else None
        ),
        "beats_conf_gate_emit_mean": (
            float(C["rmse_emit_gate"].mean()) < float(C[deploy_col].mean()) if len(C) else False
        ),
        "delta_lattice_vs_beam_blend": (
            float(C["rmse_lattice_blend"].mean() - C["rmse_beam_blend"].mean()) if len(C) else None
        ),
        "delta_oracle_vs_beam_blend": (
            float(C["rmse_oracle_stalk"].mean() - C["rmse_beam_blend"].mean()) if len(C) else None
        ),
        "oracle_note": "rmse_oracle_stalk is diagnostic only (per-well min over stalks using labels)",
        "prior_pilot_mean_blend": 11.923837695178506,
        "prior_pilot_mean_last_known": 10.788491868712727,
        "beam_v2_mean_blend_seal": 10.75625511621551,
        "public_full_cohort_selected": 14.67,
        "public_all_stack_selected_reject": 15.199496423546888,
        "crown_band": [6.0, 7.0],
        "hunt_ft": 5.0,
        "stalk": "v9: GR emission gate trial + sealed v6 conf_gate deploy; heel affine; n_nodes=97",
        "deploy_prior": "conf_gate",
        "prior_conf_gate_seal_v3": 10.748919252940581,
        "prior_conf_gate_seal_v6": 10.705360833184525,
        "trial_v8_conf_gate_40w": 10.709574438415903,
        "trial_v8_pure_lattice_40w": 10.800656754206665,
        "note": "Per-well predict-zone RMSE of private priors (not full GroupKFold residual stack)",
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
