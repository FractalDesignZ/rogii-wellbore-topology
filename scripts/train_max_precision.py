#!/usr/bin/env python3
"""
Max-precision ROGII train driver — use every core, full float64 features,
report vs 5.0 ft target under Gate3 residual law.

Harness coupling (separate process / MCP):
  restrict → this MEASURE → audit OPEN|STOP on RMSE vs 5ft + Gate3
  residue never forced: pure prior wins if residual loses floors
RESTRICT: no Kaggle submit — measure OOF/seals only.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# saturate this kit
os.environ.setdefault("OMP_NUM_THREADS", "20")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "20")
os.environ.setdefault("MKL_NUM_THREADS", "20")
os.environ.setdefault("RWT_NUM_THREADS", "20")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "20")

ROOT = Path(__file__).resolve().parents[1]
# src for rwt; repo root so private_kernel is importable under RWT_KERNEL=private
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

# Prefer float64 for feature geom / priors (accuracy over RAM on 32GB)
np.set_printoptions(precision=8)

from rwt.train import _assemble_dataset, train_lgbm_residual  # noqa: E402


TARGET_FT = 5.0


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "max_precision"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-wells", type=int, default=0, help="0 = all")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--workers",
        type=int,
        default=0,
        help="assemble workers (0=use RWT_WORKERS env or auto)",
    )
    ap.add_argument(
        "--no-max-precision",
        action="store_true",
        help="use crown-default shallow LGBM instead of deep residual head",
    )
    ap.add_argument(
        "--prior-col",
        default=None,
        help="residual base prior column (default: resolve DEFAULT_PRIOR_COL / fallbacks)",
    )
    ap.add_argument(
        "--kernel",
        default=None,
        choices=["public", "private"],
        help="force RWT_KERNEL=public|private (default: leave env / public)",
    )
    a = ap.parse_args()
    if a.kernel:
        os.environ["RWT_KERNEL"] = a.kernel

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    from rwt.kernel_api import resolve_kernel  # after path + RWT_KERNEL

    _k = resolve_kernel()
    print(
        json.dumps(
            {
                "phase": "MEASURE",
                "law": "restrict→measure→audit→OPEN|STOP",
                "target_ft": TARGET_FT,
                "kernel_id": _k.kernel_id(),
                "kernel_private": _k.is_private(),
                "kernel_note": getattr(_k, "_load_note", None),
                "RWT_KERNEL": os.environ.get("RWT_KERNEL"),
                "threads": {
                    "OMP": os.environ.get("OMP_NUM_THREADS"),
                    "OPENBLAS": os.environ.get("OPENBLAS_NUM_THREADS"),
                },
                "data": a.data,
                "max_wells": a.max_wells or "ALL",
            }
        ),
        flush=True,
    )

    # Honor CLI workers, then dashboard / parallel env
    workers = a.workers if a.workers > 0 else None
    if workers is None and os.environ.get("RWT_WORKERS"):
        try:
            workers = max(1, int(os.environ["RWT_WORKERS"]))
        except ValueError:
            workers = None
    stations, gate2 = _assemble_dataset(
        Path(a.data),
        synthetic=False,
        max_wells=a.max_wells,
        workers=workers,
    )
    print(
        f"assembled stations={len(stations)} wells={stations['well_id'].nunique()} "
        f"gate2_passed={gate2.get('n_passed')} t={time.perf_counter()-t0:.1f}s",
        flush=True,
    )
    # Guard proprietary columns on disk: residual train uses full in-memory frame.
    from rwt.train import write_stations_csv

    strip_meta = write_stations_csv(stations, out, write_private_full=True)
    stripped = list(strip_meta.get("stripped_columns") or [])

    # LightGBM: deeper + more leaves for max accuracy on this CPU
    result = train_lgbm_residual(
        stations,
        n_splits=a.folds,
        seed=a.seed,
        prior_col=a.prior_col,
        gate2_meta=gate2,
        max_precision=not a.no_max_precision,
    )
    rep = result["report"]
    selected = float(rep.get("oof_rmse_selected") or rep.get("oof_rmse") or 1e9)
    residual_oof = float(rep.get("oof_rmse") or 1e9)
    floors = rep.get("baseline_rmse") or {}
    # Prefer Gate3 best floor when present (comparable OOF row set)
    g3 = None
    for g in rep.get("gates") or []:
        if g.get("name") == "RESIDUAL_ACCEPTANCE":
            g3 = g
            break
    if g3 and g3.get("oof_rmse_best_floor") is not None:
        best_floor = float(g3["oof_rmse_best_floor"])
    elif floors:
        best_floor = float(min(floors.values()))
    else:
        best_floor = None
    use_res = bool(rep.get("use_residual"))

    beats_5 = selected < TARGET_FT
    cert = {
        "artifact": "RWT_MAX_PRECISION_MEASURE",
        "target_ft": TARGET_FT,
        "oof_rmse_selected": selected,
        "oof_rmse_residual": residual_oof,
        "best_floor_rmse": best_floor,
        "best_floor_name": (g3 or {}).get("best_floor_name"),
        "use_residual": use_res,
        "max_precision": not a.no_max_precision,
        "restrict": "no_kaggle_submit",
        "beats_5ft": beats_5,
        "delta_vs_5ft": selected - TARGET_FT,
        "n_stations": int(rep.get("n_stations") or 0),
        "n_wells": int(rep.get("n_wells") or 0),
        "kernel_id": _k.kernel_id(),
        "kernel_private": bool(_k.is_private()),
        "kernel_note": getattr(_k, "_load_note", None),
        "RWT_KERNEL": os.environ.get("RWT_KERNEL"),
        "report_kernel_id": rep.get("kernel_id"),
        "report_kernel_private": rep.get("kernel_private"),
        "stations_csv_stripped_columns": stripped,
        "gate2": gate2,
        "gates": rep.get("gates"),
        "baseline_rmse": floors,
        "fold_rmse": rep.get("fold_rmse"),
        "seconds": round(time.perf_counter() - t0, 2),
        "hardware_intent": {
            "CPU": "geometry+priors+LGBM+GroupKFold (this process)",
            "GPU": "optional neural residual / LMS scout — not force-required for trees",
            "NPU": "optional Job2 measure fabric — not TVT score authority",
            "law": "precision = correct prior floors + residual only if Gate3",
        },
        "verdict": "BEATS_5FT" if beats_5 else "NEED_IMPROVE",
        "production_open": "RESTRICT: no Kaggle submit — GO_MEASURE only (OOF/Gate3/seals)",
        "full_report": rep,
    }
    (out / "measure_vs_5ft.json").write_text(
        json.dumps(cert, indent=2, default=str) + "\n", encoding="utf-8"
    )
    (out / "cv_report.json").write_text(
        json.dumps(rep, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: cert[k] for k in (
        "verdict", "oof_rmse_selected", "best_floor_rmse", "use_residual",
        "beats_5ft", "delta_vs_5ft", "n_wells", "n_stations", "seconds"
    )}, indent=2), flush=True)
    return 0 if beats_5 else 2


if __name__ == "__main__":
    raise SystemExit(main())
