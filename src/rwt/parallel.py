"""Multi-core assemble + thread knobs. GPU/NPU not used by public rwt (NumPy/LGBM CPU)."""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import pandas as pd

from rwt.features import build_station_frame
from rwt.gates import gate_middle_out_mask
from rwt.middle_out import MiddleOutSplit
from rwt.spatial import SpatialIndex


def configure_compute_env(*, workers: int | None = None) -> dict[str, Any]:
    """
    Set BLAS/OpenMP/LightGBM-friendly env before heavy work.

    With multi-process well assemble, pin BLAS to 1–2 threads per worker
    to avoid oversubscription. LightGBM uses its own num_threads.
    """
    n = workers or int(os.environ.get("RWT_WORKERS") or 0) or max(1, (os.cpu_count() or 4) - 1)
    n = max(1, min(n, os.cpu_count() or n))
    # per-worker BLAS: keep small when multiprocessing
    blas = int(os.environ.get("RWT_BLAS_THREADS") or max(1, (os.cpu_count() or 4) // n))
    os.environ.setdefault("OMP_NUM_THREADS", str(blas))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(blas))
    os.environ.setdefault("MKL_NUM_THREADS", str(blas))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(blas))
    os.environ["RWT_WORKERS"] = str(n)
    return {
        "workers": n,
        "blas_threads": blas,
        "cpu_count": os.cpu_count(),
        "note": "Public rwt is CPU (NumPy/OpenBLAS + LightGBM). No NPU/GPU kernels in open-core.",
    }


def _assemble_one(
    payload: tuple[
        str,
        pd.DataFrame,
        MiddleOutSplit,
        pd.DataFrame | None,
        SpatialIndex,
    ],
) -> dict[str, Any]:
    """Worker: Gate2 + station frame for one well (process-safe)."""
    wid, df, sp, tw, spatial = payload
    g2 = gate_middle_out_mask(df, wid)
    if not g2.passed:
        return {
            "well_id": wid,
            "passed_gate2": False,
            "gate2": g2.to_dict(),
            "frame": None,
            "unlabeled": False,
        }
    tvt_col = "TVT" if "TVT" in df.columns else "TVT_true"
    tvt_arr = df[tvt_col].to_numpy()
    pred_idx = __import__("numpy").asarray(sp.predict_idx, dtype=int)
    labeled_pred = pred_idx[pd.notna(tvt_arr[pred_idx])]
    if len(labeled_pred) == 0:
        return {
            "well_id": wid,
            "passed_gate2": True,
            "gate2": g2.to_dict(),
            "frame": None,
            "unlabeled": True,
        }
    frame = build_station_frame(
        df,
        tw,
        sp,
        station_idx=labeled_pred,
        spatial_index=spatial,
    )
    return {
        "well_id": wid,
        "passed_gate2": True,
        "gate2": g2.to_dict(),
        "frame": frame,
        "unlabeled": False,
    }


def assemble_wells_parallel(
    loaded: list[tuple[str, pd.DataFrame, MiddleOutSplit, pd.DataFrame | None]],
    spatial: SpatialIndex,
    *,
    workers: int | None = None,
    progress: Any | None = None,
) -> tuple[list[pd.DataFrame], dict[str, Any]]:
    """
    Multi-process well assemble. PF bag inside each well is the hot path —
    this is why sequential train left cores idle.
    """
    cfg = configure_compute_env(workers=workers)
    n_workers = int(cfg["workers"])
    frames: list[pd.DataFrame] = []
    gate2_results: list[dict[str, Any]] = []
    unlabeled_skips: list[str] = []
    passed_wells: list[str] = []

    if progress is not None:
        progress.state["workers"] = n_workers
        progress.set_phase("assemble", total=len(loaded), message=f"workers={n_workers}")

    payloads = [(wid, df, sp, tw, spatial) for wid, df, sp, tw in loaded]

    if n_workers <= 1 or len(payloads) <= 1:
        results = [_assemble_one(p) for p in payloads]
        # sequential still reports progress
        ordered = results
        if progress is not None:
            for r in ordered:
                if r["frame"] is not None:
                    frames.append(r["frame"])
                    passed_wells.append(r["well_id"])
                elif not r["passed_gate2"]:
                    pass
                elif r.get("unlabeled"):
                    unlabeled_skips.append(r["well_id"])
                gate2_results.append(
                    {
                        "well_id": r["well_id"],
                        "passed": bool(r["passed_gate2"]),
                        "detail": r["gate2"].get("detail", r["gate2"]),
                    }
                )
                progress.tick(item=r["well_id"], ok=r["frame"] is not None)
    else:
        # preserve completion-order progress; collect then sort by original order
        by_id: dict[str, dict[str, Any]] = {}
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(_assemble_one, p): p[0] for p in payloads}
            for fut in as_completed(futs):
                wid = futs[fut]
                try:
                    r = fut.result()
                except Exception as e:  # noqa: BLE001 — surface per-well failure
                    r = {
                        "well_id": wid,
                        "passed_gate2": False,
                        "gate2": {"detail": {"error": str(e)[:300]}},
                        "frame": None,
                        "unlabeled": False,
                    }
                by_id[wid] = r
                if progress is not None:
                    progress.tick(item=wid, ok=r.get("frame") is not None)
        for wid, _df, _sp, _tw in loaded:
            r = by_id[wid]
            gate2_results.append(
                {
                    "well_id": r["well_id"],
                    "passed": bool(r["passed_gate2"]),
                    "detail": r["gate2"].get("detail", r["gate2"])
                    if isinstance(r["gate2"], dict)
                    else r["gate2"],
                }
            )
            if r.get("frame") is not None:
                frames.append(r["frame"])
                passed_wells.append(r["well_id"])
            elif r.get("unlabeled"):
                unlabeled_skips.append(r["well_id"])

    rejected = [r for r in gate2_results if not r["passed"]]
    gate2_meta = {
        "name": "MIDDLE_OUT_MASK",
        "passed": len(rejected) == 0 and len(passed_wells) > 0,
        "n_passed": len(passed_wells),
        "n_rejected": len(rejected),
        "passed_wells": passed_wells,
        "rejected_wells": rejected,
        "unlabeled_skips": unlabeled_skips,
        "results": gate2_results,
        "workers": n_workers,
        "compute": cfg,
    }
    return frames, gate2_meta
