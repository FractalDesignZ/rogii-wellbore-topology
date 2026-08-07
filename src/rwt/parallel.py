"""Multi-core assemble. Avoids pickling SpatialIndex + giant frames across pipes."""
from __future__ import annotations

import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from rwt.features import build_station_frame, seal_station_frame
from rwt.gates import gate_middle_out_mask
from rwt.middle_out import MiddleOutSplit
from rwt.spatial import SpatialIndex, attach_spatial_features


def configure_compute_env(*, workers: int | None = None) -> dict[str, Any]:
    """
    Set BLAS/OpenMP/LightGBM-friendly env before heavy work.

    With multi-process well assemble, pin BLAS to 1 thread per worker
    to avoid oversubscription. LightGBM uses its own num_threads.
    """
    n = workers or int(os.environ.get("RWT_WORKERS") or 0) or max(1, (os.cpu_count() or 4) - 1)
    n = max(1, min(n, os.cpu_count() or n))
    blas = int(os.environ.get("RWT_BLAS_THREADS") or 1)
    # force pin — setdefault would leave inherited OPENBLAS_NUM_THREADS=32 etc.
    os.environ["OMP_NUM_THREADS"] = str(blas)
    os.environ["OPENBLAS_NUM_THREADS"] = str(blas)
    os.environ["MKL_NUM_THREADS"] = str(blas)
    os.environ["NUMEXPR_NUM_THREADS"] = str(blas)
    os.environ["RWT_WORKERS"] = str(n)
    return {
        "workers": n,
        "blas_threads": blas,
        "cpu_count": os.cpu_count(),
        "note": "Public rwt is CPU (NumPy/OpenBLAS + LightGBM). No NPU/GPU kernels in open-core.",
    }


def _assemble_one_to_disk(
    payload: tuple[str, pd.DataFrame, MiddleOutSplit, pd.DataFrame | None, str],
) -> dict[str, Any]:
    """
    Worker: Gate2 + station frame **without** spatial index (avoids huge pickles).

    Writes frame to ``tmp_dir/{wid}.pkl`` and returns only a small metadata dict.
    """
    wid, df, sp, tw, tmp_dir = payload
    g2 = gate_middle_out_mask(df, wid)
    if not g2.passed:
        return {
            "well_id": wid,
            "passed_gate2": False,
            "gate2": g2.to_dict(),
            "path": None,
            "unlabeled": False,
            "error": None,
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
            "path": None,
            "unlabeled": True,
            "error": None,
        }
    # no spatial / seal here — parent attaches + seals once (CR P1.1: single kernel pass)
    frame = build_station_frame(
        df,
        tw,
        sp,
        station_idx=labeled_pred,
        spatial_index=None,
        seal=False,
    )
    # secure temp path inside tmp_dir (never derive filename from well_id)
    fd, path = tempfile.mkstemp(dir=tmp_dir, suffix=".pkl", prefix="well_")
    os.close(fd)
    try:
        frame.to_pickle(path, protocol=4)
    except Exception:
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return {
        "well_id": wid,
        "passed_gate2": True,
        "gate2": g2.to_dict(),
        "path": path,
        "unlabeled": False,
        "error": None,
        "n_rows": int(len(frame)),
    }


def assemble_wells_parallel(
    loaded: list[tuple[str, pd.DataFrame, MiddleOutSplit, pd.DataFrame | None]],
    spatial: SpatialIndex,
    *,
    workers: int | None = None,
    progress: Any | None = None,
) -> tuple[list[pd.DataFrame], dict[str, Any]]:
    """
    Multi-process well assemble with disk-backed frames.

    Previous hang: pickling SpatialIndex (all wells) + large frames through
    ProcessPool pipes froze ~100+ wells in. Workers write pickles; parent loads
    and attaches spatial neighbor features.
    """
    cfg = configure_compute_env(workers=workers)
    n_workers = int(cfg["workers"])
    frames: list[pd.DataFrame] = []
    gate2_results: list[dict[str, Any]] = []
    unlabeled_skips: list[str] = []
    passed_wells: list[str] = []
    seal_phase_s = 0.0
    seal_per_well: list[dict[str, Any]] = []

    if progress is not None:
        progress.state["workers"] = n_workers
        progress.set_phase("assemble", total=len(loaded), message=f"workers={n_workers} disk-backed")

    tmp_root = Path(tempfile.mkdtemp(prefix="rwt_assemble_"))
    by_id: dict[str, dict[str, Any]] = {}

    try:
        payloads = [
            (wid, df, sp, tw, str(tmp_root)) for wid, df, sp, tw in loaded
        ]
        if n_workers <= 1 or len(payloads) <= 1:
            for p in payloads:
                r = _assemble_one_to_disk(p)
                by_id[r["well_id"]] = r
                if progress is not None:
                    progress.tick(
                        item=r["well_id"],
                        ok=r.get("path") is not None,
                        message=f"rows={r.get('n_rows')}",
                    )
        else:
            # chunk submissions so a bad pickle can't block forever without ticks
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                futs = {ex.submit(_assemble_one_to_disk, p): p[0] for p in payloads}
                for fut in as_completed(futs):
                    wid = futs[fut]
                    try:
                        r = fut.result()
                    except Exception as e:  # noqa: BLE001
                        r = {
                            "well_id": wid,
                            "passed_gate2": False,
                            "gate2": {"detail": {"error": str(e)[:400]}},
                            "path": None,
                            "unlabeled": False,
                            "error": str(e)[:400],
                        }
                    by_id[wid] = r
                    if progress is not None:
                        progress.tick(
                            item=wid,
                            ok=r.get("path") is not None,
                            message=f"rows={r.get('n_rows')} err={r.get('error')}",
                        )

        # load frames + attach spatial in parent; seal once per well (single-seal).
        # Thread pool: SpatialIndex is read-only after build; NumPy-heavy seal parallelizes.
        if progress is not None:
            progress.set_phase(
                "spatial_attach",
                total=len(loaded),
                message="neighbor features + seal on assembled frames",
            )

        def _seal_one(
            item: tuple[str, pd.DataFrame, MiddleOutSplit, pd.DataFrame | None, str],
        ) -> dict[str, Any]:
            wid, df, sp, tw, path = item
            t0 = time.perf_counter()
            frame = pd.read_pickle(path)
            frame = attach_spatial_features(frame, df, spatial, wid)
            frame = seal_station_frame(frame, df, sp, tw)
            dt = time.perf_counter() - t0
            return {
                "well_id": wid,
                "frame": frame,
                "seconds": round(dt, 4),
                "n": len(frame),
            }

        seal_jobs: list[
            tuple[str, pd.DataFrame, MiddleOutSplit, pd.DataFrame | None, str]
        ] = []
        for wid, df, sp, tw in loaded:
            r = by_id.get(wid) or {
                "well_id": wid,
                "passed_gate2": False,
                "gate2": {},
                "path": None,
            }
            gate2_results.append(
                {
                    "well_id": wid,
                    "passed": bool(r.get("passed_gate2")),
                    "detail": (r.get("gate2") or {}).get("detail", r.get("gate2")),
                }
            )
            path = r.get("path")
            if not path:
                if r.get("unlabeled"):
                    unlabeled_skips.append(wid)
                if progress is not None:
                    progress.tick(item=wid, ok=False, message="no_frame")
                continue
            seal_jobs.append((wid, df, sp, tw, path))

        n_seal_threads = max(1, min(int(cfg.get("workers") or 1), len(seal_jobs) or 1))
        sealed_by_id: dict[str, dict[str, Any]] = {}
        seal_errors: dict[str, str] = {}
        if n_seal_threads <= 1 or len(seal_jobs) <= 1:
            for job in seal_jobs:
                try:
                    sealed_by_id[job[0]] = _seal_one(job)
                except Exception as e:  # noqa: BLE001 — per-well continue like process pool
                    seal_errors[job[0]] = str(e)[:400]
        else:
            with ThreadPoolExecutor(max_workers=n_seal_threads) as tex:
                futs = {tex.submit(_seal_one, job): job[0] for job in seal_jobs}
                for fut in as_completed(futs):
                    wid = futs[fut]
                    try:
                        sealed_by_id[wid] = fut.result()
                    except Exception as e:  # noqa: BLE001 — per-well continue
                        seal_errors[wid] = str(e)[:400]

        # stable well order matching loaded
        for wid, _df, _sp, _tw in loaded:
            s = sealed_by_id.get(wid)
            if not s:
                if wid in seal_errors:
                    err = seal_errors[wid]
                    for gr in gate2_results:
                        if gr.get("well_id") == wid:
                            gr["passed"] = False
                            gr["detail"] = f"seal_failed: {err}"
                            break
                    if progress is not None:
                        progress.tick(
                            item=wid, ok=False, message=f"seal_failed {err}"
                        )
                continue
            seal_phase_s += float(s["seconds"])
            seal_per_well.append(
                {
                    "well_id": wid,
                    "seconds": s["seconds"],
                    "n": s["n"],
                }
            )
            frames.append(s["frame"])
            passed_wells.append(wid)
            if progress is not None:
                progress.tick(
                    item=wid,
                    ok=True,
                    message=f"spatial+seal n={s['n']} {s['seconds']:.3f}s",
                )
    finally:
        # best-effort cleanup of temp pickles
        try:
            for p in tmp_root.glob("*.pkl"):
                p.unlink(missing_ok=True)
            tmp_root.rmdir()
        except OSError:
            pass

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
        "ipc": "disk_pickle_no_spatial_in_worker",
        # parent-side attach_spatial + seal_station_frame cost (kernel visible)
        "seal_phase_seconds": round(seal_phase_s, 4),
        "seal_per_well": seal_per_well,
        "seal_threads": int(n_seal_threads),
        "seal_errors": dict(seal_errors),
    }
    return frames, gate2_meta
