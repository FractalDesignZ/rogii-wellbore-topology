#!/usr/bin/env python3
"""
GPU neural residual MEASURE — Blackwell CUDA path under Gate 3.

  CPU  : assemble features / priors / GroupKFold
  GPU  : Temporal CNN residual ΔTVT (AMP)
  NPU  : optional ONNX export for later OpenVINO (not product accept yet)

Law: residue never forced — Gate3 vs pure prior floors. Hunt target 5 ft.
RESTRICT: no Kaggle submit — measure OOF/seals only.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "gpu_neural"))
    ap.add_argument("--max-wells", type=int, default=48)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--export-onnx", action="store_true")
    a = ap.parse_args()

    from rwt.neural_residual import train_neural_residual
    from rwt.train import _assemble_dataset

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    print(
        json.dumps(
            {
                "phase": "MEASURE",
                "path": "CUDA_neural_residual",
                "target_ft": 5.0,
                "max_wells": a.max_wells,
            }
        ),
        flush=True,
    )
    stations, gate2 = _assemble_dataset(
        Path(a.data), synthetic=False, max_wells=a.max_wells
    )
    print(
        f"assembled n={len(stations)} wells={stations['well_id'].nunique()} "
        f"t={time.perf_counter()-t0:.1f}s",
        flush=True,
    )
    rep = train_neural_residual(
        stations,
        n_splits=a.folds,
        epochs=a.epochs,
        batch_wells=8,
        max_len=2048,
        hidden=128,
    )
    d = rep.to_dict()
    d["gate2_n_passed"] = gate2.get("n_passed")
    d["assemble_s"] = round(time.perf_counter() - t0, 2)
    (out / "measure_gpu_vs_5ft.json").write_text(
        json.dumps(d, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: d[k] for k in (
        "device", "oof_rmse", "oof_rmse_selected", "use_residual",
        "beats_5ft", "n_wells", "n_stations", "seconds", "baseline_rmse"
    )}, indent=2), flush=True)

    if a.export_onnx:
        try:
            from rwt.neural_residual import export_onnx_for_npu

            meta = export_onnx_for_npu(
                stations, out / "delta_tvt_tcn.onnx", max_len=512, epochs=3
            )
            (out / "onnx_export.json").write_text(
                json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8"
            )
            print("onnx", meta.get("onnx"), "ok", meta.get("onnx_ok"), flush=True)
        except Exception as e:
            # NPU export is experimental — never fail the CUDA MEASURE certificate
            print(f"onnx export SKIP: {type(e).__name__}: {e}", flush=True)

    # rc=0 when measure completed under law (G3 may still reject residual)
    return 0 if d.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
