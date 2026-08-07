"""
CPU instruction-set awareness + AVX-friendly array contracts.

Python cannot emit raw AVX opcodes in pure Python, but our hot paths go through
NumPy/OpenBLAS which already use DYNAMIC_ARCH kernels (this machine: OpenBLAS
Haswell → AVX2). This module:

  1. Reports available ISA flags (best-effort on Windows)
  2. Forces C-contiguous float64 views (SIMD-friendly strides)
  3. Centralizes vectorized kernels used by GR match / PF / geometry

Optional: install ``numba`` for JIT SIMD loops (``pip install numba``).
"""
from __future__ import annotations

import os
import platform
import struct
import sys
from functools import lru_cache
from typing import Any

import numpy as np

# Prefer multi-thread OpenBLAS when available (env can override)
if "OMP_NUM_THREADS" not in os.environ:
    # leave unset so OpenBLAS uses its own default; document only
    pass


@lru_cache(maxsize=1)
def cpu_feature_report() -> dict[str, Any]:
    """Best-effort CPU feature map (Windows-safe)."""
    machine = platform.machine().upper()
    report: dict[str, Any] = {
        "arch": platform.machine(),
        "processor": platform.processor(),
        "python_bits": 8 * struct.calcsize("P"),
        "numpy_version": np.__version__,
        "simd": {},
        "blas_hint": (
            "NumPy ufuncs + OpenBLAS matmul use ISA kernels automatically "
            "(x86: SSE/AVX/AVX2/AVX-512 via DYNAMIC_ARCH; ARM64: NEON/ASIMD)"
        ),
    }
    if "ARM" in machine or "AARCH" in machine:
        report["simd"]["NEON_ASIMD"] = True  # universal on aarch64 userland
        report["simd_family"] = "ARM"
    else:
        report["simd_family"] = "x86"
    # NumPy 2.x exposes CPU features via __cpu_features__ / __cpu_baseline__
    try:
        feats = getattr(np, "__cpu_features__", None) or {}
        if isinstance(feats, dict) and feats:
            keys = (
                "SSE",
                "SSE2",
                "SSE3",
                "SSSE3",
                "SSE41",
                "SSE42",
                "AVX",
                "AVX2",
                "FMA3",
                "AVX512F",
                "AVX512_SKX",
                "NEON",
                "ASIMD",
                "ASIMDHP",
            )
            for k in keys:
                if k in feats:
                    report["simd"][k] = bool(feats[k])
            # also dump any true flags
            for k, v in feats.items():
                if v and k not in report["simd"]:
                    report["simd"][k] = True
        baseline = getattr(np, "__cpu_baseline__", None)
        dispatch = getattr(np, "__cpu_dispatch__", None)
        if baseline is not None:
            report["numpy_cpu_baseline"] = list(baseline) if baseline else []
        if dispatch is not None:
            report["numpy_cpu_dispatch"] = list(dispatch) if dispatch else []
    except Exception as e:
        report["simd_error"] = str(e)[:200]

    try:
        import numpy.__config__ as cfg  # type: ignore

        # string dump is heavy; keep a short flag
        report["openblas"] = True
    except Exception:
        report["openblas"] = False

    report["numba"] = False
    try:
        import numba  # noqa: F401

        report["numba"] = True
        report["numba_version"] = numba.__version__  # type: ignore[name-defined]
    except Exception:
        pass

    # Environment knobs for thread-level parallel BLAS
    report["env"] = {
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "RWT_NUM_THREADS": os.environ.get("RWT_NUM_THREADS"),
    }
    return report


def as_f64_contig(a: np.ndarray) -> np.ndarray:
    """C-contiguous float64 view/copy — preferred layout for SIMD loads."""
    a = np.asarray(a, dtype=np.float64)
    if not a.flags["C_CONTIGUOUS"]:
        a = np.ascontiguousarray(a)
    return a


def set_thread_env(n: int | None = None) -> dict[str, str]:
    """
    Optionally pin OpenMP/OpenBLAS threads (call before heavy BLAS if desired).

    ``RWT_NUM_THREADS`` env overrides ``n``.
    """
    if n is None:
        raw = os.environ.get("RWT_NUM_THREADS")
        if raw and raw.isdigit():
            n = int(raw)
    applied = {}
    if n is not None and n > 0:
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[key] = str(n)
            applied[key] = str(n)
    return applied


def vectorized_rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """AVX-friendly rolling mean via cumulative sum (O(n))."""
    x = as_f64_contig(x)
    w = max(1, int(window))
    if w == 1 or len(x) == 0:
        return x.copy()
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = (c[w:] - c[:-w]) / w
    # pad edges
    pad_l = np.full(w // 2, out[0] if len(out) else 0.0)
    pad_r = np.full(len(x) - len(out) - len(pad_l), out[-1] if len(out) else 0.0)
    return np.concatenate([pad_l, out, pad_r])[: len(x)]


def vectorized_ncc_template(
    query: np.ndarray,
    template: np.ndarray,
) -> tuple[float, int]:
    """
    Best lag-0-aligned sliding NCC of query along template via correlate.

    Uses float64 contiguous arrays → OpenBLAS/SIMD GEMM-style paths under NumPy.
    """
    q = as_f64_contig(query)
    t = as_f64_contig(template)
    nq, nt = len(q), len(t)
    if nq < 3 or nt < nq:
        return 0.0, 0
    q = q - np.mean(q)
    qs = np.std(q)
    if qs < 1e-12:
        return 0.0, 0
    q = q / qs
    # normalize template locally per window is expensive; zscore whole template then correlate
    t0 = t - np.mean(t)
    ts = np.std(t0)
    if ts < 1e-12:
        return 0.0, 0
    t0 = t0 / ts
    # full correlation; valid lags 0..nt-nq
    corr = np.correlate(t0, q, mode="valid") / max(nq, 1)
    lag = int(np.argmax(corr))
    return float(corr[lag]), lag
