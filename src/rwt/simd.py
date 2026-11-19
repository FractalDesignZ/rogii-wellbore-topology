"""
CPU instruction-set awareness + AVX-friendly array contracts.

Python cannot emit raw AVX opcodes in pure Python, but our hot paths go through
NumPy and its configured BLAS backend, which select supported ISA kernels.
This module:

  1. Reports available ISA flags (best-effort on Windows)
  2. Forces C-contiguous float64 views (SIMD-friendly strides)
  3. Centralizes vectorized kernels used by GR match / PF / geometry

Optional: install ``numba`` for JIT SIMD loops (``pip install numba``).
"""
from __future__ import annotations

import os
import platform
import struct
from collections.abc import Mapping
from functools import lru_cache
from importlib import import_module
from typing import Any

import numpy as np

# Prefer multi-thread OpenBLAS when available (env can override)
if "OMP_NUM_THREADS" not in os.environ:
    # leave unset so OpenBLAS uses its own default; document only
    pass


def _numpy_cpu_info() -> Any:
    """Locate runtime dispatch metadata in NumPy 2.x or 1.x, respectively."""
    for name in ("numpy._core._multiarray_umath", "numpy.core._multiarray_umath"):
        try:
            core = import_module(name)
            if isinstance(getattr(core, "__cpu_features__", None), Mapping):
                return core
        except Exception:
            continue
    # Some alternate builds expose this metadata at the top level.
    return np


def _numpy_blas_backend() -> str:
    """Read selected build dependencies, never infer a backend from importability."""
    try:
        cfg = import_module("numpy.__config__")
        config = getattr(cfg, "CONFIG", {})
        if isinstance(config, Mapping):
            dependencies = config.get("Build Dependencies", {})
            blas = dependencies.get("blas", {}) if isinstance(dependencies, Mapping) else {}
            if isinstance(blas, Mapping) and blas.get("found", True):
                name = blas.get("name")
                if isinstance(name, str) and name.strip():
                    return name.strip()
        # NumPy 1.24 distutils builds expose selected BLAS libraries via get_info.
        get_info = getattr(cfg, "get_info", None)
        if callable(get_info):
            for key in ("blas_ilp64_opt_info", "blas_opt_info"):
                info = get_info(key)
                if not isinstance(info, Mapping) or not info:
                    continue
                libraries = [str(value) for value in info.get("libraries", [])]
                if any("openblas" in value.lower() for value in libraries):
                    return "openblas"
                if any("mkl" in value.lower() for value in libraries):
                    return "mkl"
                return ", ".join(libraries) or "unknown"
    except Exception:
        pass
    return "unknown"


@lru_cache(maxsize=1)
def cpu_feature_report() -> dict[str, Any]:
    """Best-effort NumPy runtime CPU flags and configured BLAS (Windows-safe)."""
    machine = platform.machine().upper()
    report: dict[str, Any] = {
        "arch": platform.machine(),
        "processor": platform.processor(),
        "python_bits": 8 * struct.calcsize("P"),
        "numpy_version": np.__version__,
        "simd": {},
        "blas_hint": (
            "NumPy ufuncs and the configured BLAS backend select supported ISA kernels; "
            "build dispatch targets do not imply runtime CPU support"
        ),
    }
    if machine in {"ARM", "ARM64", "AARCH64"} or machine.startswith("ARMV"):
        report["simd_family"] = "ARM"
    elif machine in {"X86", "X86_64", "AMD64", "I386", "I486", "I586", "I686"}:
        report["simd_family"] = "x86"
    else:
        report["simd_family"] = "unknown"
    try:
        core = _numpy_cpu_info()
        feats = getattr(core, "__cpu_features__", None) or {}
        if isinstance(feats, Mapping) and feats:
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
            if report["simd_family"] == "ARM" and (feats.get("NEON") or feats.get("ASIMD")):
                report["simd"]["NEON_ASIMD"] = True
        baseline = getattr(core, "__cpu_baseline__", None)
        dispatch = getattr(core, "__cpu_dispatch__", None)
        if baseline is not None:
            report["numpy_cpu_baseline"] = list(baseline) if baseline else []
        if dispatch is not None:
            report["numpy_cpu_dispatch"] = list(dispatch) if dispatch else []
    except Exception as e:
        report["simd_error"] = str(e)[:200]

    report["blas_backend"] = _numpy_blas_backend()
    report["openblas"] = "openblas" in report["blas_backend"].lower()

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
    """SIMD-friendly rolling mean via cumulative sum (O(n))."""
    x = as_f64_contig(x)
    n = len(x)
    if n == 0:
        return x.copy()
    w = max(1, min(int(window), n))
    if w == 1:
        return x.copy()
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = (c[w:] - c[:-w]) / w
    if len(out) == 0:
        return np.full(n, float(np.mean(x)))
    # symmetric-ish pad to length n
    pad_l = w // 2
    pad_r = n - len(out) - pad_l
    if pad_r < 0:
        return out[:n]
    left = np.full(max(pad_l, 0), out[0])
    right = np.full(max(pad_r, 0), out[-1])
    return np.concatenate([left, out, right])[:n]


def vectorized_ncc_template(
    query: np.ndarray,
    template: np.ndarray,
) -> tuple[float, int]:
    """
    Best sliding NCC of query along template with per-window template z-score.

    Uses rolling sums (O(n)) so each lag window is normalized independently.
    """
    q = as_f64_contig(query)
    t = as_f64_contig(template)
    nq, nt = len(q), len(t)
    if nq < 3 or nt < nq:
        return 0.0, 0
    q = q - np.mean(q)
    qs = float(np.std(q))
    if qs < 1e-12:
        return 0.0, 0
    q = q / qs
    # rolling mean/var of template windows of length nq
    ones = np.ones(nq, dtype=float)
    sum_t = np.convolve(t, ones, mode="valid")
    sum_t2 = np.convolve(t * t, ones, mode="valid")
    mean_t = sum_t / nq
    var_t = np.maximum(sum_t2 / nq - mean_t * mean_t, 0.0)
    std_t = np.sqrt(var_t)
    # cross term: correlate(t, q) = sum t[i:i+nq] * q
    cross = np.correlate(t, q, mode="valid")
    # NCC = sum (t - mean_t)(q) / (nq * std_t * 1) ; q already unit std, mean 0
    # sum (t-mean)*q = cross - mean_t * sum(q) = cross (sum q ~ 0)
    denom = np.maximum(std_t * nq, 1e-12)
    ncc = cross / denom
    lag = int(np.argmax(ncc))
    return float(ncc[lag]), lag
