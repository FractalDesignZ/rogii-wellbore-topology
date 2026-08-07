"""
Feature / prediction watermarking for provenance.

Uses HMAC-SHA256 with RWT_WATERMARK_KEY (or public default salt).
Does not prove legal ownership alone — supports technical attribution.
"""
from __future__ import annotations

import hashlib
import hmac
import os
from typing import Any

import numpy as np
import pandas as pd

# Public default is weak on purpose if key unset — set RWT_WATERMARK_KEY privately.
_DEFAULT_SALT = b"rwt-public-watermark-v1-not-secret"


def _key() -> bytes:
    k = (os.environ.get("RWT_WATERMARK_KEY") or "").encode("utf-8")
    return k if k else _DEFAULT_SALT


def watermark_token(
    *parts: str,
    nbytes: int = 8,
) -> str:
    msg = "|".join(str(p) for p in parts).encode("utf-8")
    dig = hmac.new(_key(), msg, hashlib.sha256).hexdigest()
    return dig[: nbytes * 2]


def embed_column(
    df: pd.DataFrame,
    *,
    well_col: str = "well_id",
    md_col: str = "MD",
) -> pd.DataFrame:
    """Add deterministic watermark column (safe for public feature exports)."""
    out = df.copy()
    tokens = []
    for _, row in out.iterrows():
        tokens.append(
            watermark_token(
                str(row.get(well_col, "")),
                f"{float(row.get(md_col, 0.0)):.4f}",
                "rwt.v1",
            )
        )
    out["rwt_wm"] = tokens
    return out


def embed_prediction_array(
    y: np.ndarray,
    *,
    seed_parts: list[str],
    strength: float = 1e-9,
) -> np.ndarray:
    """
    Tiny deterministic dither on predictions (optional).

    strength keeps effect far below typical TVT RMSE noise floors when small.
    Disable for pure accuracy runs: RWT_WATERMARK_PREDICTIONS=0
    """
    if (os.environ.get("RWT_WATERMARK_PREDICTIONS") or "0").strip() in (
        "0",
        "false",
        "no",
    ):
        return np.asarray(y, dtype=float)
    y = np.asarray(y, dtype=float).copy()
    tok = watermark_token(*seed_parts)
    # map hex to [-1,1] stream
    for i in range(len(y)):
        h = hmac.new(_key(), f"{tok}:{i}".encode(), hashlib.sha256).digest()
        u = int.from_bytes(h[:4], "big") / 2**32
        y[i] = y[i] + strength * (2.0 * u - 1.0)
    return y


def verify_column(df: pd.DataFrame) -> dict[str, Any]:
    if "rwt_wm" not in df.columns:
        return {"ok": False, "reason": "missing rwt_wm"}
    ok = 0
    n = 0
    for _, row in df.iterrows():
        n += 1
        expect = watermark_token(
            str(row.get("well_id", "")),
            f"{float(row.get('MD', 0.0)):.4f}",
            "rwt.v1",
        )
        if row["rwt_wm"] == expect:
            ok += 1
    return {"ok": ok == n and n > 0, "n": n, "n_match": ok}
