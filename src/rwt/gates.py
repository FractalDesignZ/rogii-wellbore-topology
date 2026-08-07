"""
Logic quality gates (M0-M6 operational law).

Gate 1 — IP fence: public vs private kernel (fail-closed).
Gate 2 — Middle-out mask: heel prior / toe predict.
Gate 3 — Residual acceptance: LGBM head only if it beats pure prior OOF.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from rwt.baselines import rmse
from rwt.kernel_api import resolve_kernel
from rwt.middle_out import MiddleOutSplit, resolve_middle_out


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, **self.detail}


def gate_ip_fence() -> GateResult:
    """Gate 1: kernel resolution is always fail-closed to public."""
    k = resolve_kernel()
    return GateResult(
        name="IP_FENCE",
        passed=True,
        detail={
            "kernel_id": k.kernel_id(),
            "kernel_private": k.is_private(),
            "law": "RWT_KERNEL=private only if private_kernel loads; else public",
        },
    )


def gate_middle_out_mask(df, well_id: str) -> GateResult:
    """Gate 2: heel/toe split is non-empty known and predict when TVT_input masks."""
    sp: MiddleOutSplit = resolve_middle_out(df, well_id=well_id)
    ok = sp.n_known > 0 and (sp.n_predict > 0 or "TVT_input" not in df.columns)
    return GateResult(
        name="MIDDLE_OUT_MASK",
        passed=bool(ok),
        detail={
            "well_id": well_id,
            "n_known": sp.n_known,
            "n_predict": sp.n_predict,
            "ps_md": sp.ps_md,
        },
    )


def gate_residual_acceptance(
    y_true: np.ndarray,
    oof_residual_pred: np.ndarray,
    prior_pred: np.ndarray,
    *,
    prior_name: str = "prior_stack",
    last_known_pred: np.ndarray | None = None,
    floor_rmses: dict[str, float] | None = None,
) -> GateResult:
    """
    Gate 3: accept residual head only if OOF beats:
      - residual base prior
      - last_known (when provided)
      - every pure floor prior in ``floor_rmses`` (stack, pf_blend, …)

    Competition law: never ship a head that loses the best pure prior floor.
    """
    y_true = np.asarray(y_true, float)
    oof_residual_pred = np.asarray(oof_residual_pred, float)
    prior_pred = np.asarray(prior_pred, float)
    r_res = rmse(y_true, oof_residual_pred)
    r_pri = rmse(y_true, prior_pred)
    r_lk = None
    if last_known_pred is not None:
        r_lk = rmse(y_true, np.asarray(last_known_pred, float))
    floors = {str(k): float(v) for k, v in (floor_rmses or {}).items() if np.isfinite(v)}
    if r_lk is not None and np.isfinite(r_lk):
        floors.setdefault("last_known", float(r_lk))
    if np.isfinite(r_pri):
        floors.setdefault(prior_name, float(r_pri))
    best_floor = min(floors.values()) if floors else r_pri
    best_floor_name = min(floors, key=floors.get) if floors else prior_name
    beat_prior = bool(np.isfinite(r_res) and np.isfinite(r_pri) and r_res <= r_pri + 1e-12)
    beat_lk = True if r_lk is None else bool(np.isfinite(r_lk) and r_res <= r_lk + 1e-12)
    beat_all_floors = bool(
        np.isfinite(r_res) and np.isfinite(best_floor) and r_res <= best_floor + 1e-12
    )
    passed = beat_prior and beat_lk and beat_all_floors
    return GateResult(
        name="RESIDUAL_ACCEPTANCE",
        passed=passed,
        detail={
            "oof_rmse_residual": r_res,
            "oof_rmse_prior": r_pri,
            "oof_rmse_last_known": r_lk,
            "oof_rmse_best_floor": float(best_floor) if np.isfinite(best_floor) else None,
            "best_floor_name": best_floor_name,
            "floor_rmses": floors,
            "prior_name": prior_name,
            "beat_prior": beat_prior,
            "beat_last_known": beat_lk,
            "beat_all_floors": beat_all_floors,
            "decision": "ACCEPT_RESIDUAL" if passed else "FALLBACK_PURE_PRIOR",
            "delta_rmse_vs_prior": float(r_pri - r_res)
            if (np.isfinite(r_res) and np.isfinite(r_pri))
            else None,
        },
    )


def run_entry_gates() -> list[dict[str, Any]]:
    """Gates that do not need well data (boot)."""
    return [gate_ip_fence().to_dict()]
