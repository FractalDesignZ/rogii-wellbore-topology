"""
Logic quality gates (M0-M6 operational law).

Gate 1 — IP fence: public vs private kernel (fail-closed).
Gate 2 — Middle-out mask: heel prior / toe predict.
Gate 3 — Residual acceptance: LGBM / neural head only if it beats pure prior OOF.

Upgrade (feat/all-suggestions-integration):
  - gate_residual_acceptance_per_well: optional per-well fire condition
    (global Gate3 still required; residual only fires on wells that locally beat
    their own pure prior).
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


def gate_residual_acceptance_per_well(
    y_true: np.ndarray,
    oof_residual_pred: np.ndarray,
    prior_pred: np.ndarray,
    well_ids: np.ndarray,
    *,
    min_stations: int = 30,
    margin: float = 0.0,
) -> dict[str, Any]:
    """
    Per-well residual fire condition (adaptive serve helper).

    Global Gate3 must still pass. This helper only decides, for each well,
    whether the residual head locally beats the pure prior on that well's
    OOF stations. Wells that fail the local test fall back to pure prior.

    Returns:
      well_decisions: dict well_id → {"use_residual": bool, "rmse_res": float, "rmse_pri": float, "n": int}
      n_residual_wells, n_prior_wells, adaptive_oof_rmse (if enough stations)
    """
    y_true = np.asarray(y_true, float)
    oof_residual_pred = np.asarray(oof_residual_pred, float)
    prior_pred = np.asarray(prior_pred, float)
    well_ids = np.asarray(well_ids).astype(str)

    decisions: dict[str, dict[str, Any]] = {}
    for wid in np.unique(well_ids):
        m = well_ids == wid
        if m.sum() < min_stations:
            decisions[wid] = {
                "use_residual": False,
                "rmse_res": float("nan"),
                "rmse_pri": float("nan"),
                "n": int(m.sum()),
                "reason": "too_few_stations",
            }
            continue
        r_res = rmse(y_true[m], oof_residual_pred[m])
        r_pri = rmse(y_true[m], prior_pred[m])
        use = bool(
            np.isfinite(r_res)
            and np.isfinite(r_pri)
            and r_res <= r_pri + margin
        )
        decisions[wid] = {
            "use_residual": use,
            "rmse_res": float(r_res) if np.isfinite(r_res) else float("nan"),
            "rmse_pri": float(r_pri) if np.isfinite(r_pri) else float("nan"),
            "n": int(m.sum()),
            "reason": "local_beat" if use else "local_lose",
        }

    # Adaptive serve RMSE (residual where local win, prior elsewhere)
    adaptive = np.where(
        np.array([decisions[w]["use_residual"] for w in well_ids]),
        oof_residual_pred,
        prior_pred,
    )
    adaptive_rmse = rmse(y_true, adaptive) if len(y_true) else float("nan")

    n_res = sum(1 for d in decisions.values() if d["use_residual"])
    n_pri = len(decisions) - n_res

    return {
        "well_decisions": decisions,
        "n_residual_wells": n_res,
        "n_prior_wells": n_pri,
        "adaptive_oof_rmse": float(adaptive_rmse) if np.isfinite(adaptive_rmse) else None,
        "law": "global Gate3 required; residual fires only on wells that locally beat pure prior",
    }


def run_entry_gates() -> list[dict[str, Any]]:
    """Gates that do not need well data (boot)."""
    return [gate_ip_fence().to_dict()]
