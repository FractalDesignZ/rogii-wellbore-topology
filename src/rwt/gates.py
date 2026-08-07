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


def adaptive_residual_mask(
    n: int,
    *,
    pf_confidence: np.ndarray | None = None,
    conf_threshold: float = 0.35,
    well_ids: np.ndarray | None = None,
    well_fire: dict[str, bool] | None = None,
) -> np.ndarray:
    """
    Shared train↔serve contract for where residual is allowed to fire.

    Priority (first match wins):
      1. ``well_fire`` + ``well_ids`` → station fires iff its well maps True
         (unknown well_id → False / prior-only). Needs train-time map; optional.
      2. ``pf_confidence`` → finite station conf ≥ ``conf_threshold`` (default serve rule).
      3. else all-True (no adaptive thinning; residual everywhere residual head is on).

    Global Gate3 reject is outside this helper (``use_residual`` /
    ``adaptive_serve`` False). Train adaptive OOF and ``predict_stations`` must
    call this same function so OOF mask ≡ live serve mask.
    """
    n = int(n)
    if n < 0:
        raise ValueError("n must be non-negative")
    if well_fire is not None and well_ids is not None:
        wids = np.asarray(well_ids)
        if len(wids) != n:
            raise ValueError(
                f"well_ids length {len(wids)} != n={n} for adaptive residual mask"
            )
        return np.asarray(
            [bool(well_fire.get(str(wid), False)) for wid in wids], dtype=bool
        )
    if pf_confidence is not None:
        conf = np.asarray(pf_confidence, dtype=float)
        if conf.shape[0] != n:
            raise ValueError(
                f"pf_confidence length {conf.shape[0]} != n={n} for adaptive residual mask"
            )
        return np.isfinite(conf) & (conf >= float(conf_threshold))
    return np.ones(n, dtype=bool)


def apply_adaptive_residual(
    residual_pred: np.ndarray,
    prior_pred: np.ndarray,
    fire_mask: np.ndarray,
) -> np.ndarray:
    """Blend residual vs prior with a boolean fire mask (shared train/serve)."""
    residual_pred = np.asarray(residual_pred, dtype=float)
    prior_pred = np.asarray(prior_pred, dtype=float)
    fire_mask = np.asarray(fire_mask, dtype=bool)
    if residual_pred.shape != prior_pred.shape or residual_pred.shape != fire_mask.shape:
        raise ValueError(
            f"shape mismatch residual={residual_pred.shape} prior={prior_pred.shape} "
            f"mask={fire_mask.shape}"
        )
    return np.where(fire_mask, residual_pred, prior_pred)


def gate_residual_acceptance_per_well(
    well_ids: np.ndarray,
    y_true: np.ndarray,
    oof_residual_pred: np.ndarray,
    prior_pred: np.ndarray,
    *,
    global_gate: GateResult | None = None,
    min_stations: int = 20,
    pf_confidence: np.ndarray | None = None,
    conf_threshold: float = 0.35,
) -> GateResult:
    """
    Adaptive measure helper (Layer C) + diagnostic well_fire map.

    Global Gate3 remains the hard law (must pass before any residual is allowed).

    **Parity contract (adaptive OOF ≡ serve):**
      ``adaptive_oof_rmse`` is computed with :func:`adaptive_residual_mask` under
      the **station-conf** rule (finite ``pf_confidence`` ≥ ``conf_threshold``),
      the same rule ``predict_stations`` applies when ``adaptive_serve=True``.

    **Diagnostic well_fire** (not used for adaptive_oof / default serve):
      residual OOF on that well ≤ pure prior on that well
      AND (mean pf_confidence on toe ≥ conf_threshold OR pure prior already weak).
      Optional serve path if a serialized ``well_fire`` dict is present on the blob.

    Does not override global reject.
    """
    well_ids = np.asarray(well_ids)
    y_true = np.asarray(y_true, float)
    oof_residual_pred = np.asarray(oof_residual_pred, float)
    prior_pred = np.asarray(prior_pred, float)

    if global_gate is not None and not global_gate.passed:
        return GateResult(
            name="RESIDUAL_ACCEPTANCE_PER_WELL",
            passed=False,
            detail={
                "decision": "GLOBAL_REJECT",
                "note": "global Gate3 rejected; no per-well residual fire",
                "well_fire": {},
                "n_fire": 0,
                "n_fallback": int(len(np.unique(well_ids))),
                "adaptive_oof_rmse": None,
                "adaptive_mask_mode": "global_reject",
                "conf_threshold": conf_threshold,
            },
        )

    unique = np.unique(well_ids)
    well_fire: dict[str, bool] = {}

    for wid in unique:
        m = well_ids == wid
        if m.sum() < min_stations:
            well_fire[str(wid)] = False
            continue
        r_res = rmse(y_true[m], oof_residual_pred[m])
        r_pri = rmse(y_true[m], prior_pred[m])
        beat_local = bool(np.isfinite(r_res) and np.isfinite(r_pri) and r_res <= r_pri + 1e-12)

        conf_ok = True
        if pf_confidence is not None:
            c = np.asarray(pf_confidence, float)[m]
            conf_ok = bool(np.nanmean(c) >= conf_threshold) if np.isfinite(c).any() else False

        # Diagnostic: fire if residual beats local prior AND (confident PF or prior weak)
        prior_weak = bool(np.isfinite(r_pri) and r_pri > 12.0)  # soft heuristic
        well_fire[str(wid)] = beat_local and (conf_ok or prior_weak)

    n_fire = int(sum(well_fire.values()))
    n_fb = int(len(unique) - n_fire)

    # Adaptive OOF: SAME station-conf mask as default serve (train↔serve parity).
    # well_fire is diagnostic only unless explicitly serialized onto the blob.
    n = len(y_true)
    serve_mask = adaptive_residual_mask(
        n,
        pf_confidence=pf_confidence,
        conf_threshold=conf_threshold,
    )
    mask_mode = "station_conf" if pf_confidence is not None else "all_residual"
    adaptive = apply_adaptive_residual(oof_residual_pred, prior_pred, serve_mask)
    adaptive_rmse = rmse(y_true, adaptive) if n else float("nan")

    return GateResult(
        name="RESIDUAL_ACCEPTANCE_PER_WELL",
        passed=True,  # structural helper; global Gate3 is the authority
        detail={
            "decision": "ADAPTIVE_SERVE",
            "well_fire": well_fire,
            "n_fire": n_fire,
            "n_fallback": n_fb,
            "n_station_fire": int(serve_mask.sum()),
            "n_station_fallback": int((~serve_mask).sum()),
            "adaptive_oof_rmse": float(adaptive_rmse) if np.isfinite(adaptive_rmse) else None,
            "adaptive_mask_mode": mask_mode,
            "min_stations": min_stations,
            "conf_threshold": conf_threshold,
            "global_passed": True if global_gate is None else bool(global_gate.passed),
        },
    )
