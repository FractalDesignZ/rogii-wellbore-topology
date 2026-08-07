"""
Particle-filter style typewell TVT tracker (public geosteering prior).

Lineage intent: contest winners use multi-hypothesis GR↔typewell position
tracking, not pure residual trees. This is a lean public PF:

  state  = typewell TVT index (position along vertical GR signature)
  motion = slow TVT walk along lateral MD (process noise)
  obs    = lateral GR vs typewell GR at hypothesized TVT (likelihood)

Output: per-station TVT prior + confidence for residual blend with last-known.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from rwt.middle_out import MiddleOutSplit, known_prior_tvt


def _prep_typewell(
    tw_gr: np.ndarray,
    tw_tvt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    tw_gr = np.asarray(tw_gr, dtype=float)
    tw_tvt = np.asarray(tw_tvt, dtype=float)
    m = np.isfinite(tw_gr) & np.isfinite(tw_tvt)
    tw_gr, tw_tvt = tw_gr[m], tw_tvt[m]
    # sort by TVT ascending
    order = np.argsort(tw_tvt)
    tw_tvt, tw_gr = tw_tvt[order], tw_gr[order]
    # unique TVT grid
    if len(tw_tvt) < 4:
        return tw_tvt, tw_gr, 1.0
    # approximate spacing
    d = np.diff(tw_tvt)
    d = d[d > 1e-9]
    spacing = float(np.median(d)) if len(d) else 1.0
    return tw_tvt, tw_gr, max(spacing, 0.1)


def _gr_likelihood(
    obs_gr: float,
    hyp_gr: float,
    sigma: float,
) -> float:
    if not np.isfinite(obs_gr) or not np.isfinite(hyp_gr):
        return 1.0  # no information → flat
    z = (obs_gr - hyp_gr) / max(sigma, 1e-3)
    return float(np.exp(-0.5 * z * z))


def particle_filter_tvt(
    lateral_gr: np.ndarray,
    typewell_gr: np.ndarray,
    typewell_tvt: np.ndarray,
    *,
    seed_tvt: float,
    known_tvt: np.ndarray | None = None,
    n_particles: int = 400,
    tvt_span: float = 80.0,
    process_sigma: float = 0.35,
    gr_sigma: float = 22.0,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """
    Run PF along the full lateral; returns arrays length n_stations.

    Particles live in TVT space. On stations with known_tvt (heel), particles
    are re-seeded around truth so the toe freerun starts calibrated.
    """
    rng = rng or np.random.default_rng(0)
    gr = np.asarray(lateral_gr, dtype=float)
    n = len(gr)
    if known_tvt is None:
        known_tvt = np.full(n, np.nan)
    else:
        known_tvt = np.asarray(known_tvt, dtype=float)

    tw_tvt, tw_gr, spacing = _prep_typewell(typewell_gr, typewell_tvt)
    if len(tw_tvt) < 4:
        pred = np.full(n, seed_tvt, dtype=float)
        return {
            "prior_pf_tvt": pred,
            "pf_confidence": np.zeros(n, dtype=float),
            "pf_tvt_std": np.full(n, np.nan),
        }

    # GR scale calibration on known heel (lateral vs typewell at known TVT)
    kmask = np.isfinite(known_tvt) & np.isfinite(gr)
    if kmask.sum() >= 8:
        tw_at_k = np.interp(known_tvt[kmask], tw_tvt, tw_gr)
        lat_k = gr[kmask]
        # affine: lat ≈ a * tw + b
        A = np.column_stack([tw_at_k, np.ones(len(tw_at_k))])
        try:
            coef, _, _, _ = np.linalg.lstsq(A, lat_k, rcond=None)
            a_cal, b_cal = float(coef[0]), float(coef[1])
            if not np.isfinite(a_cal) or abs(a_cal) < 1e-3:
                a_cal, b_cal = 1.0, 0.0
        except Exception:
            a_cal, b_cal = 1.0, 0.0
    else:
        a_cal, b_cal = 1.0, 0.0

    t_min, t_max = float(tw_tvt[0]), float(tw_tvt[-1])
    grid = np.arange(t_min, t_max + spacing * 0.5, spacing)
    gr_grid = np.interp(grid, tw_tvt, tw_gr)
    gr_grid = a_cal * gr_grid + b_cal  # typewell GR in lateral units

    def gr_at(tvt: np.ndarray) -> np.ndarray:
        return np.interp(tvt, grid, gr_grid, left=gr_grid[0], right=gr_grid[-1])

    lo = max(t_min, seed_tvt - tvt_span)
    hi = min(t_max, seed_tvt + tvt_span)
    particles = rng.uniform(lo, hi, size=n_particles)
    weights = np.ones(n_particles) / n_particles

    prior = np.zeros(n, dtype=float)
    conf = np.zeros(n, dtype=float)
    stds = np.zeros(n, dtype=float)

    for i in range(n):
        if np.isfinite(known_tvt[i]):
            # lock: reseed around heel truth (process continues from calibrated state)
            particles = known_tvt[i] + rng.normal(0.0, process_sigma, size=n_particles)
            particles = np.clip(particles, t_min, t_max)
            weights = np.ones(n_particles) / n_particles
            prior[i] = float(known_tvt[i])
            conf[i] = 1.0
            stds[i] = process_sigma
            continue

        particles = particles + rng.normal(0.0, process_sigma, size=n_particles)
        particles = np.clip(particles, t_min, t_max)

        obs = gr[i]
        if np.isfinite(obs):
            hyp = gr_at(particles)
            like = np.exp(-0.5 * ((obs - hyp) / max(gr_sigma, 1e-3)) ** 2)
            weights = weights * like
            s = float(weights.sum())
            if s < 1e-30:
                weights = np.ones(n_particles) / n_particles
            else:
                weights = weights / s

        ess = 1.0 / float(np.sum(weights * weights) + 1e-15)
        # confidence from ESS * inverse std (tight cloud → trust more)
        prior[i] = float(np.sum(weights * particles))
        stds[i] = float(np.sqrt(np.sum(weights * (particles - prior[i]) ** 2) + 1e-12))
        conf[i] = float(np.clip((ess / n_particles) * (1.0 / (1.0 + stds[i] / 5.0)), 0.0, 1.0))

        if ess < 0.45 * n_particles:
            cdf = np.cumsum(weights)
            u0 = rng.uniform(0, 1.0 / n_particles)
            idx = np.searchsorted(cdf, u0 + np.arange(n_particles) / n_particles)
            idx = np.clip(idx, 0, n_particles - 1)
            particles = particles[idx]
            weights = np.ones(n_particles) / n_particles

    return {
        "prior_pf_tvt": prior,
        "pf_confidence": conf,
        "pf_tvt_std": stds,
    }


def blend_pf_last_known(
    prior_pf: np.ndarray,
    last_known: float,
    confidence: np.ndarray,
    *,
    conf_floor: float = 0.15,
    conf_ceil: float = 0.55,
) -> np.ndarray:
    """
    Confidence-weighted blend: low PF confidence → last-known; high → PF.

    Guarantees we never fully abandon the strong public floor.
    """
    c = np.clip(
        (np.asarray(confidence, float) - conf_floor) / max(conf_ceil - conf_floor, 1e-6),
        0.0,
        1.0,
    )
    return (1.0 - c) * float(last_known) + c * np.asarray(prior_pf, float)


# Diverse PF hyperconfigs inspired by top private LB multi-PF bagging
_PF_BAG_CONFIGS: list[dict[str, Any]] = [
    {"n_particles": 450, "process_sigma": 0.22, "gr_sigma": 16.0, "tvt_span": 60.0, "seed": 0},
    {"n_particles": 450, "process_sigma": 0.40, "gr_sigma": 24.0, "tvt_span": 90.0, "seed": 1},
    {"n_particles": 550, "process_sigma": 0.30, "gr_sigma": 20.0, "tvt_span": 70.0, "seed": 2},
    {"n_particles": 350, "process_sigma": 0.18, "gr_sigma": 28.0, "tvt_span": 50.0, "seed": 3},
]


def particle_prior_for_well(
    lateral: pd.DataFrame,
    typewell: pd.DataFrame | None,
    split: MiddleOutSplit,
    *,
    n_particles: int = 400,
    seed: int = 0,
    bag: bool = True,
) -> dict[str, Any]:
    """
    Full-well PF prior + multi-PF bagging + last-known blend.

    On known heel, locks to observed TVT_input/TVT. Bagging averages diverse
    process/obs noise configs (6th-place style multi-PF ensemble, public lite).
    """
    n = len(lateral)
    stats = known_prior_tvt(lateral, split)
    last = float(stats["prior_last_tvt"])
    if typewell is None or "GR" not in typewell.columns or "TVT" not in typewell.columns:
        arr = np.full(n, last, dtype=float)
        return {
            "prior_pf_tvt": arr,
            "prior_pf_blend": arr.copy(),
            "prior_pf_bag": arr.copy(),
            "pf_confidence": np.zeros(n),
            "pf_tvt_std": np.full(n, np.nan),
            "prior_last_tvt": last,
            "n_pf_bag": 0,
        }

    gr = lateral["GR"].to_numpy(dtype=float) if "GR" in lateral.columns else np.full(n, np.nan)

    known_tvt = np.full(n, np.nan)
    if "TVT_input" in lateral.columns:
        known_tvt = np.asarray(lateral["TVT_input"].to_numpy(dtype=float), copy=True)
    if "TVT" in lateral.columns:
        tcol = lateral["TVT"].to_numpy(dtype=float)
        kidx = np.asarray(split.known_idx, dtype=int)
        if len(kidx):
            miss = ~np.isfinite(known_tvt[kidx]) & np.isfinite(tcol[kidx])
            known_tvt[kidx[miss]] = tcol[kidx[miss]]

    seed_tvt = last if np.isfinite(last) else float(np.nanmedian(typewell["TVT"]))
    if np.isfinite(known_tvt).any():
        seed_tvt = float(known_tvt[np.isfinite(known_tvt)][-1])

    tw_gr = typewell["GR"].to_numpy(dtype=float)
    tw_tvt = typewell["TVT"].to_numpy(dtype=float)

    def _run_one(cfg: dict[str, Any]) -> dict[str, np.ndarray]:
        return particle_filter_tvt(
            gr,
            tw_gr,
            tw_tvt,
            seed_tvt=seed_tvt,
            known_tvt=known_tvt,
            n_particles=int(cfg.get("n_particles", n_particles)),
            tvt_span=float(cfg.get("tvt_span", 80.0)),
            process_sigma=float(cfg.get("process_sigma", 0.35)),
            gr_sigma=float(cfg.get("gr_sigma", 22.0)),
            rng=np.random.default_rng(int(cfg.get("seed", seed))),
        )

    if bag:
        runs = [_run_one(cfg) for cfg in _PF_BAG_CONFIGS]
        pf_tvt = np.mean(np.vstack([r["prior_pf_tvt"] for r in runs]), axis=0)
        conf = np.mean(np.vstack([r["pf_confidence"] for r in runs]), axis=0)
        stds = np.mean(np.vstack([r["pf_tvt_std"] for r in runs]), axis=0)
        n_bag = len(runs)
    else:
        single = _run_one(
            {
                "n_particles": n_particles,
                "process_sigma": 0.35,
                "gr_sigma": 22.0,
                "tvt_span": 80.0,
                "seed": seed,
            }
        )
        pf_tvt, conf, stds = single["prior_pf_tvt"], single["pf_confidence"], single["pf_tvt_std"]
        n_bag = 1

    last_f = last if np.isfinite(last) else seed_tvt
    blend = blend_pf_last_known(
        pf_tvt,
        last_f,
        conf,
        conf_floor=0.25,
        conf_ceil=0.70,
    )
    is_known = np.zeros(n, dtype=bool)
    if len(split.known_idx):
        is_known[np.asarray(split.known_idx, dtype=int)] = True
    lock = is_known & np.isfinite(known_tvt)
    blend = np.where(lock, known_tvt, blend)
    pf_tvt = np.where(lock, known_tvt, pf_tvt)
    bag_arr = pf_tvt.copy()

    return {
        "prior_pf_tvt": pf_tvt,
        "prior_pf_blend": blend,
        "prior_pf_bag": bag_arr,
        "pf_confidence": conf,
        "pf_tvt_std": stds,
        "prior_last_tvt": last_f,
        "n_pf_bag": n_bag,
    }
