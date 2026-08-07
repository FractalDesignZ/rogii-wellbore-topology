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
    process_bias: float | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """
    Run PF along the full lateral; returns arrays length n_stations.

    Particles live in TVT space. Crown-path upgrades vs naive RW:
      - heel-calibrated affine GR map typewell → lateral
      - process bias = median heel ΔTVT/station (formation walk, not pure noise)
      - likelihood: GR level × ΔGR match (shape, not only level)
      - GR dropout: inflate process noise, no false high confidence
      - adaptive process noise from previous particle spread
      - dynamic shape weight when level match is weak
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
            "process_bias": 0.0,
        }

    # GR scale calibration on known heel (lateral vs typewell at known TVT)
    kmask = np.isfinite(known_tvt) & np.isfinite(gr)
    if kmask.sum() >= 8:
        tw_at_k = np.interp(known_tvt[kmask], tw_tvt, tw_gr)
        lat_k = gr[kmask]
        A = np.column_stack([tw_at_k, np.ones(len(tw_at_k))])
        try:
            coef, _, _, _ = np.linalg.lstsq(A, lat_k, rcond=None)
            a_cal, b_cal = float(coef[0]), float(coef[1])
            if not np.isfinite(a_cal) or a_cal <= 1e-3 or not np.isfinite(b_cal):
                a_cal, b_cal = 1.0, 0.0
        except np.linalg.LinAlgError:
            a_cal, b_cal = 1.0, 0.0
    else:
        a_cal, b_cal = 1.0, 0.0

    # process bias: typical TVT step along known heel (clipped)
    if process_bias is None:
        k_idx = np.flatnonzero(np.isfinite(known_tvt))
        if len(k_idx) >= 4:
            # consecutive known stations only
            dsteps = []
            for a, b in zip(k_idx[:-1], k_idx[1:]):
                if b == a + 1:
                    dsteps.append(float(known_tvt[b] - known_tvt[a]))
            if dsteps:
                # conservative walk: laterals rarely move >0.08 TVT/ft station
                process_bias = float(np.clip(np.median(dsteps), -0.08, 0.08))
            else:
                process_bias = 0.0
        else:
            process_bias = 0.0
    else:
        process_bias = float(np.clip(process_bias, -0.08, 0.08))

    t_min, t_max = float(tw_tvt[0]), float(tw_tvt[-1])
    grid = np.arange(t_min, t_max + spacing * 0.5, spacing)
    gr_grid = np.interp(grid, tw_tvt, tw_gr)
    gr_grid = a_cal * gr_grid + b_cal
    # derivative of typewell GR in lateral units (for shape likelihood)
    dgr_grid = np.gradient(gr_grid, spacing)

    def gr_at(tvt: np.ndarray) -> np.ndarray:
        return np.interp(tvt, grid, gr_grid, left=gr_grid[0], right=gr_grid[-1])

    def dgr_at(tvt: np.ndarray) -> np.ndarray:
        return np.interp(tvt, grid, dgr_grid, left=dgr_grid[0], right=dgr_grid[-1])

    seed_c = float(np.clip(seed_tvt, t_min, t_max)) if np.isfinite(seed_tvt) else 0.5 * (t_min + t_max)
    lo = max(t_min, seed_c - tvt_span)
    hi = min(t_max, seed_c + tvt_span)
    if lo > hi:
        lo, hi = t_min, t_max
    particles = rng.uniform(lo, hi, size=n_particles)
    weights = np.ones(n_particles) / n_particles

    prior = np.zeros(n, dtype=float)
    conf = np.zeros(n, dtype=float)
    stds = np.zeros(n, dtype=float)
    prev_obs = float("nan")
    gr_sig = max(float(gr_sigma), 1e-3)
    dgr_sig = max(gr_sig * 0.65, 1e-3)

    for i in range(n):
        if np.isfinite(known_tvt[i]):
            particles = known_tvt[i] + rng.normal(0.0, process_sigma, size=n_particles)
            particles = np.clip(particles, t_min, t_max)
            weights = np.ones(n_particles) / n_particles
            prior[i] = float(known_tvt[i])
            conf[i] = 1.0
            stds[i] = process_sigma
            if np.isfinite(gr[i]):
                prev_obs = float(gr[i])
            continue

        # motion: bias (formation walk) + adaptive diffusion
        # - inflate on GR dropout
        # - mild inflate when previous residual uncertainty was high (tracked via std)
        obs = gr[i]
        sig_proc = process_sigma * (1.75 if not np.isfinite(obs) else 1.0)
        # adaptive: if last station had high particle spread, allow more diffusion
        if i > 0 and np.isfinite(stds[i - 1]):
            sig_proc *= float(np.clip(1.0 + 0.15 * (stds[i - 1] / 5.0), 1.0, 1.8))
        particles = particles + process_bias + rng.normal(0.0, sig_proc, size=n_particles)
        particles = np.clip(particles, t_min, t_max)

        if np.isfinite(obs):
            hyp = gr_at(particles)
            level_like = np.exp(-0.5 * ((obs - hyp) / gr_sig) ** 2)
            # shape term: match lateral ΔGR to typewell dGR
            # weight increases when level match is weak (shape becomes primary signal)
            shape_w = 0.25
            if np.isfinite(prev_obs):
                d_obs = float(obs - prev_obs)
                step = process_bias if abs(process_bias) > 1e-6 else spacing * 0.25
                d_hyp = dgr_at(particles) * step
                # if level residual is large, lean harder on shape
                level_resid = np.abs(obs - hyp)
                shape_w = float(np.clip(0.25 + 0.20 * np.mean(level_resid > gr_sig), 0.25, 0.55))
                shape_like = np.exp(-0.5 * shape_w * ((d_obs - d_hyp) / dgr_sig) ** 2)
                like = level_like * shape_like
            else:
                like = level_like
            weights = weights * like
            s = float(weights.sum())
            if s < 1e-30:
                weights = np.ones(n_particles) / n_particles
            else:
                weights = weights / s
            prev_obs = float(obs)
        else:
            # no observation: dilute confidence, keep prev_obs for next finite
            pass

        ess = 1.0 / float(np.sum(weights * weights) + 1e-15)
        prior[i] = float(np.sum(weights * particles))
        stds[i] = float(np.sqrt(np.sum(weights * (particles - prior[i]) ** 2) + 1e-12))
        conf_raw = (ess / n_particles) * (1.0 / (1.0 + stds[i] / 5.0))
        if not np.isfinite(obs):
            conf_raw *= 0.35
        conf[i] = float(np.clip(conf_raw, 0.0, 1.0))

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
        "process_bias": float(process_bias),
    }


def blend_pf_last_known(
    prior_pf: np.ndarray,
    last_known: float,
    confidence: np.ndarray,
    *,
    conf_floor: float = 0.20,
    conf_ceil: float = 0.65,
    max_dev: float = 18.0,
) -> np.ndarray:
    """
    Confidence-weighted blend: low PF confidence → last-known; high → PF.

    Hard-clamps PF deviation from last_known to ±max_dev so a bad GR track
    cannot destroy the public floor (real-well diag: unclamped blend +6–10 ft).
    """
    last = float(last_known)
    pf = np.asarray(prior_pf, float)
    c = np.clip(
        (np.asarray(confidence, float) - conf_floor) / max(conf_ceil - conf_floor, 1e-6),
        0.0,
        1.0,
    )
    delta = np.clip(pf - last, -float(max_dev), float(max_dev))
    return last + c * delta


# Diverse PF hyperconfigs (public multi-hypothesis bag — crown-path diversity)
_PF_BAG_CONFIGS: list[dict[str, Any]] = [
    {"n_particles": 500, "process_sigma": 0.18, "gr_sigma": 14.0, "tvt_span": 55.0, "seed": 0},
    {"n_particles": 500, "process_sigma": 0.28, "gr_sigma": 20.0, "tvt_span": 75.0, "seed": 1},
    {"n_particles": 600, "process_sigma": 0.38, "gr_sigma": 26.0, "tvt_span": 95.0, "seed": 2},
    {"n_particles": 400, "process_sigma": 0.22, "gr_sigma": 18.0, "tvt_span": 65.0, "seed": 3},
    {"n_particles": 550, "process_sigma": 0.32, "gr_sigma": 22.0, "tvt_span": 80.0, "seed": 4},
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
    known_tvt = np.full(n, np.nan, dtype=float)
    if "TVT_input" in lateral.columns:
        known_tvt = np.asarray(lateral["TVT_input"].to_numpy(dtype=float), dtype=float).copy()
    if "TVT" in lateral.columns:
        tcol = np.asarray(lateral["TVT"].to_numpy(dtype=float), dtype=float)
        kidx = np.asarray(split.known_idx, dtype=int)
        if len(kidx):
            miss = ~np.isfinite(known_tvt[kidx]) & np.isfinite(tcol[kidx])
            known_tvt[kidx[miss]] = tcol[kidx[miss]]

    # Finite heel/reference TVT only — never invent 0.0 as a TVT prior
    last_f = last if np.isfinite(last) else (
        float(known_tvt[np.isfinite(known_tvt)][-1]) if np.isfinite(known_tvt).any() else float("nan")
    )

    def _nan_pf_result(fill: float) -> dict[str, Any]:
        """No valid reference: NaN priors (caller skips / residual masks). Heel still locked."""
        arr = np.full(n, fill if np.isfinite(fill) else np.nan, dtype=float)
        arr = np.where(np.isfinite(known_tvt), known_tvt, arr)
        return {
            "prior_pf_tvt": arr.copy(),
            "prior_pf_blend": arr.copy(),
            "prior_pf_bag": arr.copy(),
            "pf_confidence": np.zeros(n),
            "pf_tvt_std": np.full(n, np.nan),
            "prior_last_tvt": float(fill) if np.isfinite(fill) else float("nan"),
            "n_pf_bag": 0,
        }

    if typewell is None or "GR" not in typewell.columns or "TVT" not in typewell.columns:
        if not np.isfinite(last_f) and not np.isfinite(known_tvt).any():
            return _nan_pf_result(float("nan"))
        fill = last_f if np.isfinite(last_f) else float(known_tvt[np.isfinite(known_tvt)][-1])
        return _nan_pf_result(fill)

    gr = lateral["GR"].to_numpy(dtype=float) if "GR" in lateral.columns else np.full(n, np.nan)
    tw_gr = typewell["GR"].to_numpy(dtype=float)
    tw_tvt = typewell["TVT"].to_numpy(dtype=float)
    tw_ok = tw_tvt[np.isfinite(tw_tvt)]
    if tw_ok.size == 0:
        # typewell present but no finite TVT grid — no PF without inventing TVT
        if not np.isfinite(last_f) and not np.isfinite(known_tvt).any():
            return _nan_pf_result(float("nan"))
        fill = last_f if np.isfinite(last_f) else float(known_tvt[np.isfinite(known_tvt)][-1])
        return _nan_pf_result(fill)

    seed_tvt = last_f if np.isfinite(last_f) else float(np.median(tw_ok))
    if np.isfinite(known_tvt).any():
        seed_tvt = float(known_tvt[np.isfinite(known_tvt)][-1])
    if not np.isfinite(seed_tvt):
        return _nan_pf_result(float("nan"))

    rng_seed = int(seed)  # caller seed (not shadowed by TVT seed)

    def _run_one(cfg: dict[str, Any], seed_offset: int = 0) -> dict[str, np.ndarray]:
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
            rng=np.random.default_rng(rng_seed + int(cfg.get("seed", 0)) + seed_offset),
        )

    if bag:
        # thread bag configs (numpy/Python mix); multi-well uses process pool outside
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(4, len(_PF_BAG_CONFIGS))) as ex:
            futs = [
                ex.submit(_run_one, cfg, seed_offset=i * 17)
                for i, cfg in enumerate(_PF_BAG_CONFIGS)
            ]
            runs = [f.result() for f in futs]
        bag_arr = np.mean(np.vstack([r["prior_pf_tvt"] for r in runs]), axis=0)
        conf = np.mean(np.vstack([r["pf_confidence"] for r in runs]), axis=0)
        stds = np.mean(np.vstack([r["pf_tvt_std"] for r in runs]), axis=0)
        # single default-config track for prior_pf_tvt (distinct from bag)
        single = _run_one(
            {"n_particles": n_particles, "process_sigma": 0.35, "gr_sigma": 22.0, "tvt_span": 80.0, "seed": 0},
            seed_offset=0,
        )
        pf_tvt = single["prior_pf_tvt"]
        n_bag = len(runs)
    else:
        single = _run_one(
            {
                "n_particles": n_particles,
                "process_sigma": 0.35,
                "gr_sigma": 22.0,
                "tvt_span": 80.0,
                "seed": 0,
            }
        )
        pf_tvt = bag_arr = single["prior_pf_tvt"]
        conf, stds = single["pf_confidence"], single["pf_tvt_std"]
        n_bag = 1

    fill = last_f if np.isfinite(last_f) else seed_tvt
    blend = blend_pf_last_known(
        bag_arr if bag else pf_tvt,
        fill,
        conf,
        conf_floor=0.20,
        conf_ceil=0.65,
        max_dev=18.0,
    )
    is_known = np.zeros(n, dtype=bool)
    if len(split.known_idx):
        is_known[np.asarray(split.known_idx, dtype=int)] = True
    lock = is_known & np.isfinite(known_tvt)
    blend = np.where(lock, known_tvt, blend)
    pf_tvt = np.where(lock, known_tvt, pf_tvt)
    bag_arr = np.where(lock, known_tvt, bag_arr)

    return {
        "prior_pf_tvt": pf_tvt,
        "prior_pf_blend": blend,
        "prior_pf_bag": bag_arr,
        "pf_confidence": conf,
        "pf_tvt_std": stds,
        "prior_last_tvt": fill,
        "n_pf_bag": n_bag,
    }
