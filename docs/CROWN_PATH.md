# Crown path (public open-core) — optimization axis

**RESTRICT:** no Kaggle submit. Measure = GroupKFold OOF + Gate3 + seals.

## Goal

Push pure prior floor and residual toward best-in-class (~6–7 LB) without cloning
competitors: typewell stalk fidelity + clamp law + residual only if it beats floors.

## Axes in this branch (`feat/all-suggestions-integration`)

| Axis | Change |
| --- | --- |
| **PF stalk** | Process bias from heel ΔTVT; GR level × ΔGR likelihood; GR-dropout noise inflate; 5-config bag |
| **Prior ensemble** | `prior_best` = conf-weighted clamp ensemble (last, stack, PF blend, local MD, neighbor) |
| **Neighbor** | Multi-neighbor weighted TVT (`neighbor_tvt_wmean`) + heel median offset (capped ±40 ft) |
| **Residual features** | Relative interactions: `pf_conf_x_rel_md`, `curv_x_pf_std`, `ncc_peak`, `ncc_lag_consistency`, `nb_offset_x_ok` |
| **Adaptive Gate3** | Global Gate3 still hard law; `gate_residual_acceptance_per_well` fires residual only on wells that locally beat pure prior |
| **Clamp** | All pure candidates still ≤ last_known ± 18 ft |
| **Compute** | Main public path = pure CPU (NumPy/OpenBLAS + LightGBM). Optional CUDA TCN residual (`neural_residual.py`) under identical Gate3; not forced |

## Opportunity space encoded

- Spatial consistency lift (multi-neighbor)  
- Residual feature honesty via relative interactions  
- Per-well adaptive serve (protect easy wells, attack hard ones)  
- Private kernel / beam / DTW / sheaf stalk remain offline (fail-closed)  
- Neural residual path available but dormant on main train line  

## What still is RESIDUE

- Full 773w residual ~14.5 (pre-upgrade) vs LB ~6–7.  
- Private beam / DTW / sheaf stalk offline.  
- Public PF is still discrete particle — not Viterbi/HMM full lattice.  

## Measure protocol

```bash
python -m rwt.train --data data/raw --out artifacts/crown_upgrade --max-wells 12 --folds 3 --workers 6
# compare baseline_rmse prior_best vs last_known / stack and Gate3 decision
# adaptive serve RMSE available via gate_residual_acceptance_per_well after OOF
```
