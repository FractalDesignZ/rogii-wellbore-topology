# Crown path (public open-core) — optimization axis

**RESTRICT:** no Kaggle submit. Measure = GroupKFold OOF + Gate3 + seals.

## Goal

Push pure prior floor and residual toward best-in-class (~6–7 LB) without cloning
competitors: typewell stalk fidelity + clamp law + residual only if it beats floors.

## Axes in this branch

| Axis | Change |
| --- | --- |
| **PF stalk** | Process bias from heel ΔTVT; GR level × ΔGR likelihood; GR-dropout noise inflate; 5-config bag |
| **Prior ensemble** | `prior_best` = conf-weighted clamp ensemble (last, stack, PF blend, local MD, neighbor) |
| **Neighbor** | Heel median offset calibration (capped ±40 ft) before transfer |
| **Residual** | Default base `prior_best`; Gate3 vs all floors; slightly stronger LGBM reg |
| **Clamp** | All pure candidates still ≤ last_known ± 18 ft |

## What still is RESIDUE

- Full 773w residual ~14.5 (pre-crown branch) vs LB ~6–7.
- Private beam / DTW / sheaf stalk offline.
- Public PF is still discrete particle — not Viterbi/HMM full lattice.

## Measure protocol

```bash
python -m rwt.train --data data/raw --out artifacts/crown_12w --max-wells 12 --folds 3 --workers 6
# compare baseline_rmse prior_best vs last_known / stack and Gate3 decision
```
