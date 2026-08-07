# Crown path (public open-core) — optimization axis

**RESTRICT:** no Kaggle submit. Measure = GroupKFold OOF + Gate3 + seals.

## Goal

Push pure prior floor and residual toward best-in-class (~6–7 LB) without cloning
competitors: typewell stalk fidelity + clamp law + residual only if it beats floors.

## Axes (public open-core, cumulative)

| Axis | Change |
| --- | --- |
| **PF stalk** | Near-PS process bias (last ≤80 known, clip ±0.05); GR×ΔGR; GR-dropout inflate; **median bag** + disagree demotion; bag/track pre-clamp ±18 |
| **NCC stalk** | `prior_ncc_tvt` sparse multi-scale template track; conf-gated in ensemble; heel lock = TVT_input∪known TVT |
| **Prior ensemble** | `prior_best` last-dominated clamp of stack/PF/local/neighbor/**NCC** (NCC weight ∝ conf only) |
| **Neighbor** | Heel `TVT_input`-only median offset (capped ±40 ft) |
| **Residual** | Default base `prior_last_tvt`; Gate3 vs all floors; LGBM + optional CUDA TCN |
| **Clamp** | All pure candidates ≤ last_known ± 18 ft |
| **RESTRICT** | No Kaggle submit — OOF/Gate3/seals only |

## What still is RESIDUE

- Full 773w residual ~14.5 vs LB ~6–7 / hunt 5 ft.
- Private beam / DTW / sheaf stalk offline.
- Public PF is discrete particle — not full lattice Viterbi.

## Measure protocol

```bash
python -m rwt.train --data data/raw --out artifacts/crown_12w --max-wells 12 --folds 3 --workers 6
# compare baseline_rmse prior_best vs last_known / stack and Gate3 decision
```
