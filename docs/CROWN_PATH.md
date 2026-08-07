# Crown path (public open-core) — optimization axis

**RESTRICT:** no Kaggle submit. Measure = GroupKFold OOF + Gate3 + seals.

## Goal

Push pure prior floor and residual toward best-in-class (~6–7 LB) without cloning
competitors: typewell stalk fidelity + clamp law + residual only if it beats floors.

## Axes (public open-core, cumulative)

| Axis | Change |
| --- | --- |
| **PF stalk** | Near-PS process bias (last ≤80 known, clip ±0.05); GR×ΔGR; GR-dropout inflate; **median bag** + disagree demotion; bag/track pre-clamp ±18 |
| **NCC stalk** | `prior_ncc_tvt` seed-local multi-scale track; TVT-sorted typewell; conf no heel→toe bleed; clamp_origin law |
| **Stack base** | `prior_stack` = last + soft dip + **median PF bag** + high-conf NCC soft pull (clamp ±18) |
| **Prior ensemble** | `prior_best` **stack-led** clamp of stack/PF/local/neighbor/NCC (NCC weight ∝ conf only) |
| **Neighbor** | Heel `TVT_input`-only median offset (capped ±40 ft) |
| **Residual** | Default base **`prior_stack`**; toe-weighted LGBM; median feature impute; Gate3 vs all floors |
| **Clamp** | All pure candidates ≤ last_known ± 18 ft |
| **RESTRICT** | No Kaggle submit — OOF/Gate3/seals only |

## What still is RESIDUE

- Full 773w residual ~14.5 vs LB ~6–7 / hunt 5 ft.
- Private beam / DTW / sheaf stalk offline.
- Public PF is discrete particle — not full lattice Viterbi.

## Measure (12w crown set-3)

| Floor / head | OOF RMSE |
| --- | ---: |
| residual on **prior_stack** (Gate3 ACCEPT) | **11.27** (was 11.49) |
| residual on prior_last_tvt | 12.30 |
| prior_stack (best pure) | **12.33** |
| prior_best (stack-led) | **12.65** (was 13.21) |
| pf_blend_last_known | 12.92 |
| last_known | 13.41 |
| prior_ncc (seed-local) | **16.27** (was 21.81 full-lag) |
| pf_bag (median) | 17.44 |

Set-3 levers: toe-weighted LGBM residual, median feature impute, high-conf NCC soft in stack, deeper default trees.
`--max-precision` 12w residual **11.12** (vs default 11.27).

## Measure protocol

```bash
python -m rwt.train --data data/raw --out artifacts/crown_12w --max-wells 12 --folds 3 --workers 6
# compare baseline_rmse prior_best vs last_known / stack and Gate3 decision
```
