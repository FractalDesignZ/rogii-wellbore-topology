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
| **Residual** | Default base **`prior_stack`**; toe-weighted L2; dual seed bag; Δ clip vs base ±18; fold-safe medians; Gate3 |
| **Clamp** | Pure priors ≤ last±18; residual Δ ≤ base±18 (may leave last band when G3 wins) |
| **RESTRICT** | No Kaggle submit — OOF/Gate3/seals only |

## What still is RESIDUE

- Full 773w residual ~14.5 vs LB ~6–7 / hunt 5 ft.
- Private beam / DTW / sheaf stalk offline.
- Public PF is discrete particle — not full lattice Viterbi.

## Measure (12w crown set-5 / set-4 honest)

| Floor / head | OOF RMSE |
| --- | ---: |
| residual on **prior_stack** (Gate3 ACCEPT) | **11.08** honest OOF |
| residual set-3 | 11.27 |
| residual on prior_last_tvt | 12.30 |
| prior_stack (best pure) | **12.33** |
| prior_best (stack-led) | **12.65** |
| last_known | 13.41 |
| prior_ncc (seed-local) | **16.27** |

### Rejected A/B (set-5)

| Lever | 12w residual | Note |
| --- | ---: | --- |
| Huber objective | 11.56 | worse than L2 |
| trust×disagreement shrink | 11.50 | over-shrinks good residual |
| re-clamp residual to last±18 | 11.35 | truth can leave pure band |
| md_frac + stack_minus_last feats | 11.34 | collinear noise |

### Kept (set-5 coherence)

Soft neighbor → stack **before** `prior_best` (ensemble saw stale stack). Residual Δ clip vs base only.

## Measure protocol

```bash
python -m rwt.train --data data/raw --out artifacts/crown_12w --max-wells 12 --folds 3 --workers 6
# compare baseline_rmse prior_best vs last_known / stack and Gate3 decision
```
