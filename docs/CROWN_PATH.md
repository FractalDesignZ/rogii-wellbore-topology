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
| **Neighbor** | Heel `TVT_input`-only median offset (capped ±40 ft raw; clamped to last±18 inside stack/prior_best); **train parallel re-finalize after attach** (serve parity) |
| **Residual** | Default base **`prior_stack`**; toe-weighted L2; dual seed bag; **GR mismatch@stalk** features; Δ clip vs base ±18; fold-safe medians; Gate3 |
| **Clamp** | Published stack/prior_best/NCC/PF ≤ last±18 (`MAX_TVT_DEV`); residual Δ ≤ base±18 (may leave last band when G3 wins). Raw neighbor_tvt feature may still show ±40 offset before blend. |
| **RESTRICT** | No Kaggle submit — OOF/Gate3/seals only |

## What still is RESIDUE

- Full 773w residual ~14.5 vs LB ~6–7 / hunt 5 ft.
- Private beam / DTW / sheaf stalk offline.
- Public PF is discrete particle — not full lattice Viterbi.

## Measure (12w crown set-10 — spatial train/serve parity)

| Floor / head | OOF RMSE |
| --- | ---: |
| residual on **prior_stack** (Gate3 ACCEPT) | **11.48** (honest, neighbor-aware train) |
| residual set-8 pre-parity | 11.06 (train skipped soft-nb; optimistic) |
| residual set-4/7 | 11.08 |
| prior_stack (best pure, neighbor-aware) | **12.62** |
| prior_stack pre-parity | 12.33 |
| last_known | 13.41 |

Set-8 GR mismatch@stalk kept. Set-10: `finalize_after_spatial` shared by parallel parent + direct build.

### Rejected A/B (set-5 / set-7)

| Lever | 12w residual | Note |
| --- | ---: | --- |
| Huber objective | 11.56 | worse than L2 |
| trust×disagreement shrink | 11.50 | over-shrinks good residual |
| re-clamp residual to last±18 | 11.35 | truth can leave pure band |
| md_frac + stack_minus_last feats | 11.34 | collinear noise |
| seed-bag median + std shrink | 11.28 | over-shrinks residual |
| 3-seed mean bag (default) | 11.25 | worse than dual seed |
| denser NCC stride=5 | ~11.25 | no residual gain |
| gr_mismatch delta \|stack\|−\|last\| | 11.14 | collinear noise |
| max-precision LGBM on 12w set-8 | 11.21 | overfit small cohort |

### Kept (set-5–9 coherence)

Soft neighbor → stack **before** `prior_best`; `prior_disagreement` post-transfer;
`compute_prior_disagreement` helper + deterministic CR test; residual Δ clip vs base only.

## Measure protocol

```bash
python -m rwt.train --data data/raw --out artifacts/crown_12w --max-wells 12 --folds 3 --workers 6
# compare baseline_rmse prior_best vs last_known / stack and Gate3 decision
```
