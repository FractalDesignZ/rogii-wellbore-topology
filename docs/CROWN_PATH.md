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
| **Neighbor** | Heel `TVT_input`-only median offset; train/serve seal after attach; **stack soft-nb weight 0** (measure); neighbors in prior_best + features |
| **Residual** | Single head **`prior_best` + deep LGBM** (set-18); pure floor **`prior_stack`**; dual opt-in only; toe-weighted L2; **4-seed mean bag**; GR mismatch@stalk; Δ clip ±18; deep n_refit floor 400+p90; Gate3 |
| **Clamp** | Published stack/prior_best/NCC/PF ≤ last±18 (`MAX_TVT_DEV`); residual Δ ≤ base±18 (may leave last band when G3 wins). Raw neighbor_tvt feature may still show ±40 offset before blend. |
| **RESTRICT** | No Kaggle submit — OOF/Gate3/seals only |

## What still is RESIDUE

- Full 773w residual ~14.5 vs LB ~6–7 / hunt 5 ft.
- Private beam / DTW / sheaf stalk offline.
- Public PF is discrete particle — not full lattice Viterbi.

## Measure (12w crown set-18 — 4-seed deep + n_refit p90)

| Floor / head | OOF RMSE |
| --- | ---: |
| **single deep prior_best 4-seed** (set-18) | **10.825** |
| single deep prior_best 3-seed (set-17) | 10.830 |
| dual deep 0.75/0.25 (set-16) | 10.836 |
| drop base from X / drop ncc lags | 10.944 (rejected) |
| deep lr=0.015 / leaves=95 / l2=2 | ≥10.84 (rejected) |
| dual residual shallow 0.75/0.25 | 10.948 |
| single deep prior_stack | 11.170 |
| prior_stack (best pure) | **12.33** |
| prior_best (pure) | 12.69 |
| last_known | 13.41 |

Set-17: dual off + deep always. Set-18: deep seed bag **4** (mean); serve n_refit
uses p90 when floor≥400; rejected feature-drop and other LGBM knobs.

### Rejected A/B (set-5 / set-7)

| Lever | 12w residual | Note |
| --- | ---: | --- |
| Huber objective | 11.56 | worse than L2 |
| trust×disagreement shrink | 11.50 | over-shrinks good residual |
| re-clamp residual to last±18 | 11.35 | truth can leave pure band |
| md_frac + stack_minus_last feats | 11.34 | collinear noise |
| seed-bag median + std shrink | 11.28 | over-shrinks residual |
| 3-seed mean bag (historical shallow residual A/B; not the retained deep bag) | 11.25 | worse than dual seed |
| denser NCC stride=5 | ~11.25 | no residual gain |
| gr_mismatch delta \|stack\|−\|last\| | 11.14 | collinear noise |
| max-precision LGBM on 12w set-8 | 11.21 | overfit small cohort |
| stack soft-nb weight 0.12 | 11.48 | residual base pollution |
| well-inverse residual weights | 11.39 | hurts easy long wells |
| residual MD-smooth w=21 | 11.09 | over-smooths stalk steps |
| dual residual equal-mean | 10.985 | dilutes best head |
| prior_delta_best_stack feat | 10.955 | collinear noise on dual |
| dual residual deep (any weight) | ≥10.836 | dilutes single-best deep |
| drop residual base from X | 10.944 | trees need base level |
| deep leaves=95 / stronger L2 | ≥10.85 | underfits 12w |

### Kept (set-5–18 coherence)

Soft neighbor → stack **before** `prior_best` **only when** `STACK_SOFT_NB_WEIGHT>0`
(default **0** by measure — neighbors still enter via `prior_best` + `neighbor_*` feats);
`prior_disagreement` post-transfer; `compute_prior_disagreement` helper; residual Δ clip vs base only.
Coverage: default-off stack identity + trusted-neighbor `prior_best` pull (set-11 tests).

**P2 closed (set-17/18 CR):** missing-`prior_stack` GR guard; dual-serve distinct
bases; deep `n_refit` floor+p90 unit tests; seal per-well error isolation +
`seal_threads` actual pool size; heel_known helper; shared Gate3 floor map.

## Measure protocol

```bash
python -m rwt.train --data data/raw --out artifacts/crown_12w --max-wells 12 --folds 3 --workers 6
# compare baseline_rmse prior_best vs last_known / stack and Gate3 decision
```
