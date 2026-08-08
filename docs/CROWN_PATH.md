# Crown path (public open-core) — optimization axis

**RESTRICT:** no Kaggle submit. Measure = GroupKFold OOF + Gate3 + seals.

**Manifold construct:** typewell = interpolated formation grid; lateral = test manifold;
stalks **index** stations onto TVT; geometry/neighbor/conf are a **multi-dim overlay**.
See [MANIFOLD_OVERLAY.md](MANIFOLD_OVERLAY.md) (integrate → fully dimensionalize → proceed).

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
- Private beam / DTW / sheaf stalk offline (gitignored).
- Public PF is discrete particle — not full lattice Viterbi.

### Private beam pilot (local only, 40w)

| Version | mean RMSE (ft) | last_known | beats last_known |
| --- | ---: | ---: | --- |
| v1 (sliding clamp) | 11.924 | 10.788 | no |
| v2 (freeze origin + GR-rel conf + NCC reseed) | 10.756 | 10.788 | yes (−0.032) |
| lattice_blend (pre-affine) | 10.804 | 10.788 | no (reject) |
| v3 conf_gate | 10.749 | 10.788 | yes (−0.040 vs last) |
| v4.1 soft conf_gate | 10.747 | 10.788 | yes (−0.041) |
| **v6 conf_gate** (heel affine + adaptive σ) | **10.705** | 10.788 | **yes (−0.083 vs last; −0.051 vs v2 blend)** |
| pure lattice v6 (diag) | 10.686 | 10.788 | yes (−0.102) — not deploy alone |
| pure multi-field lattice v9 | **10.544** | 10.788 | yes (−0.244) — floor; not sole deploy |
| **v11 conf_gate** (pure-lat boost on multi-field) | **10.699** | 10.788 | **yes (−0.090)** deploy micro |
| heatmap blend v10 | 15.98 | 10.788 | no (reject) |

Law: private source never pushed; export predictions only. Gate3 still owns residual accept on public path.
Oracle per-well stalk min (~10.46) is diagnostic only — not a deploy prior.
Reverse_pass / dense NCC reseed / heatmap field index rejected.

## Measure (12w crown set-19 — all-suggestions reassemble)

Honest **full reassemble** after merge of `feat/all-suggestions-integration` into
crown/main (multi-neighbor + adaptive PF + interactions). Older residual-only
smokes on pre-merge stations (**10.825**) are **not** Gate3-comparable.

| Floor / head | OOF RMSE |
| --- | ---: |
| residual deep prior_best 4-seed (set-19 full assemble) | **11.045** |
| adaptive residual (2/12 wells fire) | **11.040** |
| residual-only on pre-merge stations (hist) | 10.825 |
| pure **prior_stack** (set-19) | **12.148** |
| pure prior_stack (set-18 hist) | 12.33 |
| pure prior_best (set-19) | 12.64 |
| last_known | 13.41 |

Set-19: pure stack **improved** (~12.33→12.15) via multi-neighbor + adaptive PF;
global residual rose vs stale-station smoke (new feature regime). Adaptive serve
selected when it beats global residual. Branch `feat/all-suggestions-integration`
is **absorbed** into `main` (content behind; history only).

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

## Public architecture map (research-consolidated)

| Layer | Live public reality | Residue (not public default) |
| --- | --- | --- |
| **PF** | 1-D TVT walk: bias±0.05 + Gaussian; GR×ΔGR; dropout inflate; median bag; last±18 | Joint \([TVT,\theta]\) transition; dip state \(\gamma_k\); \(E_{\mathrm{ref}}\) PF weights |
| **Dip** | MD-slope prior clip ±0.008; Z-dip **0** on laterals; heel XY ridge dip | Dip inside PF process matrix |
| **SE(3) \(E_{\mathrm{ref}}\)** | Screw-norm **proxy** \(\sqrt{\theta^2+(\|t\|/L)^2}\) → residual feature `e_ref` | Full \(se(3)\) 6-vector log \(\xi=(\omega,v)\); SVD SO(3) pose stack |
| **Match stalk** | Seed-local multi-scale NCC TVT track + conf | Full-well lag (alias) |
| **Lattice** | Discrete PF, not DP | Viterbi/Dijkstra TVT lattice \(O(N\cdot M^2)\) (private beam) |
| **Residual** | Deep LGBM on `prior_best` (4-seed); Gate3 vs pure floors | Force residual without Gate3 |

Law: **restrict → measure (GroupKFold OOF + Gate3) → audit → OPEN|STOP**. Residue never force-OPEN.

## Main integration (all-suggestions + crown) — **on main**

`feat/all-suggestions-integration` was merged via #13 then absorbed into #11 merge
(`c718b64`). Remote tip of that branch is **behind** main in content (lacks crown
sets 14–18); do not re-land it. Product stack on `main`:

| Vector | Encoding |
| --- | --- |
| Spatial multi-neighbor | `neighbor_tvt_wmean`, `neighbor_weight_sum`, `neighbor_count_close` |
| Residual interactions | `pf_conf_x_rel_md`, `curv_x_pf_std`, `ncc_peak`, `ncc_lag_consistency`, `nb_offset_x_ok` |
| Adaptive Gate3 | `gate_residual_acceptance_per_well` + serve conf gate (global Gate3 hard law) |
| Adaptive PF | process noise from prior particle std; dynamic GR shape weight |
| Crown residual | single deep `prior_best`, 4-seed, dual off, n_refit floor 400+p90 |
| Single seal | `finalize_after_spatial` → interactions → kernel → watermark once |
| GPU / neural | Optional `neural_residual` (CUDA TCN); main train CPU under RESTRICT |
| Private kernel | Fail-closed stub; beam/DTW/lattice offline |

Measure: `scripts/measure_adaptive_delta.py` or `python -m rwt.train …`
