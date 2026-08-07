# Max-op CodeRabbit review ledger (public path)

Law: `restrict → measure → audit → OPEN|STOP`. Residue never forced.  
RESTRICT: no Kaggle submit from measure drivers.

## Landed on `main`

| Set | PR | Results → fixes |
|-----|-----|-----------------|
| **1** | #7 | OOF `row_i` map; Gate3 OOF mask; index shuffle; ONNX/private fail-closed |
| **2** | #7 | Floor subsets; empty OOF `ok=False`; structure-only ONNX; unit packs |
| **3** | #7 | G3 selected uses best-floor OOF; safe thread env parse; GPU `--workers` |
| **4** | #8 | Floors fully finite on residual OOF; private ImportError fail-closed; fold clamp; LGBM audit meta |
| **5** | #9 | nanmedian impute; `n_splits` clamp in API; `--prior-col`; unit impute |

## Open / in flight

| Set | PR | Results → fixes |
|-----|-----|-----------------|
| **1** (crown) | #11 | seed-local NCC band (alias 21.8→16.3); stack-led prior_best weights (12w) |
| **2** (crown) | #11 | (1) heel conf no bleed into toe; (2) clamp_origin freeze + walk last on known; (3) prior_best heel lock TVT_input-first; (4) TVT-sort typewell + expand empty seed band |
| **3** (crown) | #11 | residual default `prior_stack`; toe-weighted LGBM + deeper default; median feature impute; high-conf NCC soft in stack (bag arm kept). 12w residual **11.27** / pure stack **12.33** |
| **4** (crown) | #11 | dual seed bag; residual Δ clip ±18; prior_disagreement; soft neighbor→stack. Huber rejected. CR P1: fold-local medians + persist serve medians + serve refit. 12w residual **11.08** honest |
| **5** (crown) | #11 | neighbor soft **before** prior_best; reject last-reclamp / trust-shrink / collinear feats. Score hold 11.08. |
| **6** (crown CR) | #11 | heel-safe soft nb; MAX_TVT_DEV; serve zero-fill; **prior_disagreement recompute after neighbor** (OPEN→fixed) |
| **7** (crown CR) | #11 | `compute_prior_disagreement` helper; deterministic post-neighbor test (no skip); reject bag-std/3-seed/NCC-dense A/B. Score hold **11.08** |
| **8** (crown) | #11 | GR mismatch@stack/last/ncc (heel-cal typewell); `_prior_stack_pre_neighbor` col (no attrs concat break). 12w residual **11.06** |
| **9** (crown) | #11 | shared `fit_heel_gr_affine` once per well; reject mismatch-delta feat + maxp-12w. Score hold **11.06** |
| **10** (crown CR) | #11 | **P1 spatial parity**: finalize/seal shared; single kernel after final spatial |
| **11** (crown) | #11 | `STACK_SOFT_NB_WEIGHT=0` (soft-nb into residual base hurt stack 12.33→12.62 / res 11.06→11.48); neighbors remain in prior_best. Residual hold **11.06**. Tests: default-off stack==pre_nb; trusted nb shifts prior_best; weight 0.12 moves stack. |
| **12** (crown CR) | #11 | fold_best_iters only (no fold boosters); seal-phase timing; finalize guard/reuse nb; CR set-11 test polish |
| **13** (crown CR) | #11 | `select_n_refit` + tests; residual MD-smooth hook train/serve (**w=21 rejected 11.09**; default 0); `fold_best_iters` + `residual_smooth_window` in report/blob; CountingKernel `super()`; tautology test cleanup. Soft-nb P2 closed. |
| **14** (crown) | #11 | residual default base **`prior_best`** (12w OOF **10.95** vs stack-base 11.06); pure floor still prior_stack 12.33; PRIOR_FALLBACKS reordered. |
| **15** (crown) | #11 | dual residual heads best+stack train/serve; weights **0.75/0.25** → **10.948** (equal-mean 10.985 rejected); blob `residual_heads` + `dual_residual_weights`; tests dual serve. |
| **16** (crown CR) | #11 | named dual weights by base (not position); thread-pool seal (CR serial seal); deep LGBM under dual → **10.836**; reject `prior_delta_best_stack` feat 10.955; tests weight-by-name. |
| **17** (crown) | #11 | dual residual **off** (dilutes best head); single deep `prior_best` **10.830**; `CROWN_DEEP_RESIDUAL=True`; n_refit floor 400 deep; dual infrastructure retained opt-in. |

## CodeRabbit

Fair-usage rate limits often blocked full inline review. Local max-op review substituted; unit tests green at merge.

**Open P2 (set-17 follow-ups):** GR mismatch missing-`prior_stack` edge; residual
smoothing without `well_id`; dual-serving distinct-base stress; deep `n_refit`
integration via full `train_lgbm_residual` path (constants unit-tested).

## Sealed MEASURE (this kit)

| Cohort | Selected OOF (ft) | Notes |
|--------|------------------:|-------|
| 48-well default | ~11.12 | pure prior_stack G3 reject residual |
| 64-well max-prec | ~11.35 | residual G3 accept |
| **773-well max-prec** | **~14.67** | residual G3 accept vs stack 15.11 / last_known 15.91 |
| CUDA TCN 48-well | 13.78 residual | **G3 REJECT** → pure 11.12 |
| Hunt / crown | **5 / ~6–7** | NEED_IMPROVE — private beam/DTW |

## Proceed

Public path sealed. Crown gap → private_kernel beam/DTW (gitignored), not more public LGBM alone.
