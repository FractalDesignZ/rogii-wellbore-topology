# Verification & Validation

Independent V&V pass over the public tree at `d73c4c5`.

**Verification** — does the code do what the docs say it does?
**Validation** — do the reported numbers mean what they are quoted as meaning?

Reproduce with:

```bash
python -m unittest discover -s tests -v      # 55 tests
python scripts/vv_harness.py --out artifacts/vv
```

## Scope limit — read this before quoting anything below

The Kaggle cohort is not in the repo (`data/raw/` is gitignored) and neither is
the artifact the headline cites. **No claim about the 773-well cohort was
verified here.** Everything validated below ran on the synthetic cohort built by
`scripts/vv_harness.py`, where ground truth is known by construction. That
validates the *machinery* — assemble, GroupKFold, Gate3, submit — not the
field numbers.

## What passed

| # | Check | Result |
|---|-------|--------|
| V1 | Unit suite | 55/55 green |
| V1 | End-to-end: assemble → GroupKFold → Gate3 → submit | clean on 16 train / 8 held-out wells |
| V1 | Gate3 fail-closed on degenerate input (all-NaN, empty, inf) | rejects in every case |
| V1 | Kernel boundary — `RWT_KERNEL=private` with no `private_kernel` | falls back to public, no crash |
| V1 | `export_competition_bundle.py` output | `submission.csv` + `MANIFEST.sealed.json` only |
| V1 | Provenance seal determinism | `payload_sha256` and `public_code_tree_sha256` stable across runs |
| V1 | `feature_matrix` leak guard | raises on injected `ANCC..BUDA` form-surface columns |
| **V2** | **Reported OOF vs true held-out error** | **OOF 0.548 vs held-out 0.489 — conservative, not optimistic** |

V2 is the load-bearing one. Every headline in this repo is an OOF claim, and an
OOF number is only worth its agreement with wells the model never saw. On the
synthetic cohort the pipeline's OOF *understated* its own held-out performance
by 11%, which is the safe direction. The GroupKFold-by-well → Gate3 → submit
contract holds end to end.

## Findings

### F1 — Gate3 compared RMSEs computed on different row sets `[fixed]`

`rmse()` masks to its own finite pairs, and `gate_residual_acceptance` called it
once per predictor. A residual that was NaN on the rows it handled badly got
scored on the easy remainder while the prior was scored on everything.

Demonstrated pre-fix: a residual **10× worse than the prior on every row where
both are defined** returned `ACCEPT_RESIDUAL`, because it was scored on 6 rows
(RMSE 1.0) against a prior scored on 10 (RMSE 6.3).

Not reachable from either production caller — `train.py` and
`neural_residual.py` both pre-mask — so no shipped number is affected. But
Gate3 is the component whose entire job is refusing bad heads, and it should
not depend on caller discipline for that. It now enforces a shared finite mask
itself and reports `n_scored_rows` / `n_dropped_nonfinite_rows`.

Regression: `tests/test_vv_row_parity.py::TestGate3RowParity`.

### F2 — `oof_rmse_selected` carries selection optimism on the Gate3-reject path `[instrumented]`

When Gate3 rejects, `train.py` picks the serving prior by `min()` over ~11
candidates scored on the OOF rows, then reports that minimum as the sealed
score. The minimum of K noisy estimates is biased low — the winner's curse. The
number that gets quoted is the one that won a lottery on the rows it is quoted
against.

Measured on held-out wells (`vv_harness` V3):

| | reported | true held-out |
|---|---:|---:|
| argmin pick (`particle_filter`) | 0.5984 | **0.7688** (+28%) |
| actually-best floor (`linear_md_extrap`) | 0.7532 | 0.6796 |

The argmin pick was not even the best floor out of sample — selection regret
+0.089 ft — and the per-fold picks disagreed (`pf_bag` / `pf_tvt` alternating),
which is the signature of choosing on noise.

Two directions, opposite signs, worth separating:

- The Gate3 **decision** uses the same downward-biased `min(floors)` as the bar
  the residual must clear. That makes acceptance *harder* — conservative, safe.
- The **reported number** on rejection is that same biased minimum quoted as an
  achievement. That overstates.

`train_lgbm_residual` now also computes `selection_audit` via
`nested_floor_selection()`: the floor is chosen on the other folds and scored on
the held-out fold, so the cost of selecting is priced in. Both figures ship in
`cv_report.json`, and `oof_rmse_selected_is_argmin` flags which path produced
the headline. `oof_rmse_selected` semantics are unchanged so old seals stay
comparable — the honest figure sits beside it rather than silently replacing it.

**Which sealed numbers this touches.** From `MAXOP_CR_REVIEW_LEDGER.md`, the
reject-path rows are the exposed ones: **48-well ~11.12** and **CUDA TCN 48-well
→ pure 11.12**. The 64-well and 773-well rows are Gate3-*accept*, where the
score is the residual's own OOF and no floor selection occurs — those are not
affected by F2.

On the synthetic run the residual's OOF (0.548) still beat the *nested* honest
floor (0.617), so Gate3's ACCEPT survived the correction.

### F3 — the headline metric disagrees with itself and its artifact is absent `[documented]`

Same 773-well cohort, four different numbers:

| Source | Figure |
|--------|-------:|
| `README.md`, `KB_TOPOLOGICAL_WELLBORE_RWT.md` | 14.54 |
| `MAXOP_CR_REVIEW_LEDGER.md` | ~14.67 |
| `CROWN_PATH.md` | ~14.5 |

All three describe a Gate3-**accept** run, where `oof_rmse_selected` *is*
`oof_rmse_residual` — so 14.54 and 14.67 cannot both be right. The KB cites
`artifacts/full_train_parallel/cv_report.json` as the evidence, but
`artifacts/**` is gitignored and the file is not in the repo, so the discrepancy
cannot be resolved from source. This is the M6 CERTIFY "artifact trail" claim
without the artifact.

Not silently reconciled — picking one number without the run that produced it
would be inventing evidence. To close it: re-run
`scripts/train_max_precision.py --max-wells 0`, then commit the resulting
`cv_report.json` (or its seal block) so the figure has a checkable origin.

### F4 — IP fence used a bare `assert` `[fixed]`

`export_competition_bundle.py` guarded its outbound bundle with
`assert not (out / "private_kernel").exists()`. `python -O` strips asserts,
removing the check from exactly the hardened invocation most likely to be used
for a real export. Now an explicit `RuntimeError` that also sweeps for `.whl`
payloads and destroys the partial bundle before raising.

### F5 — spatial neighbours cross fold boundaries `[no action — measured, benign]`

`well_spatial_node_from_lateral` populates each node's `tvt` from the **full**
`TVT` column, including toe labels, not the heel-only `TVT_input`. With
GroupKFold, 20% of wells in the harness cohort drew `neighbor_tvt` from a well
sitting in the validation fold, so training-fold features contained held-out
wells' labels.

I expected this to inflate OOF. **It does not** — measured OOF came in
*conservative* (V2), and the reason holds generally: at Kaggle test time the
spatial index is built from train wells (`submit.py:105`), which are fully
labeled, so `neighbor_tvt` is legitimately available at inference with the same
semantics it had at fit time. Train and serve agree. Recorded as a known
property rather than a defect, but it is the first thing to re-measure if the
real-cohort OOF ever diverges from leaderboard performance.

## Not covered

- Any real-cohort number (no data, no artifact — see scope limit).
- `neural_residual.py` / `train_gpu_neural.py` CUDA paths — no GPU available;
  only their Gate3 call sites were read, and both pre-mask correctly.
- `dashboard_server.py`, `render_tda_network_graphics.py` — not on the
  metric-producing path.
- The private kernel, by construction.
