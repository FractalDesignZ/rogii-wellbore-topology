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

## CodeRabbit

Fair-usage rate limits often blocked full inline review. Local max-op review substituted; unit tests (48) green at merge.

## Sealed MEASURE (this kit)

| Cohort | Selected OOF (ft) | Notes |
|--------|------------------:|-------|
| 48-well default | ~11.12 | pure prior_stack G3 reject residual |
| 64-well max-prec | ~11.35 | residual G3 accept |
| **773-well max-prec** | **~14.67** ⚠ | residual G3 accept vs stack 15.11 / last_known 15.91 |
| CUDA TCN 48-well | 13.78 residual | **G3 REJECT** → pure 11.12 |
| Hunt / crown | **5 / ~6–7** | NEED_IMPROVE — private beam/DTW |

⚠ **Unresolved (V&V):** the 773-well figure is quoted as ~14.67 here and as
14.54 in `README.md` / `KB_TOPOLOGICAL_WELLBORE_RWT.md`. Both describe a Gate3
*accept* run, where selected == residual, so they cannot both be right. The
cited artifact `artifacts/full_train_parallel/cv_report.json` is gitignored and
absent, so this is not resolvable from source — re-run and commit the seal.

⚠ **Selection optimism (V&V):** the two **48-well** rows are Gate3-*reject*
scores, i.e. `min()` over ~11 pure priors scored on the rows they are then
quoted against. Measured cost of that argmin on held-out data: +28% on the
winner. The 64-well and 773-well rows are accept-path and unaffected.
See [`VERIFICATION_VALIDATION.md`](VERIFICATION_VALIDATION.md) F2/F3.

## Proceed

Public path sealed. Crown gap → private_kernel beam/DTW (gitignored), not more public LGBM alone.
