# Experimental BASE92 block residual model

This optional standard baseline provides a separate training and inference route.
It is development research, with no predictive admission. It contains no optional
operator provider and does not change the existing train or submission defaults.

The ordered 92 features use MD/X/Y/Z, observed GR, visible `TVT_input`, and the
paired typewell's TVT/GR. Target TVT, target truth aliases and tops are excluded
from feature extraction and prediction. Features include visible-heel TVT/MD
slope, heel-only GR calibration/noise, geometry and support summaries, and a
41-offset GR mismatch profile over last-known TVT ±100 ft at 5 ft spacing.
Coordinates and TVT use the existing source feet convention; no conversion is
performed. Finite coordinate values resembling curve null codes remain valid.

Consecutive toe blocks contain eight original rows, retaining the final shorter
block. A training label comes from each block's lower-middle station; its sample
weight is its represented row count. Each block residual is broadcast back to
those original toe rows. Prediction preserves all visible heel values exactly.
Missing GR remains missing, and stored training medians fill missing features.
Nonfinite model residuals use an explicitly flagged last-known fallback. There
is no residual clamp or score-selected prediction mode.

The fixed recipe is three GroupKFold splits by well; L2 LightGBM, 31 leaves,
minimum 100 samples per leaf, learning rate 0.05, 600 requested rounds, seed
20261118 and four threads. No validation target is supplied for fitting or early
stopping. Each fold fits its own medians on training wells only. After OOF
measurement, a final model and medians are fitted on all selected development
blocks using the same recipe. Final-refit predictions are not OOF evidence.

The CLI requires a manifest with disjoint `development_well_ids` and
`confirmation_well_ids`. Optional repeated `--well-id` values must be development
members. Role validation occurs before any source CSV is opened. There is no
implicit discovery of all wells. A manifest describes the caller's reservation;
the software cannot infer a reservation that the manifest omits.

```powershell
python -m rwt.block_model fit --data-root data/raw/example --manifest cohort.json --well-id DEV_A --well-id DEV_B --well-id DEV_C --out artifacts/block_experiment
python -m rwt.block_model predict --model artifacts/block_experiment/model --lateral target.csv --typewell reference.csv --well-id TARGET --out artifacts/target_predictions.csv
```

Both output paths must be fresh. Fit freezes source/cohort/input hashes and runs
one explicit subprocess with the existing execution trace launcher, bounded to
600 seconds. This is a wall-time bound, not a security sandbox. It retains exact
OOF station keys, fold IDs, truth, last-known, residual and selected predictions
in NPZ/CSV, plus per-well SSE, pooled/fold scores, fold medians and fallback
counts. All expected toe rows must be finite and scored; incomplete labels or
coverage fail rather than produce a finite-subset score. The saved NPZ is reopened
for exact scalar/key replay before a report is written. This readback uses the
same implementation and is not an independent method or human review. Frozen
source and selected input hashes are checked again after fitting; the controller
source closure includes the execution trace implementation.

Python users can call `build_block_features`, `fit_block_model`,
`save_block_predictor`, and `load_block_predictor`; the loaded predictor exposes
`predict_well`. The in-memory fit API uses explicitly supplied, caller-declared
development well pairs; unlike the CLI, it does not infer protected holdouts.
A lazy label loader can keep target access separate from feature construction.

Bundles contain LightGBM text, finite NPY medians and JSON metadata. Loading
requires matching model kind, ordered feature schema, recipe, implementation
hashes, artifact hashes and LightGBM version. No pickle is used. This model kind
is explicitly experimental and records `predictive_admitted: false`. Publishing
this implementation does not publish a trained field model or establish field
performance. Independent confirmation and a frozen promotion protocol remain
separate work.
