# Wellbore master iteration

The objective is stronger GR-to-TVT placement with reproducible evidence. Network Engine provides relational hypotheses and changing support; the wellbore evaluator decides whether they improve placement. A local score is not an official leaderboard rank.

Ruby won the completed competition with **5.639 ft** on Kaggle's private test set. Our measurements use a separate development split. External solutions supply benchmark context only. [Final leaderboard](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/leaderboard) · [First-party design rule](../AGENTS.md)

Two linked tracks share cohort identities and scoring rules. The rapid-model track tests strong standard alignment/model candidates quickly. The operator track tests changing maps, registered image relationships and higher-order consistency. Features or priors may cross between tracks only through explicit contracts; ablations must distinguish an alignment gain from an incremental operator gain.

This is an extension of this repository. The existing Network Engine and MaxOp projects remain separate providers. Their locked plans and sealed evaluation sets are unchanged. The current wellbore controller uses explicit source manifests, a dependency graph, bounded local execution, retained outcomes, and metric recomputation. It does not claim execution through a MaxOp runtime that has not registered wellbore workers.

## Stages and decisions

| Stage | Required evidence | Next step |
| --- | --- | --- |
| Plan | First-party design and parameter origins, input identities, observation policy, source closure, cohort assignment, fixed criteria and dependency graph | Start a fresh attempt |
| Mutate / measure | Allowed observations only; explicit candidate predictions and support; retained execution result | Recompute metrics |
| Verify / audit | Artifact hashes, common-support metrics, coverage and declared gate checks | Record the decision |
| Admit | First-party provenance, including upstream inputs, and all required engineering checks pass | Admit only the stated claim |

Failures remain in their original attempt directories. New criteria or source changes require a new attempt. A synthetic construct pass never admits field performance. Source ASTs and static call sites are syntax evidence; they are not a dynamic call graph or a sandbox.

## First correctness iteration

The old WP8 harness could return success when pad transfer failed. It also read synthetic oracle truth in the donor predictor and allowed self-transfer when scoring training wells. The corrected donor interface defaults to observed `TVT_input`; full `TVT` requires an explicit training-donor policy. `TVT_true` is evaluator-only.

The public spatial API also decodes declared TVT curve null sentinels when constructing donors, consuming directly constructed nodes and calibrating a target's visible heel. Such values cannot become finite donor support or heel offsets. This curve policy leaves finite coordinates intact. Cohort-role checks reject case aliases before CSV access, including on Windows filesystems where differently cased filenames identify the same well.

```bash
python scripts/measure_manifold_gr_synth.py --donor-policy observed_only
python scripts/measure_manifold_gr_synth.py --donor-policy full_tvt
# Negative control: must reject even if the formation-book check passes
python scripts/measure_manifold_gr_synth.py --n-train 0
```

Each attempt reserves a fresh output directory. Its contract records configuration, source identity and required gates before measurement. The harness retains station arrays, per-well metrics and a hash-bound artifact closure. A separate bounded audit process recomputes metrics from those station arrays and checks the worker report. This verifies metric aggregation, not the physical validity of the generator or GR model.

Both book corroboration and aggregate test-well placement improvement must pass on complete common support. Missing difficult stations cannot improve an admitted score by disappearing. An optional `--book-mae-limit` must be declared; there is no universal GR-noise-independent ceiling.

## Field iteration

Assign development and confirmation wells by stable well-ID hashes before scoring labels. Keep the overlapping bundled sample test views separate from independent validation. Measure baselines on development wells, then compare bounded relational placement candidates under the same observations and metric coverage. Check any hard output band for an unavoidable error floor before tuning within that band.

Confirmation remains unmeasured while candidates and criteria change. A frozen candidate needs an independent confirmation evaluation before field-performance admission. Publish the controller, interfaces, tests and justified conclusions in the existing draft PR; raw competition data, generated artifacts and proprietary operator sources remain outside the public tree.

## Current checked measurements

The fixed filename-only assignment contains 619 development and 154 confirmation wells. The first development pilot contains 12 wells. These are local development measurements, with no official-rank implication.

| Checked result | Scope | Finding |
| --- | --- | --- |
| Corrected public residual baseline | All 54,833 pilot grouped-OOF rows | Residual 13.892904 ft loses; Gate3 selects last-known 11.616770 ft |
| Raw last-known comparator | All 54,833 pilot toe rows | 11.616770 ft |
| GR-only whole-path selector | Same 54,833 pilot rows, corrected coordinate policy | 11.030914 ft |
| Native-augmented whole-path selector | Same library and 54,833 rows | 11.030914 ft; identical choices, zero incremental native gain |
| Historical external Forward–Backward port | All 54,833 pilot rows, corrected coordinate policy | 60.817223 ft, excluded from the candidate pool |
| Oracle projection into heel ±18 ft | 3,034,427 development toe rows | 7.4585 ft lower bound for tracks constrained to that band |
| Oracle projection into heel ±36 ft | Same development rows | 3.8306 ft lower bound; not an attained prediction |

The earlier baseline excluded one pilot row with a nonfinite residual base. Its retained OOF evidence states that 54,832-row scope. After correcting coordinate-null ingestion, the fresh baseline retains all 54,833 rows with zero excluded rows. The auditor checks unique station keys, truth values, metadata counts and replayed grouped-fold assignments before recomputing RMSE. Oracle clipping bounds use offline development truth only and do not validate an inference algorithm. The ±18 bound does not apply to unbounded structural priors.

The first native selector remains unpromoted. Its contract, predictions, metric report and execution receipts stay local; the result directs the next operator experiment rather than being described as a successful predictive integration.

### Matched full-development comparison

A subsequent rapid-model baseline uses only each well's MD, XYZ, GR and visible `TVT_input`, plus its own typewell TVT/GR. Its 92 features summarize consecutive eight-row toe blocks, geometry, heel calibration and 41 GR costs. Three well-grouped folds use training-fold medians, represented-row sample weights and 600 fixed LightGBM rounds. Predictions are expanded back to every original station before scoring. There is no output clamp or score-based checkpoint selection.

| Candidate | Development RMSE (ft) | Paired SSE change versus preceding candidate |
| --- | ---: | ---: |
| Last-known TVT | 15.993630 | — |
| Standard 92-feature baseline | 14.812938 | −110,371,428.63 |
| Baseline + 20 cost-profile graph features | 14.656995 | −13,945,090.96 |
| Baseline + graph + 7 Julia descriptors | 14.679919 | +2,040,763.20 |

All rows use the same 619 development wells and 3,034,427 toe-station keys. The graph and Julia arms preserve the baseline feature prefix bit-for-bit, and share its folds, target labels, sample weights, learner parameters and training-median prefix. No rows disappear when a representation lacks support. Independent artifact and paired-metric audits passed. The standard baseline improves 342 wells versus last-known; the graph addition improves 327 versus the standard baseline; the Julia addition improves 309 and worsens 310 versus the graph arm.

The graph prototype infers independent transports between GR-cost profiles and checks compatible edge/triangle incidence (`d1*d0 = 0`). Its features describe a fitted potential and independently measured cycle closures. This is not a claim of Hodge decomposition, recovered physical structure or execution through the native Network Engine API. The current Julia descriptors iterate at each observation separately; they do not supply memory between stations. Their incremental result is negative and they remain unpromoted.

The standard feature cache took 31.36 seconds and its three model fits about 46.47 seconds on the local CPU. The label-blind graph/Julia lift took 22.06 seconds; the six matched model fits took 101.03 seconds in total. These are scoped measurements, not an end-to-end GPU benchmark. Source, input, prediction and execution identities remain in local retained attempts. The 154 confirmation wells remain unmeasured. These development comparisons justify further work, not field-performance admission or an official leaderboard rank.

An offline sequence comparator then used the same immutable BASE92 inputs, labels, folds, represented-row weights and station mapping. Its fixed dilated convolution recipe has a 253-block receptive field, 128 context blocks on each side of a 128-block supervised core, training-fold IQR scaling and exactly 20 epochs. Context never crosses wells. This is observation order, not a physical-time model. The last epoch is the only saved model; no validation loss or checkpoint selection is used.

| Sequence comparison | Development RMSE (ft) | Finding versus graph arm |
| --- | ---: | --- |
| Sequence model alone | 15.428782 | Worse |
| Fixed 50% sequence / 50% graph prediction | 14.157296 | All three folds improve; 373 wells improve and 246 worsen |

The blend weights and comparator prediction hashes were frozen before any sequence fit or score. The blend reduces SSE by 43,691,088.97 versus the graph arm across all 3,034,427 rows; it is not a tuned weight search. The three GPU fold processes took 152.84 seconds in total on the local RTX 5070, excluding preflight tests and CPU audit. A separate CPU audit took 5.88 seconds, replaying exact station coverage, block-center labels, row weights, well-grouped folds, saved prediction/model/preprocessing hashes and terminal traces. There were no fallback predictions. Its first attempt stopped before scoring because the auditor compared a manifest content digest with the declared file SHA; that failed attempt remains retained, with the corrected check in a fresh attempt.

This result supports complementary errors between the two tracks. It does not isolate a Julia gain or validate independent field performance; confirmation remains reserved.

An ordinary multiscale context control scores **14.674427 ft**, compared with the pointwise BASE92 model's **14.812938 ft** and the graph arm's **14.656995 ft**. It adds finite GR-cost means and descriptors, local geometry slopes and support at fixed within-well block widths 9, 33 and 129, producing 254 features with BASE92 preserved bit-for-bit. The learner, folds, labels, represented-row weights and complete 3,034,427-station scoring scope are unchanged. Saved model predictions and training-median prefixes replay exactly, with no fallback. It improves 317 wells versus BASE92 and worsens 302. Feature construction took 6.16 seconds; all six bounded test/build/fit/audit nodes totaled 113.39 seconds. Peak measured training RSS was 1.61 GB.

This similar ordinary-context gain makes a context-conditioned operator comparison necessary. A frozen 274-feature hybrid appends the same 20 graph descriptors to the context model, preserving both source feature views and their training-median slices bit-for-bit. It scores **14.534777 ft**, improving pooled SSE by 12,377,657.41 versus context alone. There are 327 well improvements and 292 regressions, with two folds improving and one regressing. This is incremental development evidence for those descriptors beyond the ordinary context control.

The preregistered 50% sequence / 50% hybrid prediction scores **14.150791 ft**, versus the earlier fixed sequence/graph blend's **14.157296 ft**. This small pooled gain reduces SSE by 558,812.48 across 308 improved wells and 311 regressions. Only fold 1 improves. All 3,034,427 station keys and saved model predictions replay, no fallback occurs, and the six reference prediction hashes remain fixed. Six bounded hybrid test/build/fit/audit nodes totaled 142.67 seconds, with peak training RSS 1.72 GB. Confirmation and field-performance admission remain separate.

The ordered-Julia experiment separately compares reset, carried-state and shuffled-order recurrence descriptors on the fixed context base. All three arms have 261 features, identical C3 prefixes and training-median prefixes, the same encoding, four seeds, 24 numerical iterations and radius 2. Carried state resets escaped seeds individually and resets all seeds at missing support; it never crosses wells. The shuffled control permutes supported runs and restores original station keys afterward. The underlying graph/context inputs are offline, so forward recurrence does not make the overall predictor causal.

| Julia ordering control | Development RMSE (ft) |
| --- | ---: |
| Reset at each observation | 14.660262 |
| Carry state in path order | 14.694861 |
| Shuffled-order carry | 14.681834 |
| Context-only reference | 14.674427 |

The frozen primary gate required carried state to reduce pooled SSE versus **both** reset and shuffled carry. It fails: carried state loses every fold versus reset, increasing SSE by 3,081,965.48 overall. It also loses to shuffled carry by 1,161,298.83. All nine fixed model fits, saved-model replays, full 3,034,427-row audit and final source/input checks pass; the predictive ordering claim remains unpromoted. An additional independent recomputation confirms the four scores, station-key uniqueness and failed gate. No seed, scale or base retry follows this result.

The label-blind transform took 19.08 seconds; nine fits totaled 343.37 seconds and the separate audit took 25.36 seconds. Maximum measured fit RSS was 1.65 GB; audit RSS was not instrumented. Neither the hybrid nor the recurrence experiment chose its base or settings from its later score. The initial context controller failed on an import path before label parsing or fitting; a corrected fresh attempt retained identical feature bytes and preserved the failure.

### Transporting local layer changes between wells

Adding 12 directional geology features to the 14.534777 ft context and graph model brought development RMSE to **11.581639 ft**. All three folds improved on the same 3,034,427 stations, with 438 well improvements and 181 regressions. The pooled reduction in squared error was 234,031,299.66 ft². The fixed half-sequence blend scored 12.126157 ft.

For a training donor, let layer height be `S = TVT + Z`. Two positions along that well give a height change and a horizontal displacement. A constant vertical datum offset cancels in the difference. Nearby donor directions constrain a local horizontal gradient, which is projected along a query step and integrated from the query's visible heel. Nearly parallel donor paths give little evidence about the transverse gradient. The model receives the donor count, alignment and directional balance alongside the estimated change.

Each outer fold rebuilds these features for fitting and validation using that fold's training donors. Every query excludes its own well. The grouped splits, model recipe, row weights and original station keys match the comparator. All three saved models reproduced their predictions. Five bounded test, fit and audit processes took 125.35 seconds in total. The 154 confirmation wells remain reserved.

A follow-up used six named formation-top elevations from other training wells as additional supervision. It fitted local planes in XY and added each inferred surface's change from the query heel, alongside support measures. The original directional feature matrix and training medians replayed byte for byte. The candidate scored **11.597061 ft** against **11.581639 ft** for the directional reference. Two folds and 318 wells improved, but the remaining fold raised pooled squared error by enough to lose the primary comparison. Training-donor horizon values were excluded from their own well's query features.

### Extending structural height from the observed heel

**Historical, excluded from active selection.** This experiment has an independently written implementation, but its design contract records consultation of a competitor's implementation for using visible-prefix structural height. The exclusion also covers its blends, graph combinations and later path policies that use its features or predictions. The numbers below document past measurements.

The query well supplies its own structural-height measurements before the hidden toe starts. At measured-depth windows of 128, 512 and 1,536 ft behind the heel anchor, a ridge fit estimates how `TVT_input + Z` changes with horizontal displacement. Projecting that gradient along the survey gives a structural-height change. Subtracting the observed change in Z converts it to a predicted TVT change. The 48 added features include fit support, geometry and agreement with the training-donor gradient. A nearly straight heel provides limited evidence about the cross-track slope.

This addition reduced full-toe development RMSE from **11.581639 to 11.358078 ft**. All three folds improved, with 332 wells improving and 287 worsening. Across 3,034,427 original stations the squared-error reduction was 15,561,830.41 ft². The 619 visible heels supplied 1,047,316 finite TVT and coordinate measurements. Each fold retained the original 286 directional features and their training medians byte for byte. All three saved models reproduced their predictions. A fixed half-sequence, half-heel blend scored 12.077808 ft.

A separate serving check rebuilt one validation query per fold from its allowed horizontal-well and typewell measurements. Each saved model used its original training donors and medians. All 612,890 feature cells and 14,672 toe-row predictions matched the development artifacts bit for bit, and all 5,137 observed heel rows retained their supplied depths. This checks the raw-input prediction path for those three wells. It uses the privately retained feature providers and donor support.

### Learned alignment over candidate depths

**Historical, excluded from active selection.** This proposal consulted a competitor's implementation when specifying the alignment task. Its results and derivatives are outside the first-party candidate pool.

A fixed two-dimensional convolution model used the measured-depth and candidate-TVT axes. It assigned mass over 41 offsets at 5 ft spacing, with explicit masks for unsupported gamma-ray costs. Each training fold used the preceding sequence model's preprocessing, an unclipped regression loss and a fixed 20th-epoch checkpoint. A separate CPU audit began after all three GPU folds saved predictions.

| Full-toe comparison | RMSE (ft) | Frozen reference (ft) |
| --- | ---: | ---: |
| Learned alignment alone | 16.468145 | 15.428782 for the preceding sequence model |
| Half learned alignment, half context and graph | 14.731799 | 14.150791 for the preceding sequence blend |

Both comparisons lost all three folds. The auditor reproduced the station keys, labels, weights and grouped folds for all 3,034,427 toe rows. All predictions were finite, with zero fallbacks. The three GPU runs took 400.66 seconds in total. Peak field process memory was 2.20 GB and peak reserved CUDA memory was 407 MB.

### Two maps contributing evidence to the same transition

The soft-overlay experiment centers a 41-depth graph on the raw fold-local directional estimate. At each successive station, a candidate can stay at its offset or move one offset bin in either direction. Survey geometry and the calibrated gamma-ray cost each supply evidence on that same edge. Unsupported measurements contribute neutral evidence. The path can therefore continue through a missing reading.

The conventional and Julia arms share the directional feature matrix, candidate grid, legal transitions, ten graph summaries and fixed 600-round learner. Both propagate evidence forward and backward within each well. The Julia arm adds bounded orbit-dependent penalties before that propagation. Both learners were trained separately with the same recipe and identical fold assignments.

| Full-toe model | RMSE (ft) | Wells improved by soft Julia overlay |
| --- | ---: | ---: |
| Directional geology reference | 11.581639 | 328 of 619 |
| Conventional candidate graph | 11.643708 | 331 of 619 |
| Same graph with two soft Julia maps | **11.506689** | — |

Against the matched graph, Julia evidence reduced pooled SSE by **9,625,306.27 ft²** and improved all three folds. Against the directional model, it reduced pooled SSE by **5,250,990.05 ft²**. Two folds improved and one regressed by 60,798.49 ft². All six saved models replayed, and a second calculation verified the exact original station keys and paired errors. The result supports this pair of penalties on development. Their individual contributions and the roles of other nonlinear scoring rules are open questions.

A post-hoc paired bootstrap resampled complete development wells 5,000 times with fixed predictions and NumPy seed 20261119. The 95% interval for the Julia-minus-conventional RMSE difference was −0.231 to −0.044 ft. Against the directional reference it was −0.189 to +0.036 ft. These intervals describe sampling variation among these wells with the fitted models held fixed.

A historical follow-up, excluded with the own-heel branch, added each graph's ten summaries to the stronger 334-feature own-heel model. The underlying graphs kept the original directional centers, evidence and transitions. The conventional version scored **11.458987 ft** and the two-map version **11.424718 ft** on all 619 wells. The two maps reduced pooled squared error by 2,379,613.04 ft² against the conventional graph and improved two of three folds. The own-heel model alone retained the lowest error at **11.358078 ft**.

We replayed the orbit penalties on an RTX 5070 using the exact edge inputs from 12 development wells. CPU and GPU agreed on every discrete outcome across 1,685,100 map positions, of which 1,582,681 had measurement support. The largest penalty difference was 2.22 × 10⁻¹⁶ in double precision. Median timings over three runs included transfers in both directions.

| Orbit calculation | NumPy CPU | RTX 5070 | CPU time / GPU time |
| --- | ---: | ---: | ---: |
| Survey geometry | 160.6 ms | 68.9 ms | 2.33× |
| Gamma-ray cost | 368.2 ms | 54.5 ms | 6.75× |

Peak GPU memory was 97.2 MB allocated and 140.5 MB reserved. These timings cover batched edge scoring. Well traversal still advances in station order, and fitting and full-row evaluation have separate costs.

### Can the graph represent the measured trajectory?

A diagnostic supplied the measured depths to the path selector after model scoring. A path had to choose one of the 41 offsets at every block, move at most one offset bin per transition and reset at each well. The minimum full-row RMSE of any such path was **1.558950 ft**. Choosing the best bin independently at every block gave **1.558898 ft**. Only 5,296 of 3,034,427 original stations lay outside the ±100 ft band around the raw directional center. This gives an upper bound on what a perfect selector could recover from this grid and a lower bound on the error of a single discrete path.

In a historical run excluded with the own-heel branch, that model supplied the desired depths. We saved the selected paths before scoring them. The same reachability rule scored **11.419827 ft**, independent rounding to the nearest bin scored **11.420237 ft**, and the continuous own-heel predictions scored **11.358078 ft**. Reachability changed 43 of 379,568 block choices versus independent rounding. Most of the remaining error comes from identifying the relevant depth using observations. The 1.558950 ft diagnostic uses hidden measurements and cannot serve as a prediction.

### First-party test of relationship weights

This pilot began with the user's instruction to weight candidate moves by their observed relationships. Its declared inputs were the project's generic path solver, the existing cohort assignments and 12 raw development well/typewell pairs. The visible heel supplied the depth anchor and an estimate of gamma-ray error. Survey Z set the moving depth center. On each crossing the gamma-ray weight reflected both measurement availability and how well the typewell signal distinguished candidate depths. The equal-weight control used exactly the same states, measurements and transitions.

All predictions were frozen before reading the hidden evaluation depths. On all **54,833** toe rows, the observation-weighted graph scored **37.614555 ft RMSE**, the equal-weight graph **35.346225 ft**, and the last observed depth **11.616770 ft**. The weights changed 9,226 predictions and improved six of 12 wells against equal weighting. The pooled error increased, so this weighting rule did not pass the development comparison. Its provenance receipt binds the first-party source, raw observations, parameter origins and predictions. A label-mutation test and a second prediction run verified the input boundary and deterministic replay.

### Switching between depth models

A separate experiment tested a persistent activity-state interpretation using two frozen out-of-fold models. At each station, the gamma-ray mismatch scored each model's proposed depth. A two-state dynamic program penalized switches and reset at each well. All route choices were saved before scoring on the original 3,034,427 toe stations.

The switch penalty reduced the number of switches from 62,449 to 8,676 and increased median run length from two to 22 blocks. The persistent route scored **12.736616 ft**, the greedy route **12.718122 ft**, and the always-directional reference **11.581639 ft**. The regularized route was more stable and had higher depth error in every fold against the directional reference. This comparator tested whether the observed gamma-ray mismatch could identify which model to trust.

The first pilot runs used the earlier coordinate-null convention. Intake review then found a valid smooth Z crossing through `-9999.25` that had been mistaken for a LAS null. The corrected reader preserves finite coordinates unless their source declares a null code. Fresh attempts retain the old runs unchanged and use stronger station-key/fold audits. The two selectors formerly scored 10.997149 ft with an incomplete path library; repaired path availability changes both to 11.030914 ft. This is an ingestion/library effect, not a native gain.

## Image-to-table relationships

`rwt.registered_views` preserves station identities, missing support and both along/cross coordinates when rotating a trajectory. Explicit noncollinear control correspondences define a world-to-pixel affine map with measured residuals and an inverse. A single vertical-plane raster loses perpendicular position; the inverse currently uses that position from the paired table. Independent views or additional constraints are required for image-only reconstruction.

Color pixels inside a declared ROI are unordered candidates until correspondence is established. Control-point fit residuals are not independent registration accuracy. Supplied training figures also contain full-TVT alignment and formation surfaces, so these panels cannot condition an unseen-toe prediction. Geometry/GR views can be reconstructed from the same allowed observations at serving. Operator-cycle and recurrence hypotheses must be evaluated separately from their visualization.

## Closing loops and revisiting earlier observations

A first-party test fitted a single structural-height potential from changes measured on training donors within each fold. Integrating a potential around its grid cells closed to floating-point precision by construction. The donor geometry supplied only one qualifying near-crossing pair and no independent nondegenerate three-well loops under the declared tolerance. On all 619 development wells, the potential scored **16.446242 ft** versus **15.073425 ft** for directional transport and **15.993630 ft** for last visible depth. The potential improved 268 wells over directional transport. Its paired well-bootstrap 95% interval for the RMSE change was −0.162 to +2.970 ft. Exact algebraic closure did not deliver a lower pooled placement error.

A separate zero-fit pilot expanded forward and backward windows using the disagreement between observed and candidate states. The adaptive Julia path changed 4,348 of 54,833 station predictions and improved eight of 12 wells against the fixed-window Julia path. Pooled RMSE was **14.136748 ft** with adaptive windows, **14.214966 ft** with fixed Julia windows, **13.689438 ft** with fixed plain costs and **11.616770 ft** with last visible depth. Fresh-process intake through process exit took 1.614, 1.570, 1.499 and 0.736 seconds respectively. The two geometries share survey ancestry. This experiment tests how window selection changes a path and measures the resulting accuracy and cost.

The public nodal harness tests a different revision rule. Each candidate edge receives support from the observations reachable on its two sides. The same bundle feeds three standard learners, with a matched control for each fit. The launcher emits source, AST and observed execution graphs and replays station errors independently. [Nodal comparison and reproduction](NODAL_HARNESS.md) · [Framework and source relationships](RELATIONAL_METHODS.md)

On all 619 development wells, the relational forest scored **15.037972 ft**, the matched forest **15.111924 ft** and last visible TVT **15.993630 ft**. It passed both pooled comparisons. Its paired well-bootstrap interval for the gain over the matched forest crossed zero. No other readout beat both controls. The revised graph changed 121,738 station predictions and scored **40.469240 ft**, preserving a measurable distinction between weight changes, path changes and useful placement.

### Returning a block prediction to the original station

A separate zero-fit experiment started with the frozen first-party two-map Julia out-of-fold predictions. At the measured-depth centers of the blocks, the existing model scored **11.506689 ft** when each center prediction was broadcast over its block. Interpolating TVT along measured depth from the last visible heel through the predicted centers gave **11.505171 ft**. Interpolating structural height, `H = predicted TVT + observed Z`, then subtracting each station's measured Z gave **11.505114 ft**. The two interpolants used the same knots and endpoint policy. No station required a missing-Z fallback.

The structural-height lift improved 529 of 619 wells and all three folds against the original broadcast. The paired well-bootstrap interval for the RMSE difference was −0.001831 to −0.001335 ft. Ordinary interpolation accounts for most of the gain. Using Z added a further **0.000057 ft** improvement, with a paired interval from −0.000069 to −0.000046 ft. Generating and sealing all three outputs took 6.74 seconds and scoring plus verification took 3.30 seconds. These times exclude training the upstream model. All 3,034,427 input station keys and raw target values replayed. Confirmation remains reserved.

The public `rwt.station_lift.lift_block_predictions(md, z, tvt_input, center_station, block_tvt, method="md_height")` accepts a single well in acquisition order. It returns the complete station-depth vector, preserving the visible heel. Set `method="md_tvt"` or `method="constant"` for the two controls. It reproduced all three sealed development prediction arrays byte-for-byte and preserved 1,047,316 visible heel readings. [Station lift API](../src/rwt/station_lift.py)

### Well-level drift and signed gamma-ray registration

In the strongest frozen station model, **63.41%** of squared error came from the mean signed error within each well, and **18.20%** came from the remaining linear drift with measured depth. The final 10% of toe distance scored **15.581 ft** compared with **4.288 ft** in the first 10%. Stations more than 3,000 measured-depth feet from the visible heel contributed **63.67%** of squared error. A retrospective fit that subtracts both components using hidden targets would give **4.934 ft**. This measures how much error a per-well offset and slope can represent. Predicting them requires independent observations.

The public `rwt.station_error.decompose_station_error(md, prediction, truth)` returns signed offset, signed drift and three additive squared errors for one well. The analysis joins to each original station and assigns one vote per row. The API replay recovered all three energies across 3,034,427 rows from separately hashed raw development CSVs. [Evaluation API](../src/rwt/station_error.py)

A zero-fit test aligned signed GR changes at two lags along the candidate depth paths. It estimated a reference offset with two disjoint visible-heel windows and then traced bounded corrections around the frozen Julia block predictions. On all 619 development wells, the unchanged block predictions scored **11.506689 ft**, signed GR with registration **12.020167 ft**, signed GR with zero registration **12.025564 ft**, and unsigned GR with registration **11.719297 ft**. The signed registered path lost every fold against the unchanged path. Its paired well-bootstrap 95% interval for the increase was +0.336 to +0.691 ft. Registration gave a nonzero offset to 51 wells, with 26 improvements and 25 regressions versus zero registration. All predictions were sealed before scoring. The tested GR correction failed the development gate.

A related zero-fit experiment forecast the last quarter of the visible heel from two earlier structural-height observations. It applied that well's signed forecast error as a toe correction of at most five feet, ramped along measured depth. It scored **11.733469 ft** versus **11.505114 ft** for the original station readout and **11.864301 ft** for assigning the same signed offsets to different wells within each fold. The same-well correction lost all three folds against the original. Its signed offset correlation with the real-toe mean error was 0.043. The visible-section test supplied little information about correction amplitude along the toe.

The learned phase experiment kept the 296 original Julia features and appended 34 geometry, GR, and graph summaries on four measured-depth intervals. It used the same three folds, represented-row weights and a common structural-height target. One readout fit 600 rounds directly. The other fit 300 rounds on the projection to four heel-anchored linear knots and 300 on its residual. The partitioned learner scored **11.925652 ft** against **12.086902 ft** for the direct fit and **11.877787 ft** for projecting the direct predictions without a second learned component. The partitioned learner beat the direct fit in all three folds and lost all three against the frozen **11.505114 ft** model. Each fold parsed TVT only for its training wells. The audit used all original toe rows after all predictions had been sealed.

### Registering neighboring training-well profiles

The registered-donor experiment treats each eligible training well as a profile of `TVT + Z` along the query's horizontal axis. It compares the change from the same longitudinal coordinate as the query's heel, which cancels the donor's vertical offset. Opposite sides of the query give a transverse bracket when both have overlap. Two disjoint parts of the query's visible heel estimate each donor's offset and measure its forecast error. The blend weighs geometric overlap and the latter error. Training queries exclude their own wells from the donor set. Held-out queries use only survey coordinates and visible `TVT_input` for these features, and every donor is from that fold's training partition.

The model appends 12 profile statistics to the original 296 Julia features, with the same 600-round learner, targets and row weights. The three matched original models reproduced their predictions byte-for-byte. On all 619 development wells, the new model scored **11.077255 ft** against **11.506689 ft** using block broadcast. All three folds improved, with **344 wells better and 275 worse**. The paired whole-well bootstrap interval for the RMSE change was **−0.732 to −0.158 ft**. The profile estimates on their own scored 24.252 ft with geometry weights and 22.559 ft with visible-heel validation. The fitted comparison measures the combined 12 features.

Using the same structural-height lift for both models gave **11.075788 ft** against **11.505114 ft**, with exact coverage of 3,034,427 original toe stations. About 77% of the improvement in squared error came from reducing the within-well mean error. The group with two-sided support at its block nodes improved from 11.393 to 10.901 ft under the block readout. The remaining 95,696 stations went from 14.564 to 15.536 ft. The split groups by whole well, so nearby separate wells may enter training. The experiment provides development evidence for the full bundle and uneven performance across support regimes. The 154 confirmation wells remain reserved.

A sensitivity check grouped the same frozen station errors into 20,000-ft squares by the visible heel coordinates. Shifting the grid by half a cell along each axis yielded 30 to 33 nonempty cells. Resampling whole cells gave paired 95% intervals below zero for all four origins. The most conservative upper endpoint was −0.053 ft. All comparisons use the original folds and fixed predictions.

That subgroup result motivated a fixed conditional readout using the 308-column prediction where two-sided support exists and the 296-column prediction elsewhere. The switch was made at the original block centers before applying the same station lift. This exploratory comparison scored **11.034085 ft** across the same cohort. It improved all three fold aggregates, changed 20 wells, and improved 12 of them. The paired 95% interval against the always-augmented model was **−0.099091 to +0.000057 ft**. The same development population motivated and evaluated the rule.

The raw serving replay covered every held-out development well from MD, XYZ, GR, visible TVT and the paired typewell. Separate per-fold packages supplied the original training donors, preprocessing and models. The replay matched all 112,352,128 original feature cells and the 4,554,816 added held-out donor-profile cells. Both the 308-column model and the two-sided support switch reproduced all 379,568 block predictions and flags and all 3,034,427 lifted original stations, preserving 1,047,316 visible heel values. The serial 308-column replay took 175.02 seconds and required zero fits. A raw-query mutation of hidden TVT and formation columns left the 296-column result unchanged. The fold packages and serving receipts remain local.

A separate execution graph reran the original donor fit and score functions. The contract binds 14 source files, their Python ASTs and the frozen inputs. Eight bounded processes recorded 16 chained events. The three fitted models, feature digests, validation prediction archives, and group assignments matched the original. The raw-station audit recovered the same errors across all development wells. The local replay receipt root is `0d513b11edf6985017c3b2abf48b1c3ef8fc5e46f3befac82a9395d0ffb1c6fb`.

### Nested transfer of donor-model error

Errors in the frozen 308-feature model had correlation **0.255** between late-toe points registered less than 1,024 ft apart in the same held-out fold. This used 263 query wells. Whole-curve permutations within each fold gave a null median of −0.011 and one-sided p = 0.002. At distances beyond 3,072 ft the measured correlation was 0.003. These label-informed observations motivated a separate prediction test.

Each outer training partition was split into three inner folds. Every fit regenerated the 34 supervised directional, Julia and donor-profile features from its inner training wells, retaining the fixed 274-feature observation matrix. The worker sealed its predicted TVT before reading the inner holdout's targets and then computed signed errors. The outer-held well geometry registered these training-only residual curves to the original 308-feature predictions. The primary rule required close donors on both sides, then applied the same heel ramp, clipping and shrinkage to the spatial, permuted-curve and mean-curve controls. [Nested role validation API](../src/rwt/crossfit_roles.py)

| Final station readout | RMSE (ft) |
| --- | ---: |
| Frozen 308-feature prediction | **11.075788** |
| Row-weighted two-sided residual transfer | 11.103838 |
| Same geometry with permuted curves | 11.103358 |
| Mean residual curve on the same support | 11.075702 |
| Wider residual kernel | 11.078424 |
| Equal-weight local curves on wider support | 11.078658 |
| Permuted curves on wider support | 11.073518 |
| Mean residual curve on wider support | 11.075374 |

The two-sided residual transfer lost to the frozen model in all three folds. Its paired whole-well interval for the RMSE change was −0.010 to +0.067 ft. The wider rule used a compact distance kernel inside 3,072 ft and reduced one-sided weights. It reached 91.39% of block centers and lost to the unchanged model in two folds. Its paired interval was −0.007 to +0.012 ft. Both fixed rules failed their preregistered comparisons. The wider permutation result is a control outcome.

The two-sided correction changed 789,692 original stations. Those rows carried **13.31%** of the baseline squared error. Substituting the observed truth on all changed rows would still leave **10.312234 ft RMSE**. Actual correction directions agreed with the signed error at 52.48% of those rows. A single multiplier optimized with the actual targets would reduce baseline RMSE by less than 0.001 ft. The observed transfer has limited reach and weak signed alignment, even though the same-model diagnostic detected spatial covariance.

The training run bound source and AST identities, 14 subprocesses and 21 dependency edges. An independent check verified all nine fit partitions, 2,476 inner-fit target reads and 1,238 post-seal residual target reads. All nine frozen fits were preserved when the audit found that the first phase-bin implementation weighted block centers equally. The corrected replay weighted each center by the original rows it represents and used the exact same residual banks. Independent raw-station error replay verified both attempts on all 619 development wells and 3,034,427 toe stations. The row-weighted evidence root is `231fff1aff9d1fa9b1c15674a17f246f314d959d69360fc30d3b592619665d1f`. The 154 confirmation wells remain reserved.

An internal test then restricted both ends of the relation to wells predicted by the exact same inner model. Each query's residual curve was withheld from its own correction. Across the nine inner holdouts, the spatial rule lowered row-weighted block-center RMSE from **12.322961 to 12.303129 ft**, compared with **12.326495 ft** for permuted donors. Seven of nine groups improved. Each development well appears in two inner training contexts, and the paired bootstrap resampled that well as a single unit. This test evaluates the inner residual banks and motivates testing the same-model rule on actual outer queries.

### Reusing the inner models on actual held-out wells

Each saved inner model recomputed its supervised features for the actual outer queries with that model's own training donors. Its spatial correction used only residuals from the inner holdout predicted by that same model. The fixed combination averaged the three inner predictions at each original block center, then averaged that result 50/50 with the registered-donor model fitted on the full outer training partition. Every arm used the same structural-height interpolation to the original survey stations.

| Prediction on the 619 outer-held development wells | RMSE (ft) |
| --- | ---: |
| Full registered-donor model | 11.075788 |
| Mean of the three inner models | 11.096827 |
| Fixed 50/50 blend | **10.900780** |
| Fixed blend with same-model spatial correction | 10.900235 |
| Fixed blend with permuted residual donors | 10.899905 |
| Fixed blend with mean residual curves | 10.900651 |

The uncorrected blend improved all three folds and 371 of 619 wells. It removed 11,670,646 ft² of squared error. A secondary comparison of the fixed predictions gave a paired 95% interval of −0.317 to −0.032 ft against the full model. The spatial correction improved slightly against the uncorrected blend and lost to its permuted-donor control. Its frozen predictive gate failed. The uncorrected ensemble became the development reference for later comparisons.

Nine inference workers reused saved models and matched 5,330 probe block predictions byte-for-byte. They produced 1,138,704 outer block predictions, three per original block. Observed inference target parses covered 2,476 training-donor files and zero outer queries. The evaluator sealed all nine readouts before opening the original target columns. A separate audit reproduced every well's errors, the pooled station scores and the primary decision. Local evidence root `efe4b6277001778ca39c714ee6bc38e95edc4eabd032a283eedf6acff4573c9a`. Confirmation remains reserved.

### Common error and a whole-well neural comparison

The four frozen components often missed in the same direction. The actual depth lay outside their prediction interval at 60.37% of stations, accounting for 94.97% of the fixed ensemble's squared error. Even using the observed targets to pick a separate convex mixture at each station would leave **7.987779 ft** RMSE. The 31 wells carrying 40.65% of squared error had a final-quarter RMSE of 37.185 ft, with truth outside all four predictions at 99.7% of stations in that late segment. Two-sided registered donor coverage existed at 93.16% of the rows of those 31 wells. The measured need is a source of predictions outside the current model range.

A fixed comparison gave two neural readouts the identical 308 features and represented-row weights. Both used the same whole-well observation mean. One updated each node independently and the other additionally used temporal convolutions spanning four dilation scales within each observed well. Both used train-only feature scaling and 24 fixed epochs on the RTX 5070. Preparation reproduced all 12 donor-profile features exactly from permitted outer training donors. All six saved checkpoints reproduced their held-out predictions.

| Native station readout | RMSE (ft) |
| --- | ---: |
| Frozen four-model ensemble | 10.900780 |
| Pointwise control | 12.254424 |
| Temporal readout | 11.870908 |
| Fixed half ensemble, half pointwise | 11.095939 |
| Fixed half ensemble, half temporal | 10.853233 |

The temporal blend improved the pooled score and one of three outer folds against the frozen ensemble. The paired 95% interval for its RMSE change was −0.267 to +0.175 ft. It failed the frozen rule requiring at least two winning outer folds. Adding both neural outputs to the truth-informed mixture calculation lowered the six-model bound to **7.007805 ft**. This quantifies potential diversity and leaves a large common error component. The GPU processes took 60.5 to 71.1 seconds per fold for both readouts. The exact 3,034,427-row score and independent error replay are bound to local evidence root `440826e9e5686cae72a2097ef1ebdd068b324157a0d1145594674e7138af5a8e`.

### Registering gamma-ray events between separate laterals

Another fixed test used signed changes in measured gamma ray to choose where to sample structural height on each of two flanking training wells. It compared signed seven-sample traces within ±96 longitudinal feet of the geometric crossing in the query frame. A separate visible-heel segment checked the registration. The equal-coordinate and rotated-event-order controls shared the same observed support and correction strength. Forecast construction read held-out measurements and visible heel TVT, took structural heights only from training donors, and sealed every station value before evaluation.

The registered readout reached 329,665 stations in 111 wells and scored **10.886076 ft** against **10.900780 ft** for the frozen ensemble. Equal-coordinate transfer scored **10.885071 ft**, and using rotated event order scored **10.885145 ft**. Actual event order lost all three folds against equal coordinates. Its improvement versus the frozen ensemble was small and the matched-control gate failed. The reached stations carried 4.43% of baseline squared error. Using the actual targets at every reached station would still leave **10.656826 ft** RMSE. Independent raw-row verification reproduced the recorded errors and confirmed opposite-side donors belonged to the allowed fold.

### Synchronizing height offsets across training donors

A separate experiment used sustained overlaps between training wells to fit one vertical offset per well in each connected donor group. The same group had to support both the visible query heel and the toe location. Three matched learners appended four features to the 308-column readout using the same geometry and missing-support mask. Their height references came from synchronized offsets, independent per-well heel datums and shuffled overlap differences. Training queries excluded their own wells from the relevant fit and lookup.

On the training overlap equations, synchronization reduced weighted RMS discrepancy across the three folds to 1.27–1.84 ft, from 58.11–65.49 ft for the independent heel datums. Transfer covered 235,193 original block centers. Replacing the outer half of the ensemble with the synchronized learner gave **10.931480 ft** against **10.900780 ft** for the byte-identical baseline, **10.897363 ft** for the heel-datum control and **10.904892 ft** for the shuffled-difference control. The synchronized arm lost all three folds against the baseline. Its paired 95% interval for the difference was −0.040 to +0.102 ft. Equation agreement within the training donor group did not carry through to held-out placement.

Nine serial fits and their sealing and scoring stages took 451.35 seconds. The independent audit reproduced all eight station scores and the exact two original comparators. Neither control replaced the leading model. Local evidence root `97f0408daf4bb422188a2701b588cf972ab82437018107ae752ac366e3480e8d`.

### Raw serving for the original four-model ensemble

The **10.900780-ft** ensemble replays from the original surveys, measured gamma ray, visible TVT and paired typewells. The preparation step initializes the three inner models with their frozen training-donor state and model parameters. The held-out query computes its own 274-column observation prefix from raw measurements. Each inner model then builds the supervised features from its own allowed donor state. The public blend checks that the full model and every inner model return exactly the original block IDs. It averages their depths at the block centers before interpolation to survey stations. [Keyed blend API](../src/rwt/ensemble.py) · [Station lift API](../src/rwt/station_lift.py)

The replay covered all 619 outer-held wells and the paired reference logs. All 350,720,832 regenerated inner feature cells, 1,138,704 inner block depth predictions and 3,034,427 saved ensemble station values matched bit-for-bit. All 1,047,316 visible heel values were preserved. The verifier also reconstructed each separate inner model's station prediction from its archived centers and the original survey. Fixtures confirmed that an unseen well identifier with the same observations produces the same result, and that swapping in synthetic lateral target and formation columns leaves the result unchanged.

The workers used at most two concurrent fold processes, with one numerical thread per worker. Observed lateral target parses were confined to 2,476 authorized training-donor reads at initialization. The replay performed zero fits and produced zero new accuracy scores. Parts of model initialization still use frozen local training artifacts. The verified proof covers the original held-well replay. Local evidence root `2916115bfcde6455df89dcc7f5c66ce190a559d7af4269a8d4c3f2c9104e2c1c`. The 154 confirmation wells remain reserved.

### Recorded trajectory ahead of each station

The forward-window learner appended 32 survey and gamma-ray features to the same 308-feature model. Each window measured changes in vertical and horizontal direction and their relationship to gamma ray over 128, 256, 512 and 1,024 measured feet. The matched learner used windows extending backward by the same distances. These features use recorded survey and gamma ray from the complete query lateral and calibrate against visible heel observations. The fixed blend retains the previous mean of the three inner models for half its weight. The measurement applies to offline interpretation of a recorded lateral. Drilling-time availability and the causal effect of steering remain untested.

| Readout | RMSE (ft) |
| --- | ---: |
| Original ensemble | 10.900780 |
| Blend using backward windows | 10.966572 |
| Blend using forward windows | **10.648607** |
| Forward-window model alone | 10.709776 |

The forward blend improved all three folds and 373 of 619 wells against the original ensemble. It removed 16,489,629 ft² of squared error. The paired well-bootstrap 95% intervals for the fixed predictions were −0.369 to −0.137 ft against the original blend and −0.480 to −0.157 ft against the backward-window control. The last third of toe stations improved from 13.927433 to 13.645620 ft. Natural near-end support differs by direction because the recorded survey ends at the toe.

The freeze binds source identities, ASTs and three fit workers that produced six fixed tree fits. The observed target reads were confined to the training partition during fitting and went through the score path after sealing. An independent audit matched all 3,034,427 original station keys and recomputed squared errors for each arm. Local evidence root `f7434eeebb03e5173d847970d487b0ca743a943f98902ab56ab5d11d49fede3b`. The separate 154-well confirmation cohort remains reserved.

A separate raw replay regenerated all 129,053,120 inputs to the forward-window model, including 12,146,176 new feature values, and all three unchanged inner-model readouts. The forward and blended center depths and all 3,034,427 toe predictions matched bit-for-bit. The replay preserved all 1,047,316 observed heel values. Its four bounded workers ran zero fits and recorded zero held-lateral target reads. Frozen donor initialization read 2,476 authorized training target files across folds. Local replay evidence root `04015ea1be9ae794bbfff320da14859e1902dc854970bdeecbccc66e88f505b4`.

### Other readouts on the same development folds

Paired typewell gamma-ray distributions gave a stronger correction than preserving exact within-window order. The distribution correction reached **10.740834 ft** on the original ensemble, with a paired interval of −0.393 to +0.064 ft for the RMSE change. Matching exact order reached 10.932973 ft. A later prespecified graph that let each window infer a local depth offset reached 11.352511 ft. Its matched fixed-offset control reached 11.334449 ft. The distribution correction combined with the forward-window blend reached **10.688545 ft** and won only one fold against the forward blend alone. The composition failed its prespecified gate. Local composition evidence root `ebe5f263ceda991c6e9858ffd7bad78705a8f74f2b0ae9a067d645db8526ecb1`.

A spatial estimate of conditional structural-height change reached **10.951606 ft**. Emphasizing training wells with large cross-fitted offset and drift errors reached **10.963244 ft**. Both scored above their original 10.900780-ft reference. Mapping named typewell horizon boundaries through elevations interpolated from training wells reached **12.327599 ft** as a fixed blend. The estimated elevations supported the original named-boundary readout on 938,811 stations across 199 wells.

### Diagnosing horizon elevation error

On that same 938,811-station population, substituting the held wells' own horizon columns reduced direct typed inversion error from 23.753091 to 0.693808 ft under the same lift and fallback. At native stations with valid exact local fields, a single named top reconstructed TVT to **0.007530 ft RMSE** across 3,024,318 rows. With a fixed top and exact heel anchor, the reconstruction is `TVT(s) = TVT(heel) − ΔZ + Δtop_Z`. The serving problem is estimating the change in local formation elevation using allowed observations. The organizer identifies these held-well formation elevations as training-only and leaves their derivation unspecified. They can encode the target and were used only after the predictive results were frozen. Local diagnostic evidence root `b7b804e50fe6c9b7ed195796bba1d207adff3f1eef78137a4384a06ac3e1e3af`.

### Registered height changes along the survey

The next test represented a section of the query as two planes perpendicular to its survey direction. Eligible training laterals supplied their signed change in `TVT + Z` between those planes. A matched control took an equal-length section at a shifted position on each of the same donors. The freeze specified 1,024 measured feet for the main test, with 256 and 4,096 feet as scale checks. The query supplied its recorded survey, visible heel and frozen forward-window predictions. Formation columns were outside this test's inputs.

The direct prediction failed its primary gate. On the 2,122 supported 1,024-ft sections, registered donor increments scored **6.777094 ft** against **6.589917 ft** for increments from the forward-window ensemble and **13.275781 ft** for the shifted-donor control. These are errors in the change across an interval. For the separately prespecified station reconstruction, the forward-window ensemble retained half the weight where donors bracketed the query and three quarters with one-sided support. Unsupported intervals kept the forward-window prediction.

Reconstructing station depths with that fixed combination passed its own gate. It reached **10.433145 ft RMSE** against **10.648607 ft** for the forward-window ensemble and **11.472314 ft** with the shifted-donor control. The combined model improved all three folds and 280 of 619 wells, with 237 worse and 102 unchanged. The paired well-bootstrap 95% interval for the change against the forward-window ensemble was −0.403 to −0.039 ft. A separate audit recomputed the primary interval errors and all 3,034,427 native station errors. Six observed stages ran in 39.03 seconds with one numerical thread and zero fits. Local evidence root `72e1a9519ffb33c44a9681701caa152c4a7841f923a76fafc5a34e880f7515ba`.

The replay independently matched the input ensemble arrays to the raw trajectory seals for each fold. Using original measured-depth rows and the recorded donor increments, it regenerated all registered and shifted-control station values bit-for-bit. This replay reused the previously computed donor increments. Local binding SHA-256 `8c88fcafc0081cc4de00629de0328ae323f1427725c599e7fea591f9e806bc5b`.

The public `lift_increment_corrections` function takes one signed correction per original interval endpoint. It anchors at the final visible station, accumulates the increments and interpolates the cumulative correction in measured depth. A zero increment carries the preceding correction through an unsupported interval. The function preserves every visible heel value and retains the existing toe trace between correction knots. [Station correction API](../src/rwt/station_corrections.py)

A subsequent serving replay regenerated all **14,679** donor interval records from raw surveys and fold-training TVT. It took the forward ensemble from the earlier raw replay after validating each fold's forecast and station-order hashes. Both the registered and phase-control forecasts matched every original toe station bit-for-bit. The public API matched an independent interpolation and preserved all 1,047,316 visible heel values. Four bounded processes took 27.17 seconds. This run performed zero fits, forward-model invocations or score calls and recorded zero held-lateral target reads. Local serving evidence root `2dd54ce8d2c2c43caf853f6eede63e5426d1d548e8e2c6aead23fb7f7b6d4aed`.

### Overlapping intervals and independent depth-level estimates

| Development readout | RMSE (ft) |
| --- | ---: |
| Registered single grid | **10.433145** |
| Single grid on the ensemble with distal forward-window features | 10.437904 |
| Single grid on the matched distal backward-window ensemble | 10.460289 |
| Anchored graph with aligned short and overlapping long intervals | 10.454827 |
| Same graph with only its added donor intervals shifted | 10.810547 |
| Gamma-ray selection between two registered donor increments | 10.495419 |
| Equal weights on those donor alternatives | 10.486601 |
| Extra Trees estimate of two broad depth levels, blended with the single grid | 11.069397 |
| Extra Trees with the survey quarters rotated per well | 11.246044 |

The distal trial added 27 recorded-survey features over 2,048 ft, 4,096 ft and the whole mutually available span. Forward and backward inputs used identical span and availability at each station. The experiment replayed the original 340 features and re-estimated the relative donor correction against each new model's prediction. The distal forward model improved one fold against the current single grid despite improving 326 wells. Its paired 95% interval for the difference was −0.051 to +0.063 ft. The separate original-row audit reproduced the eight scores after six fixed fits. Local evidence root `64a3ea50e527906f715fe6729ae49cb1f63723e7f67d2fb6f117511d5c40a8a6`.

The graph put half its precision on the existing 1,024-ft intervals and allocated the rest equally between staggered 1,024-ft intervals and 256-ft intervals. In 2,118 supported loops, a registered long interval differed from the sum of four short intervals by 1.009 ft RMS. The same discrepancy with shifted donor observations was 4.856 ft. The aligned graph beat its shifted control and improved one fold against the single grid. Its paired 95% interval for RMSE change against the single grid was −0.007 to +0.049 ft. Local evidence root `5be16007a1c91e64cc74d86e2ed6357f4348384feabccb01d6f361587ea9bb21`.

The gamma-ray test used two actual training-donor alternatives on the same query interval and the query's paired, heel-calibrated reference log. Measured gamma ray favored the closer donor in 505 of 941 sections. The registered increment range excluded the realized interval change in 482 sections. Jointly choosing all interval donor mixtures to minimize each held well's error gave **10.095622 ft RMSE** on this fixed basis and original support. The realized targets supply this retrospective bound. Local bound SHA-256 `8236056c3803ae64f363b85a5e98f0bc804c63b08a8d7d894c4abe7ddd4eedb0`.

The Extra Trees test estimated mean depth in the second and fourth measured-depth quarters independently of the existing forecasts. Its direct errors were 12.440 and 17.101 ft, compared with 8.164 and 12.739 ft from the registered grid. The predeclared blend worsened all three folds. All six fits used only the relevant training-well depths. The original-row audit independently reproduced all station and band sums. Local evidence root `18692e2205f63c8bee480740529695a45735fdfdee2dfd8a519c0650150338a5`.

### Remaining well-wide offset and drift

In the 10.433145-ft readout, the within-well mean error accounts for **63.94%** of squared error and the remaining linear MD drift for **19.35%**. The first toe decile scored 3.399 ft and the last 14.344 ft. Fitting and removing an affine residual from each well using realized targets would leave **4.264585 ft RMSE**. That descriptive projection measures how much error the representation can absorb with known depths.

Resampling entire 20,000-ft heel-coordinate cells gave 95% intervals below zero for the registered grid's improvement over the forward ensemble under all four fixed grid origins. There were 30 to 33 occupied cells depending on origin. The highest upper endpoint was −0.017 ft. This is spatial sensitivity of the existing fixed predictions on the same development folds. Local diagnostic receipt SHA-256 `97c650edb9f64be185824a3b1fa68731d43ae44d2dbac2e99c0f89fb6df71ab0`.

### Combining member correction with observed gamma ray

Each member correction learned signed errors from training wells that its original model had excluded. On the outer query, a decoder applied half its predicted adjustment to each inner member with a short heel ramp. The full model and the original 50/50 blend stayed fixed. Both arms recomputed registered interval corrections after changing the member forecasts. The primary fit separate smooth and local components of the signed error and scored **10.234576 ft**. The equal-budget pointwise control scored **10.235571 ft**. The paired 95% interval between those two decoders was −0.034 to +0.030 ft. All three folds improved against the registered baseline at 10.433145 ft. The gain supports cross-fitted member correction. The incremental benefit of splitting the error remains unresolved. Local evidence root `0fb2cd4a360ada1706477bc46b4b23f6f8c6305e5d5601999f42bdb19fbc5a3b`.

The next prospective test used that calibrated path as the starting geometry. It sampled 25 coherent depth paths and compared the distribution of original gamma-ray readings with the paired reference in physical 512-ft windows. An affine gamma-ray map fitted on earlier visible heel bins predicted later visible heel bins for a reliability score. The correction was the posterior mean depth shift at half that reliability. Its matched control rotated observed gamma ray while retaining the candidate set, measurement support and reliability. The new gamma-ray step required zero fits across wells.

| Readout | RMSE (ft) |
| --- | ---: |
| Registered interval baseline | 10.433145 |
| Affine gamma ray after original registration | 10.259424 |
| Original registration after affine gamma ray | 10.264775 |
| Cross-fitted member correction with registration | 10.234576 |
| Cross-fitted correction and rotated gamma-ray control | 10.265670 |
| Cross-fitted correction and affine gamma ray | **10.030446** |

The combined readout improved all three folds against the member-corrected parent, with a paired whole-well 95% interval of −0.327 to −0.082 ft. Its interval against the rotated control was −0.360 to −0.117 ft. The affine step changed 416 wells. Applying gamma ray before the original registered interval correction missed that separate test's gate. The independent audit reproduced all eight scores on 3,034,427 original stations after the prediction seal. Local evidence root `85028da6228a49b4a23e4af2f571a9d499efc635e4f6ff24631673d4286980d5`.

A separate diagnostic recovered every well's squared error via the public error decomposition. Out of the 25,005,673 ft² removed versus the original registered path, 18,430,941 ft² came from well-wide offsets and 4,647,879 ft² from linear drift. The paired intervals stayed below zero for both stages when resampling whole spatial cells at all four fixed grid origins. These are fixed-prediction sensitivity checks on the same development wells. Local diagnostic receipt SHA-256 `72be4765272e3ffa39ccf1ea36bbac91eba8dbaa7f1395fcf8aaa625362fb9e7`.

The public model-blend API validates the original block IDs separately on each member forecast and on its signed adjustment, then applies those adjustments before the fixed blend. The caller supplies the matching model and correction and authenticates their training roles. [Keyed member correction API](../src/rwt/member_corrections.py)

The raw replay rebuilt the original forward and inner-model forecasts, both new decoder arms and the 2,657 measured donor intervals from authorized raw inputs and frozen training artifacts. Both public member blends matched all 379,568 centers bit-for-bit. A separate worker bound the regenerated calibrated forecasts to the affine calculation by hashes of their exact fold-native values and row keys. It read the original query GR and paired typewells again and reproduced both the affine and rotated forecasts across all 3,034,427 toe rows bit-for-bit. The serving runs recorded zero fits, score calls and held toe target parses. The original field fit had retained the model forecasts, so the comparison reconstructed the original decoder-input matrix independently after sealing the regenerated predictions. Local raw member and combined evidence roots `4de12fb13f9b917eb4e00b16b0bed848595a513bbd569f1e72cba065982e838d` and `0f70302ea22e5b2146ab807527fd110f317f4c3032737b50a533e8a2648381f0`.

### Testing what the gamma-ray correction uses

A fold-trained model lowered gamma-ray amplitude error on the visible-heel validation bins from 13.353 to 12.319. Its toe correction scored 10.264439 ft. Using the affine reference alone for toe depth scoring with the same learned reliability gave 10.248721 ft. Learning the toe response missed its gate against that control. Recomputing reliability with only the affine map gave 10.246150 ft in a follow-up attribution. The ordering experiment used only heel bins common to all candidate maps for its reliability benchmark. The earlier attribution recomputed the early-heel benchmark on the selected map's support. Holding the toe posterior fixed and changing the reliability reproduced the two affine comparisons exactly. Local source and score checks bind the learner to 619 development well identities and preserve the original station target seal.

Three other matched experiments missed their gates. Comparing gamma-ray directions across bin pairs reached 10.465696 ft, against 10.459503 ft for the rotated-sign control. Weighting training peers by their paired reference waveform reached 11.409936 ft, against 11.345306 ft for geometry alone. Registering long donor intervals with disjoint visible heel segments reached 11.099013 ft. Its signed well-mean adjustment correlated −0.019 with the required correction.

### Preserving inbound tuples for joint relations

The typed observation and neighbor APIs preserve original parsed values, channel-level masks, native row IDs and physical measured-depth offsets. The raw audit covered 27,018,426 parsed values and 2,630,300 supported sender positions. [Tuple and relation axes](NODAL_HARNESS.md#typed-observations-and-neighbor-axes)

The first joint relation test kept 95,137 measured-depth bins with three positions and four fields per bin. It compared gamma-ray distributions after grouping bins by signed survey slope. It scored **10.036962 ft** against **10.030446 ft** for the incumbent and **10.032018 ft** with the same survey-group labels rotated within each window. The true versus rotated interval was −0.012 to +0.027 ft. The control moved 73,798 labels and the predictions differed on 2,046,530 original stations. An independent audit recovered the input values, support and all scores from raw data. This grouping missed its gate. Local metrics SHA-256 `62679ebb5ba1f83d230098bf4ad7c088ffa2275584c26a3b2581c30b88088297`.

A higher-layer test kept gamma-ray evidence indexed by measured-depth section, forecast member and one of 25 candidate depth shifts. It applied the same heel reliability and support to a collapsed-mean control and an alternating-section member rotation. The joint model scored **9.993074 ft** on all original stations. The incumbent scored **10.030446 ft**, the collapsed mean **10.028961 ft** and the member rotation **10.001180 ft**. The joint model improved one of three folds against the incumbent, with a paired 95% interval of −0.104 to +0.023 ft. Its prespecified gate failed. The rotation changed predictions in all 416 wells with evidence. An independent audit matched the exact incumbent bytes and all four scores. Local evidence root `7dd1035314bf6b4e78fa9cf89284e5ed5277daebd403f23d09a47baa1d06164f`.

### Julia recurrence in measured-depth and evidence order

The sequence test retained each section's native ID and the distribution losses of the same 25 candidate depth paths. The complex value for each candidate and section encoded its excess loss and signed GR amplitude difference. The recurrence applied `z_next = z*z + c` to that candidate in the chosen sequence and reset at gaps in measured support. The same capped orbit energy affected the candidate weight in each Julia arm. The evidence-order arm went from weak to strong within each supported run, ranking sections by the interquartile range of their candidate losses. Physical section IDs resolved ties. This ordering used only prediction-time data.

| Sequence | RMSE (ft) |
| --- | ---: |
| Additive incumbent | 10.030446 |
| Julia in measured-depth order | 10.030812 |
| Julia reset at each section | 10.030633 |
| Julia with permuted section order | 10.029433 |
| Julia ordered by evidence | **10.027222** |

Evidence order met its separately frozen pooled-error and fold gate. It improved all three folds against physical order, with a paired 95% interval of −0.0075 to −0.0004 ft. It improved two folds against the additive baseline, with a paired interval of −0.0125 to +0.0048 ft. The physical-order primary missed its gate. The evidence order moved 3,501 section positions within the observed runs. The independent audit reproduced all five results on 619 development wells and all 3,034,427 original toe stations. Local metrics SHA-256 `d36fad0674f484ddfe743e5ac9b6556f28385ddb0467191fc02fc50f5ea9d2d9`.

A bounded inference replay bound the source graph and AST, re-read the 1,238 authorized lateral and reference files, and reproduced every ordered section list and native forecast hash. It selected lateral MD, GR and `TVT_input` and paired typewell TVT and GR, with zero held lateral target reads. Local replay evidence root `fee098c4a10036a62d0de7c52a6cd6b282890f48cafdf58a651a97f3a5b7f32e`.

### Training the strength and schedule of evidence

Three further frozen comparisons covered the same 619 wells and 3,034,427 original toe stations.

| Candidate | RMSE (ft) | Matched finding |
| --- | ---: | --- |
| Four-knot path with adaptive evidence traversal | 10.072117 | Its four-knot additive control scored 10.071660. The prior merit path scored 10.027222. |
| Signed visible-heel activity calibration | 10.034288 | Shuffling the calibration signs scored 10.032844. The additive incumbent scored 10.030446. |
| Phase-specific member gains learned inside training folds | 10.025714 | One fold improved against fixed gains at 10.030446. The shuffled-target gain control scored 10.239519. |

All three candidates missed their prespecified gates. The adaptive traversal changed 3,248 section positions relative to measured depth. Expanding the path to four independently varying knots also worsened its additive control. The visible-heel calibration improved a reserved artificial-shift sign assay on 174 wells from 62.64% to 66.09% accuracy. The ordered-only control had the best sign assay and the worst real-toe error at 10.057990 ft. This test exposed a transfer gap between recovering imposed heel errors and predicting hidden geology.

The gain study generated a second level of well-disjoint decoder predictions within each original member's held training bank. Those predictions calibrated early/late coherent and local gains before refitting on the outer training bank. Its paired 95% interval against the fixed affine model was −0.077 to +0.053 ft. The full decoder refits reproduced their original model and median files byte-for-byte. A score adapter initially addressed the wrong stored comparator column and stopped before scoring. A separate no-refit closure authenticated the original sealed forecasts and corrected the column mapping. Local closure evidence root `d856fd1317e017346dd165939e7565c93d3a7049fb4cd7b8c87bc7b9b8bc5a53`.

The signed calibration run retained source, AST, process and original-row audit evidence at root `ff769aaa0c884a26306bc152ea9cd1efa1c6193031b2ad3f907177f864e5c45c`. The four-knot metrics have SHA-256 `f65087f44253884757226ae1ed0afb126a1d755eda496d64b841ca59a53731c4`. The 154 confirmation wells remain sealed.

The initial mathematics and ML cross-reference authenticated six documents and linked eight concepts to source, AST definitions and checks. The subsequent implementations below add exact adjacent component tracking and a restricted nested operator selector. The current machine-checked inventory and remaining proposed extensions are recorded in the [reference graph](METHOD_REFERENCE_GRAPH.md).

### Candidate connectivity and representation limits

The H0 component-history API computes adjacent intersection graphs and maps their connected components into each neighboring revision. Synthetic checks preserve a component while its members change, cut and restore a bridge, and verify that the intersection removes nonshared edges. The field diagnostic used 25 candidate depth states per supported GR window and tracked 3,690 adjacent window pairs. Measured GR produced 807 component splits and 730 merges. Half-rotating the same observations produced 740 splits and 701 merges. These are connectivity measurements of candidate-parameter sets. The diagnostic produced no new forecast and parsed no hidden toe targets. Local evidence root `4dbcc15d86c442c954cafde5bcb93513d6d32c9afd15a2e2b4aef94b7a4abe50`.

A separate retrospective bound held the current GR support and recorded 0.5*q gain fixed. Choosing the best of the 25 two-knot paths using true depths would score **8.387868 ft**. Continuous coefficients in the same box would score **8.362231 ft**. The 625 four-knot paths would score **8.155884 ft**, with **8.122954 ft** for their continuous box. Even perfect predictions on the 416 supported wells would leave **5.871019 ft** from the other wells. This support covers 2,046,530 native rows and 67.09% of the calibrated parent's squared error. Every discrete bound was checked by direct station reconstruction against its quadratic calculation. The calculation selected no serving coefficients or parameters. Local evidence root `54a9f08d08ec1515845d1944f93f32fa754efccb6bb6c1121930946cf129f388`.

The 5.871019-ft figure is conditional on retaining the forecasts of the 203 unsupported wells. Their own station RMSE is 10.289541 ft. The two-knot family uses an exponential heel ramp, while the four-knot family uses linear quarter-length hats. Their continuous bounds compare different trajectory families. Within either family the discrete-to-continuous comparison isolates the grid effect.

### Nested selection with an incumbent action

Each original inner model supplied OFF, affine GR and merit Julia forecasts on its own excluded training bank. Two fixed visible-heel reliability regimes were compared using training-bank native errors. The public helper verifies the bank identities and outer exclusion roles. An active operator needs at least 12 bank wells, a unique lowest pooled error and a gain over OFF exceeding one whole-well standard error of that paired gain. These are study rules for choosing an action. Model exclusion and station-key alignment remain separately authenticated inputs.

The selected action was then applied to that same member on outer queries before the corrected-member blend, donor registration and final affine GR step. OFF reproduced the incumbent exactly. Among 1,857 member queries the rule selected OFF 804 times, affine 450 times and merit 603 times. Missing evidence forced 609 of those OFF choices. Six hidden-column mutations and three missing-GR fixtures verified the decision boundary.

| Native-station readout | RMSE (ft) |
| --- | ---: |
| OFF / incumbent affine path | 10.030446 |
| Always apply member affine | 9.984733 |
| Always apply member merit | 9.985035 |
| Selection using shuffled error tracks | 10.029053 |
| Training-bank selected member operator | **9.980316** |

The selected path improved one of three folds against the historical 10.027222-ft merit forecast. The paired interval for that RMSE change was −0.132 to +0.029 ft. Its gate failed. Transferring bank rules by a vote to the final blended path scored 10.089542 ft and also failed. The private audit reproduced all original station sums. Local evidence root `9ee56f46561af8cd5ea0a2b83f40cf832894995741c4a43da91b71572420b046`.

The public helper subsequently tightened whitespace aliases and exact equality at the one-standard-error boundary. The original source snapshot remains preserved. A separate replay and exact-arithmetic calculation reproduced all 36 recorded real and shuffled bank decisions under the revised helper. It performed no fits, scoring or raw reads. Local amended binding root `023d3e061b914242bc21fec460ffbc85c4452fc2dfc695b7da4375ebfa2ffc45`.

### Ordinal votes on candidate boundaries

The bounded-loss consensus trial scored 10.024088 ft, but its equally weighted midrank control scored 9.997405 ft. A separately frozen experiment repeated that rank operator with its own rotated-activity and recurrence-OFF controls. Each measured section ranked the same candidate paths, retaining the original support, visible-heel reliability and depth amplitude. The merit-ordered Julia recurrence then accumulated those ordinal comparisons.

| Readout | RMSE (ft) |
| --- | ---: |
| Prior additive forecast | 10.030446 |
| Prior merit traversal | 10.027222 |
| Pairwise-majority alternative | 10.001204 |
| Midrank with recurrence OFF | 10.012035 |
| Midrank with rotated observed activity | 10.168240 |
| Midrank and merit-ordered recurrence | **9.997405** |

The rank operator passed the separately frozen pooled and fold gate against all five comparators. It beat the rotated activity and recurrence-OFF controls in all three folds. Their paired 95% RMSE intervals were −0.238 to −0.105 ft and −0.025 to −0.005 ft respectively. The interval against the previous merit path was −0.092 to +0.031 ft. Two independent inference processes reproduced every native forecast before scoring. Five bounded stages took 29.43 seconds, including the original-row independent audit. Local execution receipt SHA-256 `de89362506ac2f149f83277b6cb9eda0fee6b708aca188b015b06249c57e0354`. All 154 confirmation wells remain sealed.

### Exhaustive GPU paths and reference-map uncertainty

The next frozen experiment compared 25 historical paths, 625 four-knot paths and 15,625 six-knot paths. Each grid retained complete coherent depth paths. All three used the intersection of their finite reference support before constructing the same 512-ft sections. The larger grid used the same ordinal votes and merit-ordered recurrence, with a physical slope/bending prior declared before prediction. Missing evidence retained the geometric parent.

| Native-station readout | RMSE (ft) |
| --- | ---: |
| Archived 25 paths, original support | 9.997405 |
| 25 paths, common support | 9.996844 |
| 625 four-knot paths, common support | 10.039549 |
| 15,625 six-knot paths, common support | 10.107876 |
| Six-knot recurrence OFF | 10.110202 |
| Six-knot rotated observed GR | 10.202485 |

The six-knot primary missed its gate. It lost all three folds against both smaller grids. Its paired whole-well 95% RMSE difference against the common-support 25-path control was +0.057 to +0.167 ft. Measured GR still beat rotated activity in all three folds, with an interval of −0.172 to −0.021 ft. The smaller-grid common-support result is a comparator in this experiment. It has no separate prospective admission gate and does not replace the archived development lead.

The RTX 5070 processed 993,390,625 logical bin–path combinations and 64,203,125 section–path slots for the six-knot grid. Shared local interpolation results were reused across complete paths. Summed CUDA-event spans were 16.07 seconds, including mixed host/device sequencing and synchronization. Field inference took 23.51 seconds including CPU parity checks; the complete governed run and audit took 47.45 seconds. Peak PyTorch allocation was 72.52 MiB and peak reserved memory was 90 MiB. These are allocator measurements. They exclude the driver context and do not measure physical SM occupancy or tensor-core usage.

All 416 supported wells were independently checked on the 625-state grid and the first three on the 15,625-state grid. Maximum expected-knot discrepancy was 3.20 × 10⁻¹⁴ ft. A separate direct dense-basis CPU reconstruction on one well also checked all four-knot and six-knot arms independently of the field worker's shared interpolation lookup. The native audit covered all 619 wells and 3,034,427 original stations after the prediction seal. The stricter support removed 20 of 63,597 GR bins while retaining the same 416 wells and 2,046,530 supported stations. Local prediction seal `ee2ad030e0cff5004cc0d2195f85dc180dc3d46a82b9c29acefaea087cac57fa`; independent raw-row audit SHA-256 `be9ad03c8481ea957bd4195751acb5d3751ac449dd9147590ff52c219aeea8ee`.

A separate frozen study combined 15 affine reference registrations with the 25 candidate paths. Reference weights came only from earlier visible-heel bins; the held heel segment supplied the existing reliability. The weighted map mixture scored **10.038597 ft**, versus **10.010549 ft** for the selected early-heel map on the same support, **10.084186 ft** for a uniform mixture and **10.036977 ft** after alternating the section-to-map relation. It improved one fold against the common-support selected map and missed the declared gate. The new support retained 413 wells, 4,072 sections and 63,006 bins. Independently verified original-row errors and the early-versus-held-heel weight falsifier bind the result to its permitted evidence. Local evidence root `9a109e046f690725394c3421b34784f408ba35cf51e468b377b8c137467f48c2`.

These tests establish that large coherent searches are affordable on the available GPU. Their predictive results direct the next refinement toward evidence support and policy decisions. Simply increasing the candidate count did not improve native-depth accuracy. [Measured compute routes and timing scope](COMPUTE_ROUTES.md)
