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
