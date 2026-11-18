# Wellbore master iteration

The objective is stronger GR-to-TVT placement with reproducible evidence. Network Engine provides relational hypotheses and changing support; the wellbore evaluator decides whether they improve placement. A local score is not an official leaderboard rank.

Two linked tracks share cohort identities and scoring rules. The rapid-model track tests strong standard alignment/model candidates quickly. The operator track tests changing maps, registered image relationships and higher-order consistency. Features or priors may cross between tracks only through explicit contracts; ablations must distinguish an alignment gain from an incremental operator gain.

This is an extension of this repository. The existing Network Engine and MaxOp projects remain separate providers. Their locked plans and sealed evaluation sets are unchanged. The current wellbore controller uses explicit source manifests, a dependency graph, bounded local execution, retained outcomes, and metric recomputation. It does not claim execution through a MaxOp runtime that has not registered wellbore workers.

## Stages and decisions

| Stage | Required evidence | Next step |
| --- | --- | --- |
| Plan | Input identities, observation policy, source closure, cohort assignment, fixed criteria and dependency graph | Start a fresh attempt |
| Mutate / measure | Allowed observations only; explicit candidate predictions and support; retained execution result | Recompute metrics |
| Verify / audit | Artifact hashes, common-support metrics, coverage and declared gate checks | Record the decision |
| Admit | Every required engineering gate passes | Admit only the stated claim |

Failures remain in their original attempt directories. New criteria or source changes require a new attempt. A synthetic construct pass never admits field performance. Source ASTs and static call sites are syntax evidence; they are not a dynamic call graph or a sandbox.

## First correctness iteration

The old WP8 harness could return success when pad transfer failed. It also read synthetic oracle truth in the donor predictor and allowed self-transfer when scoring training wells. The corrected donor interface defaults to observed `TVT_input`; full `TVT` requires an explicit training-donor policy. `TVT_true` is evaluator-only.

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
| Reviewed plain Forward–Backward candidate | All 54,833 pilot rows, corrected coordinate policy | 60.817223 ft; rejected |
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

The first pilot runs used the earlier coordinate-null convention. Intake review then found a valid smooth Z crossing through `-9999.25` that had been mistaken for a LAS null. The corrected reader preserves finite coordinates unless their source declares a null code. Fresh attempts retain the old runs unchanged and use stronger station-key/fold audits. The two selectors formerly scored 10.997149 ft with an incomplete path library; repaired path availability changes both to 11.030914 ft. This is an ingestion/library effect, not a native gain.

## Image-to-table relationships

`rwt.registered_views` preserves station identities, missing support and both along/cross coordinates when rotating a trajectory. Explicit noncollinear control correspondences define a world-to-pixel affine map with measured residuals and an inverse. A single vertical-plane raster loses perpendicular position; the inverse currently uses that position from the paired table. Independent views or additional constraints are required for image-only reconstruction.

Color pixels inside a declared ROI are unordered candidates until correspondence is established. Control-point fit residuals are not independent registration accuracy. Supplied training figures also contain full-TVT alignment and formation surfaces, so these panels cannot condition an unseen-toe prediction. Geometry/GR views can be reconstructed from the same allowed observations at serving. Operator-cycle and recurrence hypotheses must be evaluated separately from their visualization.
