# Learner and relationship comparisons

These seven frozen comparisons retained the **9.988991-ft** development route. Each used 619 development wells and all 3,034,427 original toe stations. None passed its complete native-error gate. The 154 confirmation wells remain sealed.

The runs tested stronger learners, a physical target coordinate, donor evidence, nonlinear GR calibration and the choice to execute a residual decoder. Their shared endpoint uses the original corrected-member blend, geographic registration and the verified historical assignment of 416 ordinal, 13 self-GR, 189 TRAIN-conditioned and one OFF route. New posteriors were recomputed on each candidate geometry. The adapter reproduced all original forecasts before these comparisons. Local adapter evidence root `e40cb843600806588c901ada08a4245fbf05673367add663a5ed97bd6d5491bc`.

| Comparison | Candidate RMSE (ft) | Matched control RMSE (ft) | Decision |
| --- | ---: | ---: | --- |
| Same-model donor innovations in the residual learner | 10.024409 | 10.006346, shifted donor positions | Failed all three folds against control and incumbent |
| Repeated upward/downward depth crossings | 9.989327 | 9.989245, equal crossing weights | Failed; only one eligible well |
| Inner-bank selection of three learner recipes | 10.041329 | 9.988991, frozen full model | Failed; one selected fold worsened |
| Structural-height training target | 10.355288 | 10.317546, reversed-coordinate target | Failed; incumbent won all three folds |
| Linear leaf regressions | 10.005083 | 9.994579, normalized constant leaves | Failed; one winning fold |
| Monotone per-well GR response | 10.004563 | 9.988991, original affine response | Failed; one winning fold |
| Forward members with residual decoder omitted | 10.039143 | 9.984405, same members with decoder | Failed all three folds against matched control |

These are recorded-lateral offline comparisons. Their performance populations differ from an official competition test set.

## Donor evidence and decoder execution

The donor-innovation experiment supplied the residual learner with measured TRAIN-donor structural-height changes minus the same member's predicted changes. At 256 and 1,024 ft it retained local and accumulated innovations, whole-well context and support, for 16 added columns. Shifted positions on the same donors gave a matched feature-count control. Each original inner model's own TRAIN donor pool supplied both its excluded bank and actual outer query. Twelve coherent/detail heads were fitted with the original recipe.

The true innovations improved the blend from 10.477118 to 10.431824 ft in all three folds. After the existing outer geographic correction, the candidate scored 10.267749 ft against 10.234576 ft. Its final 10.024409-ft score was worse than both controls. The paired whole-well 95% difference against the incumbent was −0.022909 to +0.090880 ft. These stages use overlapping donor evidence, but the experiment does not establish the cause of the reversal. Runtime was 266.30 seconds. Evidence SHA-256 `388fa4bb626474917af35966734b7d93294d5dd9521bd08e67fe7abb342eb389`.

A separate zero-fit comparison followed up the earlier forward-model rebuild. It removed the residual decoder from the three forward members and compared them with backward members, decoder-on forward members and the incumbent.

| Member branch | Before geography | After geography | Final RMSE (ft) |
| --- | ---: | ---: | ---: |
| Retained 308, decoder on | 10.477118 | 10.234576 | 9.988991 |
| Backward 340, decoder off | 10.638659 | 10.397676 | 10.171617 |
| Forward 340, decoder on | 10.517638 | 10.228286 | 9.984405 |
| Forward 340, decoder off | 10.542127 | 10.293506 | 10.039143 |

Removing the decoder failed. The slightly lower pooled decoder-on comparator improved only one fold against the incumbent and remains unpromoted. Its ordinal-only intermediate reproduces the earlier 9.992353-ft result; its final endpoint adds the two independently validated fallback routes. Every branch's native lift and chord correction was independently reconstructed from original model outputs and observed survey rows. The run took 81 seconds and fitted no models. Evidence SHA-256 `2214745ce52fe8636e4cf0b0ab108e439aafdd64b1a420bfdbba1e1cf6e9e48d`.

## Full-model family and capacity

The three declared recipes used the same 340 input columns. They were the original 31-leaf/600-round LightGBM, a 63-leaf/1,200-round LightGBM, and depth-six/1,200-round CatBoost. Each original inner split rebuilt its supervised donor features from its own TRAIN wells. Excluded-bank native SSE selected the recipe using the existing complete-bank, paired one-standard-error rule. Outer labels were unavailable to selection. Only a selected new recipe was refitted on its full outer TRAIN set.

| Outer fold | Baseline bank RMSE | Capacity bank RMSE | CatBoost bank RMSE | Accepted recipe |
| --- | ---: | ---: | ---: | --- |
| 0 | 11.7161 | 11.9463 | 11.6786 | Baseline; CatBoost gain below one standard error |
| 1 | 11.5762 | 11.5154 | 11.4385 | CatBoost |
| 2 | 10.8417 | 10.9185 | 10.8828 | Baseline |

The selected fold's bank gain did not transfer. Its actual outer direct-model error rose from 10.299682 to 10.600026 ft, and its final error from 9.747948 to 9.907975 ft. The other two folds stayed byte-identical. The selected policy's pooled error was 10.041329 ft, with a paired interval for its increase of −0.008172 to +0.116650 ft. This tests the selection policy, rather than proving a result for one fixed family on all folds.

The 19 fits and 48 bounded stages completed in 717.64 seconds. CatBoost 1.2.10 used the RTX 5070 through its standard API. GPU refitting can vary in summation order; the checkpoints were retained and every saved-model prediction replayed exactly. The library was installed separately on D, leaving the public environment unchanged. [CatBoost GPU interface and determinism scope](https://catboost.ai/docs/en/features/training-on-gpu)

All 284 initial bindings, 125 distinct exported source inputs, 41 repository sources and owned model/output artifacts passed final hashes. Evidence SHA-256 `9562879fc30a46bbaed79e4b32955af6c2331c608fdfe175370bbde0e1f83bea`.

## Target coordinate and continuous leaves

The signed-target test fitted the same 340 predictors and 600-tree/31-leaf recipe to `TVT_residual + ζ ΔZ`, where ζ was +1 or −1. It subtracted the same observed binary64 term at prediction. The +1 response represents structural height. The baseline ζ=0 reused the authenticated original model. The target identity preserves depth error in exact arithmetic; it does not prescribe identical fitted trees. All 379,568 held-center inversions and native lifts were checked independently. LightGBM's internal label conversion differs from the supplied binary64 values by at most 0.0000147 ft in this run.

The +1 coordinate scored 10.355288 ft and worsened all folds against the incumbent. Its paired 95% increase was +0.133385 to +0.587335 ft. Six new fits and all original-row checks passed. Evidence root `2adcca20eb5e0d2c647ae2276a1e763aef8abfff964380c8caf2638c7261f4d3`.

The linear-leaf comparison retained the same tree budget and predictors while expanding the within-leaf basis. Both new arms normalized the imputed inputs using only TRAIN means and standard deviations. The linear arm used ridge coefficient 1; the matched arm retained constant leaves. Linear leaf regressors use numerical features on each branch. This is a different complete learner, since subsequent boosting residuals can also change its split topology. [LightGBM linear-tree interface](https://lightgbm.readthedocs.io/en/stable/Parameters.html#linear_tree)

The fitted models contained 118,065–119,968 nonzero branch-local slopes per fold. Each regressor used only its actual split ancestors. Direct native error was 10.990422 ft for linear leaves, 10.678802 ft for normalized constant leaves and 10.709776 ft for the original full model. Linear leaves lost all three direct comparisons. Final error was 10.005083 ft, against 9.994579 and 9.988991 ft, with one winning fold. Six fits and independent native checks completed in 290.93 seconds. Evidence root `9524f43cd3fc1e86369eba6eb2c2f1a8907dc927163dd330f92c6732ec4dc9b3`.

## GR calibration and available crossings

The repeated-depth method compared upward and downward GR gradients through the same depth bands. It preserved crossing station endpoints and fractions across 25 path hypotheses. Among 1,332 monotone passes, two wells supplied enough opposed bands and one also passed the declared reference calibration. That one well covered 5,927 rows and 0.0372% of incumbent squared error. Agreement weighting scored 4.73958 ft there, versus 4.64938 ft with equal weights and 4.36155 ft for the incumbent. Even perfect predictions on those rows would leave 9.987131 ft overall. The footprint and merit both limit this route. Evidence root `2782cd6196069adcad8bfa76ccfd5dc74dafc26a251adcacf3432afd432a6658`.

The monotone-response method kept the original shift/stretch registration, noise, reliability, states and comparison footprint. An early visible-heel isotonic fit replaced the affine response only when it strictly improved separate late-heel GR error. A shuffled early-pair response and rotated toe activity used the same selected wells and masks.

It selected 203 of the original 416 ordinal wells, covering 1,003,680 stations. Their late-heel GR SSE fell from 527,004 to 423,654. True pairings beat both perturbations in all folds, but final depth error rose to 10.004563 ft. It improved only one fold against the incumbent. The paired difference was −0.002778 to +0.035486 ft. Better activity calibration therefore supplied no admitted depth gain in this comparison. Evidence root `57199fa3206f45b4b88b8184ba741a77b6d8cd90a2f8edc9cfa6b95b036875ab`.

All seven runs retain source/AST identities, scoped read records, worker exits, native prediction seals and separate original-row error checks. They are field-research receipts with their own scope. The public code remains covered by its existing [source and native verification](JOVIN_ENROLLMENT.md). Raw data, full fitted artifacts and private implementations remain local.
