# Calibrating the complete native-station forecast

**The complete nested calibration study finished, but its primary failed the admission gate.** The fitted affine policy scored **9.858253 ft**, improving all three folds over the retained **9.909565-ft** policy. Reassigning its observed reliability scalar scored **9.848064 ft** and beat the primary in two folds. The primary passed eight of nine comparisons; neither it nor that control was promoted. All 619 wells and 3,034,427 original toe stations remain in every score. All 154 confirmation wells remain sealed. [Retained development policy](ORDINAL_SHAPE_WEIGHTS.md)

The study asked whether the visible heel can tell us how far to move from the incumbent curve **T** toward a paired typewell chart curve **S1**. The weight was learned on whole forecasts after station lifting, geological registration and gamma-ray routing. This lets the loss measure the change a field user would actually receive at each original station.

## Which wells supply each quantity?

The three outer folds retain the original well and station assignments. Within each outer training set, three smaller groups take turns receiving a complete forecast from the other two. A further split supplies depth forecasts for fitting residual corrections.

| Role | Wells per split | Use |
| --- | ---: | --- |
| O, outer held | 206–207 | Receives the frozen policy. True toe TVT opens only after all ten outer forecast arrays are sealed across all three folds. |
| F, outer training | 412–413 | Supplies the three native prediction banks and one fold-specific calibration. |
| B, held within F | 137–138 | Receives one complete native forecast per well. Its labels become calibration outcomes only after all nine banks are sealed. |
| R, remainder of F | 274–276 | Supplies the geological donors, residual training examples and GR profiles for that B bank. GR profile validation has its own three disjoint partitions within R. |
| H, held within R | 91–92 | Receives a depth forecast from D. Its sealed forecast and subsequent observed error supply residual supervision. |
| D, remainder of R | 182–184 | Supplies labels and donor features to one depth model. |

The run completed **27 new depth-model fits**, **18 residual-head fits** and replay of **nine existing full depth models**. Each D model predicts its H group and the corresponding B group. Once the three H forecasts are sealed, two residual heads fit their errors on R. The B ensemble combines the existing R-trained model and the mean of three corrected D-model forecasts with equal weight. Chord donors, GR profiles and route eligibility are reconstructed on that ensemble's registered geometry using R. On positive-pointwise ordinal paths, the incumbent uses the admitted shape weighting. A fallback chart with measured support may act even when the TRAIN-conditioned GR source has no support.

Training labels arrive in separate packages keyed to the authorized role and original stations. Each validation profile is frozen before its held GR targets are issued. The nine B banks must be independently replayed and sealed together before the calibration custodian opens any B objective. Each policy fits only its own F partition's numerical objectives. The global scheduling receipt also exposes descriptive bank metadata; those cross-context values are not coefficient inputs. Across the three outer training sets, the nine banks contain 1,238 well occurrences and 6,068,854 stations. Each development well occurs in two different F sets. The final field test has 619 unique wells. Smaller donor and training populations in the B forecasts make transfer to O an empirical question.

## Calibration and admission

On an eligible well, let `q` be the visible-heel chart skill. The primary fits two bounded endpoint weights per outer fold and interpolates between them as `w(q) = (1−q)t₀ + qt₁`. It minimizes original-station squared error with a fixed penalty toward the identity weight. With fewer than 12 eligible training wells or zero movement between curves, it keeps T. Weights zero and one copy T and S1 byte-for-byte. Interior weights produce `T + w(q)(S1−T)`. S1 already includes the chart's original native half-strength factor.

The nine controls are a fitted constant weight, unchanged T, weight q, equal mixing, half the primary weight, full S1, the primary with q reassigned within the held fold, the historical own-skill chart and the historical fold-median chart. Each historical comparison uses its archived chart component on 119 eligible wells (584,611 stations) and copies the **current** T on the other 500 wells (2,449,816 stations).

Admission requires the primary to lower pooled native RMSE and reduce native squared error in at least two of three folds **against every control**. A tie fails that comparison. The report includes all ten curves and 4,000 paired whole-well bootstrap draws. Earlier development results motivated this hypothesis. These results apply to this reused development cohort.

## Final native-station result

| Readout | Pooled RMSE, ft | Fold 0 | Fold 1 | Fold 2 |
| --- | ---: | ---: | ---: | ---: |
| Fitted affine, primary | **9.858253** | 9.108104 | 9.562260 | 10.824037 |
| Fitted constant | 9.859511 | 9.110002 | 9.579498 | 10.810627 |
| Retained T / OFF | 9.909565 | 9.179210 | 9.618019 | 10.855088 |
| Identity-q mixture | 9.874259 | 9.140967 | 9.560471 | 10.841664 |
| Equal mixture | 9.864893 | 9.117642 | 9.582884 | 10.815915 |
| Half fitted action | 9.873140 | 9.131875 | 9.581756 | 10.827475 |
| Full chart | 9.863441 | 9.111321 | 9.585900 | 10.814598 |
| Reassigned q | **9.848064** | 9.087117 | 9.584222 | 10.794392 |
| Historical own-skill chart, rebased | 9.875474 | 9.140108 | 9.560565 | 10.845625 |
| Historical fold-median chart, rebased | 9.864404 | 9.113654 | 9.583751 | 10.817169 |

Against T, the primary's change was **−0.051312 ft**, with paired 95% interval **−0.111015 to +0.000859 ft**. It improved 69 wells, worsened 50 and preserved 500. Against reassigned q, its change was **+0.010189 ft**, interval **−0.013614 to +0.036118 ft**, with one winning fold. The intervals include zero. A lower point estimate against T and eight passing comparisons do not establish the matching scalar's predictive merit.

All action arms changed the same 119 eligible wells and preserved the other 500 forecasts exactly. The primary's eligible error was 9.321885 ft, versus T's 9.600124 ft and reassigned q's 9.265814 ft. Reassignment preserves recipient curves and uses the frozen within-fold donor order. It changes which observed q feeds that fold's fitted affine map. It supplies no substitute admission authority.

| Outer policy | Eligible F wells | t0 | t1 | Matched constant |
| --- | ---: | ---: | ---: | ---: |
| 0 | 72 | 1.000000 | 0.181760 | 0.647653 |
| 1 | 78 | 0.276142 | 1.000000 | 0.716746 |
| 2 | 88 | 0.474835 | 0.990124 | 0.725129 |

These are fitted action parameters, not probabilities. Their slope changes sign between folds. Independent original-station bounded least squares recovered both endpoint coefficients and the matched constant with maximum absolute error **4.56 × 10⁻¹⁵**. Each check opened only its own F's three B prediction packs, three B target packs and objective pack. Neither another F's numerical statistics nor O outcomes entered those fits.

## Full-run proof and cost

All nine native banks were sealed before B outcome issuance. They contain 1,238 well occurrences and 6,068,854 stations, including 238 eligible occurrences. Independent saved-head reconstruction recovered all 27 corrected member curves with **0 ft drift**, all nine cached full-model forecasts and every native station key. All 27 deep checkpoints and their original reload evidence were verified; their D-dependent feature matrices were not independently regenerated.

All ten O surfaces were independently reconstructed before O target issuance, including 500 exact protected copies. The final source closure covers all 108 native stages. Across both controllers there are **262 bounded worker receipts**, including two synthetic preflights. A separate scorer then reparsed only the authorized TVT numeric field from all 619 original lateral CSVs, checked original row joins, used `math.fsum` for well/arm errors and reproduced every pooled/fold score, paired interval, win count and gate decision. It decoded no confirmation values and changed no predictor.

The remaining eight-bank batch took **2,191.17 s elapsed** with two context pipelines. The prior pilot took 541.97 s. The five sequential suffix phases took 10.72 s for the bank seal, 99.94 s for B custody, 61.45 s for objectives/policies, 43.69 s for O deployment and 95.67 s for grading/closure. These active intervals exclude intervening source review and independent audits. Summed bounded worker wall time was **4,036.97 s**; it is not elapsed wall time or CPU time. The 27 depth workers contributed 2,385.60 s, nine two-head workers 241.31 s and the three analytic policy-fit workers 11.31 s. The last measurement includes interpreter/data overhead, not just the small convex solve.

The user's proposed weights by clustered condition/activity types were recorded before this O grading, with parent-policy fallbacks and matched extra-capacity, q-only and type-reassignment controls. That separate study's implementation and predictive result remain pending. A cluster must earn its weight on excluded complete forecasts; grouping by held error is not a permitted type definition.

## Completed baseline replay

| Outer fold | Wells | Native stations | Eligible fallback chart wells | Worker time |
| --- | ---: | ---: | ---: | ---: |
| 0 | 206 | 1,011,543 | 47 | 3.640 s |
| 1 | 207 | 1,011,334 | 41 | 3.719 s |
| 2 | 206 | 1,011,550 | 31 | 3.609 s |

Every installed T value matched in binary64, with a maximum absolute difference of **0 ft**. The replay reproduced 416 ordinal routes, 13 same-lateral GR routes, 189 TRAIN-conditioned routes and one registered-geometry fallback. An independent check reconstructed S1 on all 119 eligible wells and verified exact T copies on the remaining 500. The final seal worker took 1.547 s. The replay parsed permitted lateral observations and paired typewell references, with zero hidden lateral TVT reads and zero new fits.

## Completed mechanical pilot

One inner bank completed all **25 stages** on **138 held wells and 674,695 original toe stations**. Its three new depth models used 183, 183 and 184 D wells and ran all 600 trees. The 275-well R readout supplied two residual heads. The existing full model replayed its historical held predictions exactly. The resulting complete forecast has exactly those 275 R label ancestors and excludes all 138 B wells and 206 O wells.

An independent process reloaded both heads and reproduced all three corrected member curves with **0 ft drift**. Separate native workers reproduced every curve and original station key byte-for-byte. The independent audit verified the three deep checkpoints and their within-worker reload evidence; it did not independently rebuild their D-target-based feature matrices. The pilot opened no B or O outcome pack and produced no new accuracy score.

The run took **541.97 s elapsed**, comprising **419.18 summed bounded worker-seconds**. Retained artifacts grew by **144.69 MiB**; ending scratch was zero. One-second process-tree samples peaked at **1.344 GiB private memory** and **1.403 GiB working set**. These are sampled peaks of different resources, and shared pages can contribute to multiple working sets. The minimum available host RAM was 17.864 GiB. Scratch and cache paths were local to D for each job.

The pilot supports releasing the remaining eight banks with at most two contexts active and four fit threads per context. The fixed stage scope ends at each bank's independent repeat. The global bank seal, B calibration targets, O deployment and final grading require their own later releases. The source-signed scientific contracts precede the pilot; the external serial driver was added afterward and has separate in-flight and terminal source-continuity evidence.

| Reviewed evidence | SHA-256 |
| --- | --- |
| Split and model recipe | `c45d0aa7b8ef09a0d76594d949b7d7e966334bda15b2fac3754cfa5d2192e3ef` |
| Model and custody contract | `d0fe26e422c9564657d99449ee7342b65ff93647b4ed8c346796739e6c2b9876` |
| Native readout contract | `621a1a7deb90ef66ad83ff2580754ff6639f6263752b30869d3d0744d24aa771` |
| Baseline forecast seal | `56a978c514b521b907340a95c011e762c260da287ed710f674e4f6b79125866d` |
| Independent baseline parity receipt | `aa7ee84e666f5c7faada020a15165c00cd5c7a0f81e112c029221b33ca410bc4` |
| Pilot execution closure | `835dc8b3451611e64880c3a2da38397a7be78c932fb67e46cadf7bd63c014885` |
| Independent pilot audit | `08893de76763d2f479c6f300b5a920aeec0a4cae51ade3c7a3f7476f6c8bb42c` |
| Independent head replay | `56ce93e2bc2a7b619b7687e31e422364a2296a7d2176eb6b4436e8f494a22d24` |
| Remaining-eight execution closure | `9c2c1375fc6c16c590b82af87c31b85654cfd1fda092ea2d16b3f16bc4133563` |
| Independent nine-bank replay | `0c61ccc4ad3987c972ee84ed510d7ec99721a6e3712e47973ce83421ed014561` |
| Final ten-arm forecast seal | `c5c7d21589f7169675a35dd6c7438858d713eb0fcad50443a91b2f10dcbc4bc2` |
| Pre-grade ten-arm reconstruction | `6257a67f12af6450a0fc1e2a8a7daa6481c3c56fe84af46a31db6d42d63b6f9a` |
| Ten-arm aggregate | `07a8c50a17e687684877720a5a94b6ba6ea50dd8f6092ac4454814b168c8d9c4` |
| Native source/trace closure | `d959f5ab06dffa166d9b6ed9947b9d8707d8278d218a7ba10bd0c59555fb9af9` |
| Independent raw-CSV scoring | `fc5c351450ae8b55a923e4b7283f508316744e440c64e99a201e06d489c4551a` |
| Root rejection, incumbent unchanged | `9504d2f77a106e88b7413cdcd5e395fe70319edd513996a35dcf412474a6306c` |

Raw measurements, numerical models and private operator implementations remain local. [First-party design rule](../AGENTS.md)
