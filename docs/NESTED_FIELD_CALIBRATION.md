# Calibrating the complete native-station forecast

**The current development policy replayed exactly on all 619 wells and 3,034,427 toe stations.** Its previously measured RMSE is **9.909565 ft**. The single-bank mechanical pilot also passed. Complete nested calibration and the outer-field comparison remain pending. All 154 confirmation wells remain sealed. [Current development policy](ORDINAL_SHAPE_WEIGHTS.md)

The study asks whether the visible heel can tell us how far to move from the incumbent curve **T** toward a paired typewell chart curve **S1**. The weight is learned on whole forecasts after station lifting, geological registration and gamma-ray routing. This lets the loss measure the change a field user would actually receive at each original station.

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

The design calls for **27 new depth-model fits**, **18 residual-head fits** and replay of **nine existing full depth models**. Each D model predicts its H group and the corresponding B group. Once the three H forecasts are sealed, two residual heads fit their errors on R. The B ensemble combines the existing R-trained model and the mean of three corrected D-model forecasts with equal weight. Chord donors, GR profiles and route eligibility are reconstructed on that ensemble's registered geometry using R. On positive-pointwise ordinal paths, the incumbent uses the admitted shape weighting. A fallback chart with measured support may act even when the TRAIN-conditioned GR source has no support.

Training labels arrive in separate packages keyed to the authorized role and original stations. Each validation profile is frozen before its held GR targets are issued. The nine B banks must be independently replayed and sealed together before the calibration custodian opens any B objective. Each policy fits only its own F partition's numerical objectives. The global scheduling receipt also exposes descriptive bank metadata; those cross-context values are not coefficient inputs. Across the three outer training sets, the nine banks contain 1,238 well occurrences and 6,068,854 stations. Each development well occurs in two different F sets. The final field test has 619 unique wells. Smaller donor and training populations in the B forecasts make transfer to O an empirical question.

## Calibration and admission

On an eligible well, let `q` be the visible-heel chart skill. The primary fits two bounded endpoint weights per outer fold and interpolates between them as `w(q) = (1−q)t₀ + qt₁`. It minimizes original-station squared error with a fixed penalty toward the identity weight. With fewer than 12 eligible training wells or zero movement between curves, it keeps T. Weights zero and one copy T and S1 byte-for-byte. Interior weights produce `T + w(q)(S1−T)`. S1 already includes the chart's original native half-strength factor.

The nine controls are a fitted constant weight, unchanged T, weight q, equal mixing, half the primary weight, full S1, the primary with q reassigned within the held fold, the historical own-skill chart and the historical fold-median chart. Each historical comparison uses its archived chart component on 119 eligible wells (584,611 stations) and copies the **current** T on the other 500 wells (2,449,816 stations).

Admission requires the primary to lower pooled native RMSE and reduce native squared error in at least two of three folds **against every control**. A tie fails that comparison. The planned report includes all ten curves and 4,000 paired whole-well bootstrap draws. Earlier development results motivated this hypothesis. Any field result applies to this development cohort.

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

Raw measurements, numerical models and private operator implementations remain local. [First-party design rule](../AGENTS.md)
