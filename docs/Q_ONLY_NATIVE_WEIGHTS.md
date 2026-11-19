# Q-only grouping: a separately frozen merit test

Q-only grouping scored **9.853415 ft RMSE**, versus the retained **9.909565 ft**, over **619 development wells and 3,034,427 original toe stations**. It beat the incumbent in all three folds and passed **16 of 18** required comparisons. It lost to the new same-group constant-weight control (**9.850860 ft**) and the stronger historical reassigned-q control (**9.848064 ft**), each in two folds. **The full gate failed; the incumbent stays installed.** Neither control was promoted. All **154 confirmation wells remain sealed**.

The primary-minus-incumbent difference was **−0.056150 ft**, with nominal paired whole-well 95% interval **−0.115128 to −0.004870 ft**. The q-only primary was selected for this follow-up after its earlier development result was known. Reproducing it does not create fresh validation, and this interval does not account for that selection history.

## What was frozen

The [exact frozen plan](Q_ONLY_MERIT_PLAN.md) fixed one unchanged q-only primary, 18 controls, original well roles, the admission rule and uncertainty procedure before this run's numerical stages. The primary and all 16 overlapping controls reproduce the previous [clustered study](CLUSTERED_NATIVE_WEIGHTS.md) byte for byte. The new evidence is the q-specific constant and reassignment comparison under that fixed test. The strongest known control stayed in the gate; there was no new parameter, seed or group-count search.

The permitted observation q is visible-heel chart shape skill. Eligible F wells alone supply its median/IQR normalization and Ward discovery at at most three groups. Zero IQR uses scale one; standardized values are clipped to [−8,8]. Original group means remain fixed. Both F and O use nearest-mean routing, with canonical ties and no centroid refitting. Other descriptors are available only to the separate five-feature controls.

Within each q type, the same bounded two-parameter affine weight minimizes original-station squared error plus the fixed 0.01 movement-energy-scaled penalty toward its F parent. At least 12 distinct eligible F wells and positive movement energy are required; otherwise the original affine parent remains. The child replaces the parent action weight, applies recipient q once and emits T+w(S1−T). Zero and one copy exact endpoint bytes. [Objective and parent-role semantics](CLUSTERED_NATIVE_WEIGHTS.md#fixed-question-and-action)

Two new controls isolate the q-specific choices:

- **`q_only_constant`** uses exactly the same groups and penalty, but constrains supported child coefficients to one weight per group. Its fallback remains the full affine parent.
- **`q_reassigned_types`** reuses the primary's fitted coefficients and cyclically shifts only the O group assignments by positive floor(N/2) in eligible lexical well order. Each recipient keeps its own q and T/S1 curves. This preserves the group counts and fitting budget.

Only **119 wells / 584,611 stations** are eligible for the nine newly generated arms. They copy the other **500 wells / 2,449,816 stations** from T exactly. All nine q groups satisfy fitting support; no q-only query falls back to its parent.

| Outer fold | Eligible F | F q-type counts | Eligible O | O q-type counts |
| --- | ---: | --- | ---: | --- |
| 0 | 72 | 23 / 17 / 32 | 47 | 11 / 13 / 23 |
| 1 | 78 | 20 / 33 / 25 | 41 | 9 / 18 / 14 |
| 2 | 88 | 36 / 19 / 33 | 31 | 14 / 7 / 10 |

## Every arm and gate

Admission requires strictly lower pooled original-station RMSE **and at least two winning folds against every control**. Ties fail. Controls cannot be substituted for a failed primary.

| Arm | Pooled RMSE, ft | Fold 0 / 1 / 2 RMSE, ft | Q-only winning folds | Gate |
| --- | ---: | --- | ---: | --- |
| `q_only_types` | 9.853415 | 9.101444 / 9.559540 / 10.818824 | — | primary |
| `q_only_constant` | 9.850860 | 9.096373 / 9.563551 / 10.812565 | 1 | fail |
| `q_reassigned_types` | 9.869908 | 9.121690 / 9.578445 / 10.830148 | 3 | pass |
| `typed_affine` | 9.865546 | 9.111902 / 9.580833 / 10.824352 | 3 | pass |
| `typed_constant` | 9.865016 | 9.112734 / 9.571027 / 10.830872 | 3 | pass |
| `random_types` | 9.871395 | 9.129799 / 9.582735 / 10.823586 | 3 | pass |
| `reassigned_types` | 9.878319 | 9.149679 / 9.582660 / 10.825812 | 3 | pass |
| `global_parent_affine` | 9.858111 | 9.108704 / 9.562384 / 10.823033 | 3 | pass |
| `global_parent_constant` | 9.859482 | 9.109940 / 9.579501 / 10.810599 | 2 | pass |
| `native_affine` | 9.858253 | 9.108104 / 9.562260 / 10.824037 | 3 | pass |
| `native_constant` | 9.859511 | 9.110002 / 9.579498 / 10.810627 | 2 | pass |
| `incumbent_off` | 9.909565 | 9.179210 / 9.618019 / 10.855088 | 3 | pass |
| `q_identity_mixture` | 9.874259 | 9.140967 / 9.560471 / 10.841664 | 3 | pass |
| `equal_mixture` | 9.864893 | 9.117642 / 9.582884 / 10.815915 | 2 | pass |
| `half_adaptive` | 9.873140 | 9.131875 / 9.581756 / 10.827475 | 3 | pass |
| `full_chart` | 9.863441 | 9.111321 / 9.585900 / 10.814598 | 2 | pass |
| `permuted_adaptive_q` | 9.848064 | 9.087117 / 9.584222 / 10.794392 | 1 | fail |
| `historical_chart_shape` | 9.875474 | 9.140108 / 9.560565 / 10.845625 | 3 | pass |
| `historical_fold_median` | 9.864404 | 9.113654 / 9.583751 / 10.817169 | 2 | pass |

Q-only routing beat its reassigned-type control and matched untyped affine in all three folds. That is encouraging development evidence for the grouping. However, constant weights inside those same groups beat the affine primary, so this test does not justify its extra within-type slope. The earlier shuffled-q policy also remained stronger. These are separate failures of the frozen merit rule, not integration failures.

## Paired uncertainty

All intervals use the same 4,000 original-well resamples, PCG64(42), and canonical lexical well order. They are descriptive reused-development intervals, not multiplicity-adjusted evidence. Only the incumbent comparison excludes zero; the intervals against both defeating controls include zero. Intervals are reported alongside, rather than substituted for, the frozen admission rule.

| Comparator | Q-only minus comparator, ft | Paired 95% interval, ft |
| --- | ---: | --- |
| `q_only_constant` | +0.002555 | -0.005815 to +0.012104 |
| `q_reassigned_types` | -0.016493 | -0.053431 to +0.013771 |
| `typed_affine` | -0.012131 | -0.036030 to +0.011757 |
| `typed_constant` | -0.011600 | -0.036081 to +0.012661 |
| `random_types` | -0.017980 | -0.057143 to +0.022109 |
| `reassigned_types` | -0.024904 | -0.057518 to +0.003266 |
| `global_parent_affine` | -0.004696 | -0.014342 to +0.005018 |
| `global_parent_constant` | -0.006067 | -0.026544 to +0.013238 |
| `native_affine` | -0.004838 | -0.014574 to +0.005053 |
| `native_constant` | -0.006095 | -0.026446 to +0.013185 |
| `incumbent_off` | -0.056150 | -0.115128 to -0.004870 |
| `q_identity_mixture` | -0.020844 | -0.059974 to +0.012459 |
| `equal_mixture` | -0.011478 | -0.033497 to +0.008537 |
| `half_adaptive` | -0.019725 | -0.048791 to +0.006326 |
| `full_chart` | -0.010026 | -0.048667 to +0.027453 |
| `permuted_adaptive_q` | +0.005351 | -0.020048 to +0.031552 |
| `historical_chart_shape` | -0.022059 | -0.061891 to +0.011685 |
| `historical_fold_median` | -0.010988 | -0.033362 to +0.009278 |

## Execution and verification

The contract froze **14 source files and 175 runtime binaries**. The syntax projection contains **5,471 AST nodes, 5,462 AST edges and six explicit local imports**. These source records accompany, rather than replace, the observed process and numerical-open traces. Fifteen bounded workers completed: three each for features, F fitting, separate coefficient checks, O inference and original-row grading.

Synthetic checks exercised the actual fitting and inference workers, all 19 arms, q-only isolation from other axes, supported constant weights, affine parent fallback, the reassignment direction, recipient q, protected copies and rejection of a missing global policy seal. Separate algebra checks compared the direct solver with augmented BVLS and tested boundary, support and routing cases without field reads.

The coefficient verifier independently rebuilt F descriptors, group assignments and A/b/C from original F station targets, with **zero statistic discrepancy**. It checked 51 family/group policy records: 47 supported records against augmented original-station BVLS and four required fallback records. Its implementation imports neither the producer nor its solver. Every complete 19-arm O forecast pack was sealed before any grader received an O target lease. The graders rebuilt all new native mixtures and original-station scores, checked role exclusions and verified protected T copies.

A separately implemented verifier, also frozen before execution, checked all 15 process lifecycles, source snapshots, runtime identities, seals, output bindings, numerical leases and full-F parent ancestry. It authenticated 920 distinct byte objects and compared all **17 overlapping forecast surfaces byte for byte** against the previous sealed study. It independently recomputed the metrics, all 18 decisions and paired intervals. Maximum differences were **5.33×10⁻¹⁵ ft** for metrics and **5.20×10⁻¹⁵ ft** for intervals. The incumbent pointer remained unchanged.

The 15 workers spanned **19.596919 seconds** from first observed launch to last finish, with **47.283 summed worker wall-seconds**, at most three CPU workers and one numerical thread each. This covers the additional weighting stages and coefficient/grade checks; it excludes historical model and bank construction, preflight authentication, synthetic checks and final aggregate verification. No new geological model or GPU computation was needed.

Raw measurements, private implementations, fitted policies and per-well predictions remain local under `D:/rwt_private_artifacts/q_only_native_weight_001/`. This result authorizes no confirmation access or control promotion. [First-party design rule](../AGENTS.md) · [Iteration history](MASTER_ITERATION.md)

| Evidence | SHA-256 |
| --- | --- |
| Frozen plan | `3d272971b338e580baa9223bbfcf563fd79c18eb4788eb13e17e5a0891197190` |
| Execution contract | `0b308baee3a737ec543e1a0ec652a7c55855e1dcfb0a58ffdd4fb2ec48d604a1` |
| Three-policy seal | `02ddf71844ad72ce93a4a6ca4b8d0b5169ff1d1a3b660022e9c07fc12c74bbd5` |
| Global forecast seal | `7b3a83e0ced944ed71defc844b5d80a0f1f545048a4da437b6fecf519f23cf54` |
| Declared lineage | `7ef2ce447bbc38a74a15c80d02238bbc9fa627c70a072fa8d53d788978b3ea00` |
| Aggregate metrics and gate | `b0e312774743a1e845414da760f66d1e0cb53f882262af8c5bba45d90e630895` |
| Completion record | `e0a70ea925d5bb13b28ff44efb6717e1c95705b2dfe3272c41516548bdf86b87` |
| Separate aggregate/trace/surface verifier | `3419f44fad55a4cf58a9361ca17312d79b31c3945197a7a16acbccc3312943f7` |
| Root rejection / retained incumbent | `8666891e20265a902eabdfd6fbdfb23c12a6e7cbaf7aa14084211f41e113ee9c` |
