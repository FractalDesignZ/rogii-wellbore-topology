# Cluster-conditioned weights on complete native forecasts

The fixed five-feature clustered primary scored **9.865546 ft RMSE**, versus the retained **9.909565 ft**, over **619 development wells and 3,034,427 original toe stations**. It improved all three folds against that incumbent, but passed only **6 of 16** required comparisons. It lost to q-only grouping, both matched untyped refits and the original untyped affine policy. **The primary was rejected; the 9.909565-ft incumbent remains installed.** Controls have no substitute promotion authority. All 154 confirmation wells remain sealed.

The primary-minus-incumbent difference was **−0.044019 ft**, with original-well paired 95% interval **−0.109781 to +0.013901 ft**. This is reused-development evidence, not independent confirmation or an official rank.

A later [separately frozen q-only test](Q_ONLY_NATIVE_WEIGHTS.md) retained this study's q-only recipe as its primary and added same-group constant and reassigned-type controls. It reproduced all 17 overlapping surfaces exactly and passed 16 of 18 comparisons, but failed against constant weights within q types and the historical shuffled-q policy. That follow-up also retained the incumbent and the confirmation seal.

## Fixed question and action

The user's suggestion to weight by clustered types was recorded before the preceding nested study's outer grading. A pre-grading addendum fixed the parent, matched prior controls, routing and ID-control semantics. This comparison reused its authenticated first-party native banks; it did not use the supplied OODA implementation, outside solution recipes, or new geological-model fits. The earlier nested outcome was already known when this implementation ran. [Earlier complete-pipeline comparison](NESTED_FIELD_CALIBRATION.md) · [First-party rule](../AGENTS.md)

Only the existing eligible fallback-chart action may change: **119 wells / 584,611 stations**. All seven new arms copy the other **500 wells / 2,449,816 stations** exactly. The source T and S1 curves, station keys, geological models and Julia/path readout remain fixed.

Each eligible case has five permitted descriptors:

1. Visible-heel chart shape skill q.
2. RMS of the proposed native movement D=S1−T, in feet.
3. Mean(D)/RMS(D), with zero at zero movement.
4. Endpoint contrast (u1−u0)/sqrt(u0²+u1²), with zero for a zero vector.
5. Supported chart-bin count divided by supported-section count.

These are condition/activity/support signatures, not geological class labels. They are correlated descriptions of shared observations. Target residuals, loss statistics, formation tops and ID spelling do not enter type discovery. IDs appear only in the declared random control and deterministic ordering.

Each fold fits medians and IQR scales on its own eligible F cases, clips standardized coordinates to [−8,8], builds a Ward tree and cuts at at most three groups. Original group means are frozen; both F fitting cases and O queries use the same nearest-centroid assignment, with canonical ties. Centroids are not iteratively refitted. q-only grouping repeats this procedure on q alone.

For x=(1−q,q), each supported child directly solves the original-station objective

`sum(C + 2*b*(x·t) + A*(x·t)^2) + 0.01*sum(A)*||t−t_parent||²`, with `t∈[0,1]²`.

Here A=D·D, b=(T−y)·D and C=(T−y)·(T−y), from that F's excluded native forecasts. A group with fewer than 12 distinct eligible F wells or zero movement energy copies the parent. The constant control constrains t0=t1 when supported; its unsupported groups also retain the original affine parent. The child replaces the parent action weight, applies recipient q once, and emits T+w(S1−T). Weights zero and one copy exact endpoint bytes. This uses bounded two-dimensional algebra without model-training epochs. [Weight and role semantics](POLICY_REFINEMENT.md#weights-learned-on-complete-forecasts)

## Population and support

| Outer fold | All F wells | Eligible F | Effective type counts in F | Eligible O | Type counts in O |
| --- | ---: | ---: | --- | ---: | --- |
| 0 | 413 | 72 | 33 / 22 / 17 | 47 | 21 / 17 / 9 |
| 1 | 412 | 78 | 22 / 50 / 6 | 41 | 12 / 28 / 1 |
| 2 | 413 | 88 | 30 / 49 / 9 | 31 | 12 / 15 / 4 |

The six- and nine-member F types use the required parent fallback, affecting five O queries. Every child's ancestry retains the complete shared F calibration parent once. A small cluster cannot erase that parent exposure. Matching O contexts pass exclusion; both sibling contexts are rejected. These declarations supplement separately observed numerical file leases.

## All matched results

Lower error is better. The frozen gate requires lower pooled RMSE and at least two winning folds against **every** comparator. The last two columns compare the five-feature primary with that row; intervals do not form an additional admission rule.

| Arm | Pooled RMSE, ft | Fold 0 / 1 / 2 RMSE, ft | Primary winning folds | Gate |
| --- | ---: | --- | ---: | --- |
| `typed_affine` | 9.865546 | 9.111902 / 9.580833 / 10.824352 | — | primary |
| `incumbent_off` | 9.909565 | 9.179210 / 9.618019 / 10.855088 | 3 | pass |
| `typed_constant` | 9.865016 | 9.112734 / 9.571027 / 10.830872 | 2 | fail |
| `q_only_types` | 9.853415 | 9.101444 / 9.559540 / 10.818824 | 0 | fail |
| `random_types` | 9.871395 | 9.129799 / 9.582735 / 10.823586 | 2 | pass |
| `reassigned_types` | 9.878319 | 9.149679 / 9.582660 / 10.825812 | 3 | pass |
| `global_parent_affine` | 9.858111 | 9.108704 / 9.562384 / 10.823033 | 0 | fail |
| `global_parent_constant` | 9.859482 | 9.109940 / 9.579501 / 10.810599 | 0 | fail |
| `native_affine` | 9.858253 | 9.108104 / 9.562260 / 10.824037 | 0 | fail |
| `native_constant` | 9.859511 | 9.110002 / 9.579498 / 10.810627 | 0 | fail |
| `q_identity_mixture` | 9.874259 | 9.140967 / 9.560471 / 10.841664 | 2 | pass |
| `equal_mixture` | 9.864893 | 9.117642 / 9.582884 / 10.815915 | 2 | fail |
| `half_adaptive` | 9.873140 | 9.131875 / 9.581756 / 10.827475 | 3 | pass |
| `full_chart` | 9.863441 | 9.111321 / 9.585900 / 10.814598 | 1 | fail |
| `permuted_adaptive_q` | 9.848064 | 9.087117 / 9.584222 / 10.794392 | 1 | fail |
| `historical_chart_shape` | 9.875474 | 9.140108 / 9.560565 / 10.845625 | 2 | pass |
| `historical_fold_median` | 9.864404 | 9.113654 / 9.583751 / 10.817169 | 2 | fail |

The five-feature primary beat random types in two folds and reassigned types in all three. Those checks alone do not establish incremental type merit: q-only grouping scored **9.853415 ft** and beat it in every fold; matched untyped affine scored **9.858111 ft** and also won every fold. The primary-minus-q-only interval was **−0.011757 to +0.036030 ft**. The five-feature construction therefore did not earn its extra representation in this comparison. The stronger controls remain controls.

## Execution and independent proof

The new controller froze 12 source files, 175 numerical runtime binaries, the permitted feature schema, all controls and a dependency graph before execution. Its syntax projection records 4,470 AST nodes, 4,463 AST edges and four explicit local file-import edges. These describe syntax and declared source relationships; actual process and numerical-open evidence has its own receipts.

Three feature workers could read only their permitted sealed curve banks. Three fit workers could read only their own F feature/objective packs. Inference received O descriptors, sealed curves and F policies, without O targets. All three complete 17-arm forecast packs were sealed before the independent graders' O CSV leases opened. Python audit hooks reject wrong-role numerical paths; they are process-level checks, not an OS security sandbox.

Three independent coefficient audits reconstructed A, b and C from original F station targets with **zero discrepancy**, and checked 42 family/group policies: 38 supported records used augmented original-station BVLS rather than the producer's quadratic edge solver; four records verified required parent fallbacks. The audits rebuilt discovery, routing, fallbacks and ID controls. Three graders joined original O station keys, rebuilt every new native mixture, verified protected T copies and preserved all ten historical comparator surfaces bit for bit. Raw per-well SSE drove the pooled/fold scores and the fixed 4,000-draw paired whole-well bootstrap with PCG64(42).

An initial attempt stopped before any fit auditor opened numerical data: it compared lexical parent labels with the same role IDs in their original metadata order. Sorting validated role identities corrected that audit integration bug. No feature, coefficient objective, threshold, model or comparator changed. The failed run and its exact source identities remain retained; the completed attempt was freshly frozen.

A further independent checker authenticated source snapshots, runtime identities, all 15 successful bounded processes, trace chains, logs, seals and score-table bindings. It recomputed pooled/fold errors and all 16 gates using `math.fsum`, and independently regenerated the 4,000 paired draws through resampling-frequency sums. Metric differences were at most 5.33×10⁻¹⁵ ft; interval differences were at most 6.88×10⁻¹⁵ ft. It confirmed rejection and the unchanged incumbent pointer.

The 15 completed workers used **45.733 summed wall-seconds**. Their first observed launch to last observed finish spanned **18.790168 seconds**, with at most three workers and one numerical thread per worker. This measures the additional weighting study stages, including their independent coefficient/grade workers. It excludes historical bank/model construction, preflight authentication, synthetic checks, the failed initial attempt, post-run audit and review time. No GPU or end-to-end model-training speed claim follows from this interval.

Raw measurements, private operators, per-well predictions and fitted policy payloads remain local. Local evidence is under `D:/rwt_private_artifacts/clustered_native_weight_001/attempt002/`.

| Evidence | SHA-256 |
| --- | --- |
| Prospective plan | `43149fc4545e96369cf8054f784c11b4eb50d902c4d03d2421d5dce7e967fb88` |
| Pre-grading addendum | `0cf277d540aabc47774b33a7cb80aee1fe26b6e8ef823834a53969517e94357f` |
| Completed-attempt contract | `2e58b35c99dbf3a9a35c5a962bb47ebfb8f3c2ee1fd9cd0a702ce274a7728bed` |
| Three-policy seal | `87043be669b55ea0616df33e0cd7bec0bdd2f12087bbd4e6f9f494b76519f0fc` |
| Global forecast seal | `cfcb11f55f2f023f1731a7d38eec007d7eb183eb7ba52647056838864dc9b54b` |
| Declared lineage | `4c996d4ec8bc155465c4ae7b752e7d92f3ec82ec3551ee20549d0b38e81f2b8b` |
| Aggregate metrics and fixed gate | `56558303b1951e929fd921c49f26402dba9dd885ec2c776d66c4820531458865` |
| Completion record | `b2744f6eea7c4c09ee8dac73135525a477c413e40dfa4a39e1196c0a2d948f71` |
| Independent aggregate/trace audit | `c322a8ab314ab07b805af7b8a2aefa86f24de391d8c378edff48573adab7f01a` |
| Root rejection / retained incumbent | `1f70850de2e156028e9f07fe13249e19b7c647ebb48cb44ae66f22c2fb31194d` |
