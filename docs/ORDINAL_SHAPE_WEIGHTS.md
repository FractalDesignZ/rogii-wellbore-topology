# Weighting the ordinal depth path from visible-heel GR

Changing the weight on an existing depth path lowered development RMSE from **9.988991 to 9.909565 ft** across 619 wells and 3,034,427 original toe stations. The prespecified primary passed all eight required comparisons. It improved all three folds against the pointwise incumbent, with a paired whole-well 95% interval of **−0.118721 to −0.041558 ft** for the RMSE difference. The comparisons that isolate the value of assigning each well its own weight increase were closer. Their three intervals include zero.

## Which relationship changed

The earlier typewell chart already measured how well each lateral well's gamma ray (GR) agreed with its paired reference in the visible heel. On the first two thirds of the last 2,048 measured-depth feet, the existing fit chose a depth shift/stretch and an affine GR response. The final third supplies a check at known depths. Both measures use the same eligible 32-foot bins, retaining finite observed GR and reference values across all 15 registration hypotheses. Across the 416 recipients there were **13,277 earlier and 7,398 later heel bins**.

Let `Y` be measured GR on those later bins, `P` the calibrated paired-reference GR there and `m` the mean measured GR on the earlier bins. Define

```text
D = sum((Y − m)²)
p = clip(1 − sum((Y − P)²) / D, 0, 1)
s = clip(1 − sum((sort(Y) − sort(P))²) / D, 0, 1)
```

A denominator at or below 10⁻¹² gives zero skill. The original pointwise skill `p` checks GR at paired positions. The distribution skill `s` compares the same measured amplitudes after sorting each vector independently. Sorting minimizes squared disagreement across permutations, so `s ≥ p` in exact arithmetic. All 416 observed increments were positive. Replaying the historical binary64 operations reproduced all 416 values of each scalar bit for bit.

On the toe, the original ordinal chart ranks 25 two-endpoint depth hypotheses within 512-foot sections. The original Julia recurrence contributes to the cost used to weight the hypotheses. The resulting mean endpoint pair `U` determines one smooth correction along measured depth. Every arm uses the same registered reference geometry `H`. At distance `l` past the heel, with final distance `L`, the native readout is

```text
Φ(l) = (1 − exp(−l/512)) × (1 − l/L, l/L)
TVT(l) = H(l) + 0.5 × q × Φ(l)·U
```

The primary sets `q = s` directly. Its paired pointwise control uses `q = p` on the identical path. Both use the archived station-level binary64 calculation, including the original basis and matrix-product ordering. There were zero new model fits. Earlier chart experiments used different recipient populations and preserved these 416 wells. [Earlier source-priority study](SOURCE_PRIORITY_SHAPE.md)

## Controls and role correction

The half-increment control uses `q = (p+s)/2`. Three controls preserve each recipient's own pointwise anchor and vary the fraction of its remaining headroom. For `p < 1`, set `h = (s−p)/(1−p)` and transfer a fraction `t` using `q = p + (1−p)t`. At `p = 1` the defined fraction is zero and the resulting weight is one. No enrolled well reached that boundary. Observed fractions ranged from 0.000307 to 0.960890.

One control assigns the fraction from the well halfway ahead in the same held fold's sorted well-ID list, wrapping at the end. All 416 assigned fractions differed from the recipient's. The other two controls use one mean or median across eligible wells in the outer training partition, with one vote per well.

| Held outer fold | Eligible TRAIN wells | TRAIN mean h | TRAIN median h |
| --- | ---: | ---: | ---: |
| 0 | 280 | 0.387066 | 0.350835 |
| 1 | 279 | 0.384878 | 0.370318 |
| 2 | 273 | 0.401302 | 0.384568 |

The initial design selected these training references through each well's historical toe-window eligibility. A different fold's predicted toe geometry can inherit labels from the fold currently held out. Before computing any new weights, the amended rule screened **every** outer training well through a separate visible-heel projection. Eligibility requires a valid calibration, finite `p > 0` and `s` within their allowed range, and at least 12 early and six late valid bins. Each TRAIN worker read only its own projection. The old and amended rules happened to select the same members in all three folds. Replaying their declared dependencies found held-label overlap of 206/207/206 wells in the old selector and zero in the amended one. [Label-lineage guard](LABEL_LINEAGE.md)

Three source controls keep `q = s`. One removes the Julia energy while retaining the original prior and rank evidence. A second rotates measured toe GR by half the chronological accepted-bin count. A third rotates intact rows of 25 reference responses by the same amount, retaining each candidate's column identity. Each rotation preserves the admitted physical positions and section membership of the recipient. The new reference rotation moved all 63,597 response rows across section boundaries and changed the source comparison in all 4,109 sections and all 416 wells.

## Native station results

All scores are RMSE in feet, calculated from sums of squared errors on the original rows. The active folds contain 136, 137 and 143 wells. The 416 active wells cover 2,046,530 stations. Every arm copied the same incumbent values on the remaining 987,897 stations from 203 wells: 13 use the same-lateral GR route, 189 use the training-well GR relation and one uses the registered geometry.

| Arm | All 619 | Fold 0 | Fold 1 | Fold 2 | Active 416 | Other 203 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Own distribution skill (primary) | 9.909565 | 9.179210 | 9.618019 | 10.855088 | 9.733647 | 10.264410 |
| Original pointwise skill | 9.988991 | 9.253748 | 9.747948 | 10.894236 | 9.853287 | 10.264410 |
| Half increment | 9.944898 | 9.211466 | 9.678533 | 10.870909 | 9.786931 | 10.264410 |
| TRAIN mean headroom fraction | 9.915733 | 9.179088 | 9.635786 | 10.856335 | 9.742956 | 10.264410 |
| TRAIN median headroom fraction | 9.919289 | 9.185204 | 9.639669 | 10.857459 | 9.748321 | 10.264410 |
| Another well’s headroom fraction | 9.913562 | 9.180945 | 9.592634 | 10.886980 | 9.739679 | 10.264410 |
| Measured toe GR rotated | 10.146609 | 9.478528 | 9.911122 | 10.990320 | 10.089252 | 10.264410 |
| Reference response rows rotated | 10.133663 | 9.429291 | 9.888582 | 11.017107 | 10.069941 | 10.264410 |
| Julia recurrence removed | 9.926771 | 9.204260 | 9.644698 | 10.857368 | 9.759607 | 10.264410 |

The primary changed all 2,046,530 active station predictions relative to the pointwise incumbent. The admission rule required lower pooled native RMSE and lower native SSE in at least two of the three folds **against every comparator**. The intervals below are descriptive percentiles from 4,000 paired whole-well resamples of all 619 wells, in lexical ID order, using PCG64 seed 42. A negative difference favors the primary.

| Comparator | Primary minus comparator (ft) | Paired 95% interval (ft) | Folds won | Wells won / lost / tied | Gate |
| --- | ---: | --- | ---: | ---: | --- |
| Original pointwise skill | −0.079426 | −0.118721 to −0.041558 | 3/3 | 233 / 183 / 203 | Pass |
| Half increment | −0.035333 | −0.053861 to −0.016795 | 3/3 | 231 / 185 / 203 | Pass |
| TRAIN mean headroom fraction | −0.006168 | −0.025213 to +0.012673 | 2/3 | 217 / 199 / 203 | Pass |
| TRAIN median headroom fraction | −0.009724 | −0.028823 to +0.009136 | 3/3 | 219 / 197 / 203 | Pass |
| Another well’s headroom fraction | −0.003997 | −0.037124 to +0.033053 | 2/3 | 217 / 199 / 203 | Pass |
| Measured toe GR rotated | −0.237044 | −0.325423 to −0.151232 | 3/3 | 247 / 169 / 203 | Pass |
| Reference response rows rotated | −0.224098 | −0.315182 to −0.139147 | 3/3 | 238 / 178 / 203 | Pass |
| Julia recurrence removed | −0.017206 | −0.029822 to −0.005690 | 3/3 | 218 / 198 / 203 | Pass |

The primary also improved on taking only half the weight increase. Its pooled advantage over reassigning another well's fraction was 0.003997 ft and over the TRAIN mean was 0.006168 ft. Each of those comparisons changed sign in one outer fold. The present development evidence supports increasing the chart weight more strongly than it supports precise well-by-well calibration of that increase.

## Verification and timing

The producer sealed four unscaled endpoint pairs for each of the 416 recipients and all nine sets of weights. Fresh processes reproduced the fold outputs. The three previously available endpoint sources (true, rotated measured GR and recurrence removed) matched their archived pairs bit for bit in all 1,248 comparisons. An independent implementation reconstructed the physical masks and four endpoint sources from 832 permitted lateral/typewell reads. Its largest endpoint discrepancy was 7.11 × 10⁻¹⁵ ft. The nine-arm native seal contains **27,309,843** station predictions. It reproduced the complete pointwise incumbent bit for bit. The independent raw-input reconstruction agreed within 1.82 × 10⁻¹² ft per station. Following the native seal and score release, an independent scorer checked all original station keys, all nine pooled and stratified scores, all eight intervals and the admission decision.

The producer's 11 bounded processes took **19.11 summed process-seconds**, including fixtures, separate TRAIN summaries, three fold calculations, three fresh-process replays and assembly. The five native-field processes took **23.70 summed process-seconds**, including 11.61 for native forecast assembly, 4.47 for scoring and 4.44 for audit. The 443.90-second wall interval from field-source freeze to final verification also includes reviews and release barriers.

This is a development admission on a repeatedly studied cohort. All 154 confirmation wells stayed sealed. External leaderboard rank remains unmeasured. The study used first-party measurements, operators and lineage. [Repository provenance rule](../AGENTS.md)

## Installed development policy

The public weight graph materialized each actual observed `s` as a snapshot `FixedWeight`, with its calibrated source and footprint hashes, and applied the original 0.5 factor once. Query-heel observations are not presented as held-training labels. Separate processes rebuilt all 3,034,427 primary station values byte for byte, including all 203 protected curves. An independent installation audit rehashed 609 source, input, runtime and output identities and repeated the read-only native replay. Installation introduced no new fitting, objective evaluation or forecast choice.

The label-lineage graph binds 38 declarations and 36 numerical role-exclusion proofs to authenticated model, donor and profile records. Each presealed numerical fold excludes its own outer labels. The later development selection carries all 619 evaluation-label ancestors and correctly fails those outer exclusions. Its separate proof excludes all 154 confirmation IDs using the split manifest; it reads no confirmation data. This preserves the distinction between generating the forecasts and choosing a development policy.

Root admission `13eefc9046d9855d94e000cac7f3155facaa9649723a5af14800189d1cce182f` installed the verified 9.909565-ft snapshot as the current development incumbent. The bundle is scoped to the recorded cohort, existing model/donor artifacts and original station keys. Arbitrary unseen-well serving and independent confirmation remain unverified. [Weight and role contracts](POLICY_REFINEMENT.md)

| Evidence | SHA-256 |
| --- | --- |
| Original selector plan | `2a33bdb7e9033a8dbb7870500039c8497260b746fd38630ab82a6d0905af8969` |
| Amended plan and independent visible-role approval | `c6af59085301c8e34a099b885db663711254fe64a62545ccc144304cff32fc85` · `5dd5091addf947e132ea01dd9e636d7df6da935f5a9dc5bd58719358fbf6f2f3` |
| Producer and native receiver contracts | `76b5cb666c6ca4c7b3d4fdc7eee32e05f8d23ce279f82a0d376df55f0ecba086` · `8e23fab983b9f72d78f35b3ad00d9f50194ab1e0279346fa9c15ba458e3552f3` |
| Sealed endpoints and weights | `84d46082c258f16f195e0012fa13777082300a926a50da551e1609a98d1fb892` |
| Independently reconstructed physical masks | `ed865a9bf9c2d0751030b0d9d8a032f9b48a1560a8c0a2f61db1bee5f6c5ff9b` |
| Sealed nine-arm native curves | `999a75f8b808343fa55b52073307b56345f07dff9b675d1253b687c9107dd3c4` |
| Independent pre-score native reconstruction and closure | `324d44e9168f4c8553a4d4e33076776be30f53e6262a10ef66e30e35f6b5433e` · `d7f2e67e8502bb186e9c15c0246b3d9d08259df4af0a5bdcab9ed984848cbdde` |
| Field scores and field audit | `64bfe552ed549811b3c816ff1be5d92252f356449394911602ea911d4d6c4092` · `6a21d6b9a4bf2eaf6c7217aaeaa992112f18cd86e9ea9ef7736f2f2249e9cf8c` |
| Field terminal verification (file SHA-256) | `655594ca750fac1a67cc347ca28404281b1ecfa60e9ed5839739fd932183b6c4` |
| Independent original-row score and final closure (logical roots) | `8e7477094ac5970447137e37f03a58e8fce59f68c416dbcfda18d7b644e3b305` · `ccc0054960a5837b90e69410c1707c27ed390db4ded3e1c34e162a99462c2a23` |
| Installed policy bundle and materialized native file | `378821f9018841a8fcf9c5eb25eb4d0d0e4232a31563ee60f997f2ffae0b708e` · `6920a86da4f063e54975682e2c9c25985b0aaec2f3d3b780daf88dac0ec0ef8e` |
| Independent installation audit (logical root) | `d04244ceec2703bceffbcdd280b6337b1b2f43a483a50de787864c4454041882` |
| Root development admission | `13eefc9046d9855d94e000cac7f3155facaa9649723a5af14800189d1cce182f` |

