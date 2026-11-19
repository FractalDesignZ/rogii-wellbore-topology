# Giving the paired typewell chart priority

The own-heel chart scored **9.955172 ft RMSE**, compared with **9.988991 ft** for the incumbent. It won all three folds against the incumbent and passed six of nine required comparisons. Giving the same chart path a different well's calibration scored **9.952161 ft**. Using one mean or median per fold scored **9.944966** and **9.944190 ft**. The prespecified rule therefore retains the 9.988991-ft route. The controls remain development comparisons.

This study tests whether a well's own visible heel can set the strength of its typewell-chart correction. The primary curve was already known. It scored **9.955172 ft** as a control in the preceding study, in which a joint chart and training-well map missed its admission criterion. Here the new evidence comes from nine prespecified comparisons, including exchanging calibration values among wells while keeping each well's own geometry and chart path. The development population is reused. [Preceding joint-map study](GR_MAP_BUNDLE.md)

## Source order and coverage

At each well the first available source supplies the curve. The original ordinal chart takes precedence, followed by the observed-heel same-lateral GR method, the sorted-distribution typewell chart, the outer-training-well GR relation and finally the registered geometry. The new position for the sorted-distribution chart is immediately before the training-well fallback.

| Source selected | Wells |
| --- | ---: |
| Original ordinal chart | 416 |
| Observed-heel same-lateral GR | 13 |
| Sorted-distribution typewell chart | 119 |
| Training-well GR fallback | 70 |
| Registered geometry | 1 |

The 119 chart recipients cover 584,611 original toe stations, split across 47, 41 and 31 wells in the three outer folds. Each arm copies the same incumbent curve on the other 500 wells and 2,449,816 stations. All comparisons cover 619 wells and 3,034,427 native stations.

Chart eligibility depends on its own visible-heel calibration and the measured coverage of its own candidate paths. On this development set, each of the 119 eligible recipients also had training-map coverage. The run used previously verified original chart section IDs as a batch identity check. A synthetic case exercised chart selection with training-map support absent. The data contain no corresponding live case among these 119 wells.

## What the visible heel contributes

The chart uses the last 2,048 measured-depth feet of visible heel. The earlier two thirds supplied the previously fitted reference depth shift, stretch, GR offset, slope and robust residual scale. The study reused that fit. On the final third, it compared the sorted distributions of observed GR and calibrated reference-chart GR. The well's amplitude factor was

```text
q = clip(1 − sum((sorted observed late GR − sorted chart late GR)²)
             / sum((observed late GR − mean observed early GR)²), 0, 1).
```

The calculation uses the original common 32-foot bin support across all 15 registration hypotheses. A zero denominator gives zero skill. A chart recipient requires the original zero pointwise skill, positive distribution skill, at least 12 early and six later usable heel bins, and at least two 512-foot toe sections with four commonly measured bins each. Matching the sorted distributions can survive a displacement of a peak within a section. It discards the order of the GR samples in that particular comparison.

Across the 119 wells, the archived calibration values were reproduced bit for bit from **3,864 earlier** and **2,176 later** observed heel bins. The held portion here is still the visible heel. The skill values ranged from 0.002268 to 0.979976, with mean 0.509668 and median 0.554900.

| Outer fold | Chart recipients | Mean q | Median q |
| --- | ---: | ---: | ---: |
| 0 | 47 | 0.532978 | 0.573914 |
| 1 | 41 | 0.499326 | 0.478800 |
| 2 | 31 | 0.488004 | 0.506085 |

## Path and controls

Within each eligible toe section, the chart compares measured GR with the paired typewell's GR under the same 25 signed depth hypotheses. Each hypothesis has two endpoint offsets drawn from −32, −16, 0, 16 and 32 feet. Per-section loss ranks and the chart's original Julia recurrence determine a weighted mean endpoint pair. The original depth basis projects that pair onto every measured-depth station, rising smoothly from the visible heel. The primary multiplies the resulting displacement by **0.5 × the well's own q** once. It uses all 1,176 original chart sections across these wells.

The ten arms use the same footprint. The incumbent retains its original training-well fallback on the 119 recipients. The fixed-quarter arm substitutes q = 0.25. Three source controls respectively remove the Julia energy, half-rotate the measured GR bins, or half-rotate whole 25-response reference-chart rows. Those two rotations cross section boundaries and preserve each recipient's eligible positions and candidate order. The half-strength arm halves the primary's completed native correction.

Three controls keep the recipient's own endpoint pair and change only its q. The donor control sorts chart recipients by well ID within their original outer fold and assigns the value half a group ahead, wrapping at the end. All 119 recipients received a different value. The two constant controls use the mean or median of the permitted visible-heel q values within that fold. Each fold has an odd number of recipients, so one recipient per fold retains its own q under the median control. These are offline cohort comparisons. The primary requires only the individual well's observed heel.

## Native TVT results

Every arm includes all 619 development wells. The active-column RMSE covers the 119 chart recipients. All arms gave 10.079571 ft RMSE on the other 500 wells. On every active row each chart arm changed the incumbent's native value.

| Arm | Pooled RMSE (ft) | Active RMSE (ft) | Fold 0 | Fold 1 | Fold 2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Own-heel chart priority | 9.955172 | 9.416019 | 9.214962 | 9.691264 | 10.884807 |
| Original incumbent | 9.988991 | 9.600124 | 9.253748 | 9.747948 | 10.894236 |
| Fixed quarter | 9.962692 | 9.457212 | 9.217028 | 9.726332 | 10.872417 |
| Recurrence removed | 9.956464 | 9.423108 | 9.216026 | 9.692039 | 10.886761 |
| Lateral GR rotated | 9.988654 | 9.598305 | 9.225235 | 9.682672 | 10.975389 |
| Reference chart rows rotated | 9.993244 | 9.623071 | 9.237324 | 9.698135 | 10.964098 |
| Half native correction | 9.965186 | 9.470839 | 9.226035 | 9.713905 | 10.882737 |
| Calibration from another well | 9.952161 | 9.399485 | 9.239735 | 9.721402 | 10.828532 |
| Fold mean calibration | 9.944966 | 9.359870 | 9.191038 | 9.713435 | 10.857253 |
| Fold median calibration | 9.944190 | 9.355594 | 9.188723 | 9.714138 | 10.856453 |

The rule required lower pooled RMSE and wins in at least two original outer folds against **each** comparator. A negative difference favors own-heel priority. Intervals are the 2.5th and 97.5th percentiles of 4,000 paired whole-well bootstrap samples with replacement, using PCG64 seed 42. They include zero for all nine comparisons.

| Comparator | Own minus comparator (ft) | Paired 95% interval (ft) | Folds won | Wells won / lost / tied | Passed |
| --- | ---: | --- | ---: | ---: | --- |
| Original incumbent | −0.033819 | −0.086127 to +0.013207 | 3/3 | 72 / 47 / 500 | Yes |
| Fixed quarter | −0.007520 | −0.042699 to +0.025393 | 2/3 | 71 / 48 / 500 | Yes |
| Recurrence removed | −0.001292 | −0.005496 to +0.002170 | 3/3 | 58 / 61 / 500 | Yes |
| Lateral GR rotated | −0.033482 | −0.092101 to +0.017165 | 2/3 | 74 / 45 / 500 | Yes |
| Reference chart rows rotated | −0.038072 | −0.098864 to +0.009255 | 3/3 | 72 / 47 / 500 | Yes |
| Half native correction | −0.010014 | −0.035163 to +0.013774 | 2/3 | 69 / 50 / 500 | Yes |
| Calibration from another well | +0.003011 | −0.035193 to +0.048351 | 2/3 | 62 / 57 / 500 | No |
| Fold mean calibration | +0.010206 | −0.015618 to +0.041028 | 1/3 | 61 / 58 / 500 | No |
| Fold median calibration | +0.010982 | −0.014516 to +0.041708 | 1/3 | 61 / 55 / 503 | No |

The primary beat the matched source rotations and the fixed-quarter chart under the declared point-estimate rule. None of the three comparisons that preserved the chart path and reassigned its visible-heel amplitude passed. In particular, the fold-median control tied the primary on the 500 protected wells and the three median recipients. The visible-heel skill measures agreement of GR distributions on the known heel. This study did not establish that it gives the best TVT correction magnitude on the unseen toe.

## Verification and scope

The first synthetic launch selected a same-named fixture from the preceding study. A write-once guard stopped the worker before any field observation was read or prediction produced. The failed attempt and its evidence were preserved. The corrected producer loaded the intended fixture by explicit source path and verified the identities of the controller and reused mathematical modules. Both attempts use the same numerical design.

Fresh processes reproduced all four endpoint sources and the earlier native chart-shape, quarter, recurrence-off and measured-GR-rotation curves on all 119 wells. The native receiver reproduced the earlier full-field incumbent, chart-shape and chart-quarter curves byte for byte. Starting independently from the original measurements reproduced all 476 unscaled endpoint pairs within 6.22 × 10⁻¹⁵ ft, and the ten native curves within 1.82 × 10⁻¹² ft. The complete native surface comprised 30,344,270 arm-by-station values, including 24,498,160 fixed incumbent copies. The study fit zero new models and kept all 154 confirmation wells sealed. It establishes no external leaderboard rank. [First-party provenance rule](../AGENTS.md)


Initial target-free inference took 6.23 seconds summed across the three fold workers, fresh-process replays took 6.29 seconds and final assembly took 1.46 seconds.

The four endpoint pairs for each recipient and their source-to-station mappings stayed sealed through scoring. Scoring and the native audit took 3.69 and 3.80 process-seconds. All five downstream field processes took 15.45 process-seconds in total. The elapsed time from field-source freeze through review and final verification was 374.42 seconds. An independent scorer re-read permitted development TVT after the seal and reproduced all 10 pooled, per-fold and active scores, all nine paired intervals and the failed admission decision.

The final independent closure rehashed 1,534 source, input, trace, release and output files across eight producer and five receiver stages, including the preserved failed fixture attempt. Private implementations, measured data, fitted source artifacts and row-level predictions remain local.

| Evidence | SHA-256 |
| --- | --- |
| Source-priority plan | `91ff044be00543848c84a0cba428db72a47003d68c2fa7ed2364dd541b0a3498` |
| Independent calibration assignments | `91c12da5921eaf6a29a3242c875a0ab3069c396d2e6bf722eb483d5e20149333` |
| Preserved first synthetic failure | `73a90b184dfd1c75a6e0fbd4a253a0cf491a87c6a181b4ab5cc7c7d9565b6d34` |
| Corrected producer contract | `12d9ae57c9b459baaa1df26f0f86e043810f7b996219ed80a32da6a97af8f682` |
| Corrected synthetic verification | `d7fe31866892b1c04cbc3e0f1072bedf82cb61f05951279ab953bc0a41ea58e0` |
| Independent producer source review | `a8336ec66234fd65964b84ee81d67789ed21a796b410a2cd446e3955938e234c` |
| Native field contract | `a088c47e453100eb9cb02a0e9d420222d5bd4bdd560322984e6bddf3f4a1d54e` |
| Receiver synthetic verification | `460e442aa80dbc267c424c685e4bcec8e3b7204499c52544de0ce28bfeae31ba` |
| Sealed chart sources and calibration | `c343f5c3f94adca162ee3c40bc86cc83a6fd886f88d91275b089f3f101cb6da8` |
| Independent endpoint replay | `ef99315490573d1216fc22a3c7b561b2feee66d937a90c0a54d97bf2e1401cd6` |
| Sealed ten-arm native predictions | `28e4ffffc18c459f5e9426e88454c900cd40517fbabfcec4745b66fe7c4e531a` |
| Independent observed-input native replay | `636a0464246253a194aafc953b90e6dc376549c07a79906ab77e163401cb401b` |
| Independent pre-score closure | `55f9dec730d67a2944f294b56528086c89e1d2d4c6fd2c2ba2a118dd233a9945` |
| Native scoring and audit closure | `f22531e2682d03de0c370038af397bc04cc62380b0da54b3b9eb548401e1d986` |
| Independent original-row grade and intervals | `1bf16ab165b387ac5f7400e3ba7e793afa2123042ae5b48f6dddcd1be26d571d` |
| Independent source, failure record and final score closure | `3ec1464c32d8acba3943bbc7ed033d3f9bc599ea48346ea3c241446cbcef9ff0` |
