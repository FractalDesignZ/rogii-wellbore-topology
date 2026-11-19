# Two gamma-ray maps on the same candidate depth paths

The joint chart and training-well map scored **9.977000 ft** native development RMSE against the incumbent's **9.988991 ft**, improving all three folds. The chart-only comparison at the same gain scored **9.972554 ft** and beat the joint path in all three folds. The strongest prespecified standalone, the historical chart-shape curve on the same 119 wells, reached **9.955172 ft**. The primary passed **9 of 15** required comparisons, so this experiment retains the 9.988991-ft incumbent. The stronger standalone result is recorded as a control.

## What the two maps share

A lateral gamma-ray (GR) measurement can constrain a depth path in two ways. The **chart map** compares the distribution of observed GR in a 512-foot measured-depth section with the GR that each candidate path would encounter on the well's paired typewell. The **training-well map** compares changes in the same lateral GR with the changes observed at similar depth separations in other wells. Each outer fold uses only its own previously fitted training-well profile. Candidate inference uses observed lateral GR, survey geometry, the visible heel and the paired typewell. This experiment required zero new model fits.

Both maps evaluate the same 25 signed endpoint pairs. Each coordinate is one of −32, −16, 0, 16 or 32 feet. The displacement grows from zero at the last visible heel station and varies with distance along the original recorded well. A 512-foot exponential ramp determines how quickly the correction takes effect. The source keys retain well identity, original station, measured-depth section and candidate identity throughout the calculation.

| Coverage | Count |
| --- | ---: |
| Development wells and original toe stations | 619 / 3,034,427 |
| Wells and stations with both relationships | 119 / 584,611 |
| Active wells by outer fold | 47 / 41 / 31 |
| Shared 512-foot sections | 1,176 |
| Chart bins and contributing original finite GR measurements | 18,218 / 389,130 |
| Training-map edges on the shared sections | 70,848 |
| Edges from visible heel to toe / between toe bins | 5,500 / 65,348 |
| Wells and stations copied from the incumbent | 500 / 2,449,816 |

Every chart bin and all 389,130 contributing measurements also appear at a training-map edge endpoint. The two maps draw additional information from different sources, the paired reference chart and the outer training wells. The number of edges counts relationships among readings, including repeated use of the same readings. The 119 wells contain between six and 20 shared sections each.

## How the joint path is chosen

In each section, each map ranks the 25 candidate paths from lowest to highest mismatch. Let *S* and *T* be the two loss ranks at a given section and candidate, scaled to [0, 1] with tied ranks averaged. The primary uses the larger of the two costs. The additive control uses their mean. At that same section and candidate, the difference is exactly half the absolute difference between the ranks.

```text
max(S, T) = (S + T)/2 + |S − T|/2
```

The well-level rule sums those costs across matching physical sections. The primary and additive control apply the same symmetric endpoint prior once and the same quarter-weighted Julia energy from the chart map. The chart recurrence follows its original order of increasing loss dispersion within each contiguous section run and resets at physical gaps. The training-well relation supplies its original rank score. This preserves the original asymmetry of the two maps.

The resulting weights give one mean endpoint pair. Its native amplitude is the fixed average of the archived chart-quarter gain (0.125) and the training-map gain (half its fold-specific held-training skill). The three native multipliers were approximately 0.0751–0.0755. Each was applied once. Close agreement on a candidate affects its rank weight without treating shared GR observations as independent probability estimates.

## What the controls distinguish

- The additive control tests the extra penalty for disagreement at the **same section and state**. Moving whole training-map section vectors keeps their candidate-specific summed ranks exactly equal. It changes which chart and training ranks meet in each physical section.
- Chart-only and training-map-only estimates use the same joined sections, gain and prior as the primary. Their arithmetic midpoint tests averaging outputs after each graph has already chosen its endpoint pair. A second midpoint averages the two historical native corrections at their original gains and supports.
- The historical chart-quarter, chart-shape and incumbent curves test the strongest existing individual alternatives on the same 119 wells. The two half-strength controls test the effect of reducing the individual correction magnitude. Previous chart-only figures of 9.960076 and 9.967692 ft were measured against an earlier baseline on 127 chart-eligible wells, including eight protected here.
- Three more controls rotate the measured lateral GR sequence, the intact 25-response reference-chart rows, or training-map query edge activities. A separate control removes chart Julia energy. Each keeps its recipient's original station keys, applicable support and control-specific gain.

Before scoring, separate processes reproduced five historical native curves on each of the 119 wells and reproduced the new unscaled endpoints byte for byte. The complete chart-reference rotation crossed section boundaries at all 18,218 bin rows. Both chart rotations altered losses in all 1,176 shared sections. The training-section rotation changed all 1,176 rank vectors at their assigned positions while preserving the additive result exactly on every well. Each of the nine new graph controls yielded a different endpoint from the primary on every eligible well.

## Admission rule

The primary must achieve lower pooled native TVT RMSE and win at least two of the three original outer folds against each of 15 prespecified controls. Every arm uses the same 3,034,427 native toe station keys. The 4,000-draw paired bootstrap resamples complete wells with replacement and reports 95% intervals for the primary-minus-control difference. The comparison reuses the development cohort. All 154 confirmation wells remained sealed and no leaderboard result was measured. [First-party provenance rule](../AGENTS.md)


## Native TVT results

All errors and intervals are in feet. The difference column is primary minus comparator, so a negative value favors the primary. The final column counts outer folds in which the primary had smaller summed squared error.

| Arm | Pooled RMSE | Fold 0 | Fold 1 | Fold 2 | Difference | Paired 95% interval | Folds won |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| Joint maximum rank | 9.977000 | 9.234851 | 9.738260 | 10.885967 | — | — | — |
| Incumbent (historical TRAIN) | 9.988991 | 9.253748 | 9.747948 | 10.894236 | −0.011992 | −0.021136 to −0.003931 | 3/3 |
| Historical chart, quarter skill | 9.962692 | 9.217028 | 9.726332 | 10.872417 | +0.014308 | +0.000059 to +0.030771 | 0/3 |
| Historical chart, shape skill | 9.955172 | 9.214962 | 9.691264 | 10.884807 | +0.021828 | −0.020621 to +0.071008 | 0/3 |
| Half-strength historical chart-quarter | 9.975354 | 9.234042 | 9.736365 | 10.883825 | +0.001645 | −0.005438 to +0.009580 | 0/3 |
| Half-strength historical TRAIN | 9.989810 | 9.254044 | 9.748331 | 10.895893 | −0.012810 | −0.022556 to −0.004255 | 3/3 |
| Midpoint of historical native corrections | 9.974509 | 9.233637 | 9.736002 | 10.882170 | +0.002490 | −0.003330 to +0.009598 | 0/3 |
| Chart only, common gain | 9.972554 | 9.230332 | 9.734085 | 10.881313 | +0.004445 | −0.003500 to +0.013746 | 0/3 |
| TRAIN only, common gain | 9.985998 | 9.252164 | 9.747699 | 10.887570 | −0.008998 | −0.019628 to +0.000661 | 3/3 |
| Midpoint of common-gain endpoints | 9.978590 | 9.240238 | 9.740342 | 10.883905 | −0.001590 | −0.006273 to +0.003491 | 2/3 |
| Additive mean rank | 9.977190 | 9.233889 | 9.738915 | 10.886721 | −0.000191 | −0.002450 to +0.002015 | 2/3 |
| Joint with recurrence removed | 9.977416 | 9.235461 | 9.738249 | 10.886605 | −0.000417 | −0.001102 to +0.000207 | 2/3 |
| TRAIN section vectors rotated | 9.977392 | 9.234739 | 9.740450 | 10.885183 | −0.000393 | −0.003154 to +0.002279 | 1/3 |
| Measured chart GR rotated | 9.980885 | 9.242078 | 9.731718 | 10.896363 | −0.003886 | −0.014369 to +0.005356 | 2/3 |
| Reference chart rows rotated | 9.983436 | 9.247551 | 9.734823 | 10.895956 | −0.006436 | −0.015050 to +0.001119 | 2/3 |
| TRAIN-map query activities rotated | 9.980478 | 9.241181 | 9.738197 | 10.890218 | −0.003479 | −0.009766 to +0.002376 | 2/3 |

On the 119 eligible wells, the primary scored **9.535179 ft**, the incumbent **9.600124 ft**, and the historical chart-shape control **9.416019 ft**. All arms scored **10.079571 ft** on the 500 preserved wells. The true section pairing slightly improved pooled RMSE over its rotated counterpart and won one fold. The specific physical-section association fell short of the prespecified decision rule.

## Native replay and evidence

Six bounded workers issued and repeated the three fold forecasts. Initial inference took 6.62 worker-seconds across the folds. Fresh-process repeat inference took 6.60 worker-seconds. They produced exactly identical arrays. An independent implementation re-read observed columns from the original well files and reproduced all 1,190 new graph endpoint pairs to within 8.89 × 10⁻¹⁵ ft.

The native receiver then produced 48,550,832 arm-by-station outputs, including 9,353,776 on the common 119 wells and 39,197,056 incumbent copies. A separate decoding of the same sealed endpoints reproduced the native outputs byte for byte. An independent reconstruction starting with the raw observations differed by at most 1.82 × 10⁻¹² ft. All these checks preceded scoring.

Scoring and native audit took 4.36 and 4.20 worker-seconds. The five downstream field workers totalled 16.19 worker-seconds, including synthetic verification, native readout and sealing. Their controller intervals totalled 17.68 seconds, including receipt hashing and trace checks. Wall time from the field-source freeze through review and terminal verification was 304.66 seconds. An independent audit rechecked all 16 arms, all three folds, the active and protected populations and all 15 paired confidence intervals from the original development targets.

The final independent closure rehashed 1,497 unique source, input, station, process and proof files, covering eight bounded producer stages and five field stages. Raw prediction reads used permitted observations only; original lateral TVT opened after the native seal for authorized development scoring. The private implementations, fitted source profiles, measurements and row-level predictions remain local.

| Evidence | SHA-256 |
| --- | --- |
| Prospective plan | `21df0540d42c84461cb3bcdc6babb0da2d1862d07f8f180f43354ccc8a7a4b8c` |
| Producer contract | `86088d26b804b89b3cc0755ebac4795bcfe0e349b26d60a1090ef6a866577859` |
| Producer syntax graph | `d85aba907c4588172bbac738bcaae1341d4eb9bc4f361ffef3a83ab8eb2bbfc2` |
| Producer synthetic fixture | `4044aff8bfc9b35447ab750b872346cffc48035ba14d89215b363fdac4c1f941` |
| Field contract | `1ceed3de230389ad7d5838d9c764fbd0db1b973d8f75ef2effb2491330546ea2` |
| Field source manifest | `8d486aeb607ce4f23faf4286e768b2585647837f156237c2f3bf60a7b1f6d756` |
| Field synthetic fixture | `ac4f8fef19d4348a1a9a68278bf97b9d624d0268770078e5296b31bba3a57201` |
| Independent source and role checks | `00d527af05994c783e44f952367ebb520a9843169302367a78b932b6f59af36e` |
| Sealed graph endpoints | `5b38cf954b1b6cd7166ef43e7868525b55669514d214d43eed4501acc09d4797` |
| Independent original-observation endpoint replay | `5f242e03cbbe6edb653d92903080a12e205ab5de354cb4aeafd9e98672db43fb` |
| Sealed native predictions | `0246aea6cc29052c1bac8a97a57fb7bfe9ff90af281899cd0d4a328d90633580` |
| Independent original-observation native replay | `a57ef22dd20123b4541e3247efc26407713f50b46e3ae6475435f399c7b19bb1` |
| Independent pre-score closure | `cb8d79abf480910dbb269702e26984f624d36ac0721d160b8a7ab6bd02072cf6` |
| Native score and audit closure | `5dff0e094296c35e408122d967980f646f59572c32dacf561a867f2c052a2180` |
| Independent original-row score and paired intervals | `d2e8f37c863491b4005d6e81aeb32ee0d130f4c64d24d2095f3e21da65a7bdb2` |
| Independent terminal source and process closure | `345a028e90f5267a09b38ab9a306207f9225454be8660bb2a0e7b2f611d800ff` |
