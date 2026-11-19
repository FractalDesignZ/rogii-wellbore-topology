# Kernel endpoint features for the coherent depth decoder

Adding four features derived from a well's predicted depth error raised final development RMSE from **9.988991 to 10.016188 ft**. Assigning the same endpoint estimates to different wells scored **10.001499 ft**. The well-matched features lost all three outer folds to the incumbent and two to the permutation control. The prespecified comparison therefore retains the 9.988991-ft route.

## What the decoder received

The original first-party decoder's 334 inputs at each block center comprise 274 observation and geometry features, 32 features from the recorded forward survey, and 28 describing the particular member forecast and trajectory. The experiment appended two signed estimates from a Gaussian radial basis function (RBF) kernel, `u` and `v`, and two functions of them. The estimates come from previously fitted models of an individual geological member's error. Their inputs included observed survey geometry, GR, candidate/reference-GR relationships, support and member context. [Original kernel studies](NATIVE_KERNEL_STUDIES.md)

Let `s` be measured depth at the original block-center station minus measured depth at the last visible heel station. Let `T` be measured depth at the final recorded station minus that same heel depth. Both are in feet. For a pair `(u,v)` in feet, the displacement at distance `s` is

```text
d(s) = (1 − exp(−s/512)) × ((1 − s/T) × u + (s/T) × v).
```

The four appended columns, in order, are `u`, `v`, `d(s)` and `d(s) − d(max(0, s−256))`. The last two express the same pair at the recipient's surveyed depth and over its preceding 256 MD-ft. The calculation used the original binary64 measured depths and rounded the four features to binary32 for the trees. It assumes the completed recorded survey is available.

To build decoder **training** features, each of nine geological members supplied its out-of-fold predictions on wells excluded from that member's fit. For each such well, study A supplied the raw RBF endpoint pair from the rotation in which that well was in TEST. The RBF model and scaler had seen only their 81–84 FIT wells. The 1,238 training well/member cases use each development well in two outer training contexts. During **held-well prediction**, study B supplied the matching member's endpoint estimate from a kernel trained on the 412 or 413 wells of that outer training fold. Both sources supplied the raw pair before any calibration decision. The training and serving kernel estimates consequently have different training-set sizes and may have different sampling variability.

The primary used the recipient's own pair. The matched control moved whole pairs between different wells while keeping all other inputs fixed and evaluating the two displacement columns on the recipient's measured depths. Within each original study A TEST group, a deterministic hash ordered the wells and a half-group rotation reassigned the pairs. Within each study B held fold and member, hash-ranked wells were partitioned into balanced groups of at most 28, then rotated the same way. Each group retained exactly the same collection of pairs. This offline control uses other held wells' target-free estimates. The primary needs only the recipient's own pair.

## Fitting and station reconstruction

Each arm fit two LightGBM heads per outer fold, for 12 new heads in total. The supervised target was the signed error of the corresponding excluded-well geological member at the block center, `true TVT − member TVT`. Its coherent component was the original station-count-weighted, heel-anchored piecewise-linear projection with equally spaced segments of at most 768 MD-ft. The detail target was the remainder. The median replacements came only from the appropriate training fold. For both arms the first 334 median values matched the incumbent byte for byte. All three folds reproduced the original signed, coherent and detail target hashes. The original settings gave 250 coherent and 150 detail trees, 15 leaves maximum, a 200-example minimum per leaf and learning rate 0.05.

The predicted center correction was the sum of the projected coherent readout and the detail readout, multiplied once by the original linear ramp `0.5 × min(s/512, 1)`. The two kernel estimates entered only as tree features. The evaluator blended the corrected forecasts at the original centers, assigning half the weight to the unchanged 340-feature full model and half to the mean of the three corrected 308-feature members. It then lifted the blended curve to the native survey stations, repeated the 1,024-MD-ft training-donor registration, and recomputed each downstream GR readout on that arm's curve. The fixed assignments were 416 ordinal GR, 13 same-lateral GR, 189 training-conditioned GR and one without a GR adjustment.

Before any of the new fits, two evaluations used the saved original heads. One read the 334-column matrix directly. The other built a 338-column matrix with four zeros and passed its original prefix to those heads. Both reproduced **1,138,704** historical member-center predictions bit for bit. Both also reproduced all **15,172,135** archived station-stage values and all 3,095 per-well stage digests. This check established that disabling the appended features recovered the incumbent through the entire evaluation path.

## Complete field result

All entries are RMSE in feet on the same **619 development wells and 3,034,427 original toe stations**. The fixed gate required the primary to improve pooled final RMSE and win at least two of three outer folds against each comparator.

| Decoder | Center blend, lifted | Registered | Ordinal GR | Added same-lateral GR | Final |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original coherent decoder | 10.477118 | 10.234576 | 9.997405 | 9.994968 | **9.988991** |
| Well-matched kernel features, primary | 10.500328 | 10.246624 | 10.025110 | 10.022445 | 10.016188 |
| Permuted kernel features | 10.479456 | 10.235375 | 10.009624 | 10.007158 | 10.001499 |

| Final readout | Outer fold 0 | Outer fold 1 | Outer fold 2 | Original ordinal support, 416 wells | Remaining 203 wells |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original coherent decoder | 9.253748 | 9.747948 | 10.894236 | 9.853287 | 10.264410 |
| Well-matched kernel features | 9.300881 | 9.784119 | 10.896588 | 9.869222 | 10.313982 |
| Permuted kernel features | 9.287941 | 9.723861 | 10.921049 | 9.873797 | 10.260991 |

| Primary minus comparator, final | Pooled difference (ft) | Paired 95% interval (ft) | Folds won | Wells won / lost / tied |
| --- | ---: | ---: | ---: | ---: |
| Original coherent decoder | +0.027197 | −0.012170 to +0.068964 | 0/3 | 303 / 316 / 0 |
| Permuted kernel features | +0.014689 | −0.019319 to +0.051596 | 1/3 | 299 / 320 / 0 |

The intervals use 4,000 paired whole-well bootstrap draws with PCG64 seed 42, preserving each well's complete station sequence. Both include zero. Across its six fitted heads, the primary used the added features in 888 of 16,800 tree splits and the permuted control used them in 768 of 16,800. Their summed training split gains were 545.0 million and 390.0 million, respectively. These figures describe the trees' use of the features during fitting. Held-well errors supplied the admission decision.

## Replay, compute and scope

The original geological and kernel models were reused. Training parsed only the authorized training-bank targets for its fold. The global model seal preceded outer prediction. Independent checkpoint replay reproduced all **2,277,408** learned member-center outputs exactly. Both learned arms, the incumbent and all five native stages were sealed before the scorer opened outer true TVT. An independent reconstruction reproduced both learned curves through registration and GR on every original station, then verified the literal-row scores, fold and support tables, bootstrap intervals and seven field-worker exits. Its field closure checked 1,225 distinct file identities.

The three sequential four-thread decoder fit workers took 152.63 seconds in total, including model saves. Producing held forecasts and replaying them from reloaded heads took 44.61 worker-seconds upstream. The seven downstream field workers totalled 64.25 worker-seconds, including the two learned native readouts run in parallel, the historical replay, scoring and audit. Elapsed time from the field source freeze through reviews and terminal verification was 633.75 seconds. These are timings for this run on its host.

This experiment reused development wells and prior development findings. All 154 confirmation wells remained sealed. The results establish a development comparison, with no external rank measurement. The private implementations, measurements, fitted models and row-level predictions remain local. [First-party provenance rule](../AGENTS.md)

| Evidence identity | SHA-256 |
| --- | --- |
| Feature-producer contract | `d169dc4a181b350e047feb0085a5014cc7bcdefb23c5831a2cd1288f09c22c62` |
| Matched features and permutations | `dab55080af648ce7ddde5f25cd3156c9bdb647bc6e36dd1cdf93a57094a9323e` |
| Field contract | `d6f5ecd92b06297e23827d934aa4d37deb6a0c24c15a119b0868ed29df07e393` |
| Original and disabled-feature native replay | `23f554c3fa3de86e469879c4fa2b2216ceaffae2c859699083fca39bcbd15684` |
| Independent training targets, feature prefix and medians | `621e46a5ec6b1c412588651610e1103467fb05e2921f4a3c4aeec06e0566b174` |
| Twelve-head training seal | `bb3b898c31559f9500b8e48e92c1dc4ed5f90eb0a4f1f45753fcdb7d2075a710` |
| Held member-center forecast seal | `8487c60a7e5933bfcc1b07ad21f5b4e52fc2983423c14827f24cc31c0c1ad3e7` |
| Independent checkpoint replay | `edde5f6e6930f8b58a1995f52e48060cbf9f79013fb0f29b52884b20b822c1a2` |
| Reviewed three-arm native forecast seal | `49b45e477fffd73f98c9e3121358509cc11c2022ac82d9018f9928dd98cbf27b` |
| Independent target-blind native reconstruction | `5a5d6070d7c650f6dca5a9ff344cc9c4adea35ccd0b4446b6119617f39f541a5` |
| Field terminal verification | `938c81d86fee618b01c0f4f9403d9b1d3c54f99b71a0f0cd4ab64485e6bd1db5` |
| Independent native score and ancestry check | `ab7ec2ef64a16b7206b6a1ffd8e628319ee3423f0cce8ea8fa5afb16382717ed` |
