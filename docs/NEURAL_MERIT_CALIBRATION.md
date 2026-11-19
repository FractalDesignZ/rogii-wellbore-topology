# Calibrating neural action weights on excluded wells

The [attention studies](ATTENTION_MEMBER_DECODER.md) showed why fitting loss cannot supply a deployment weight: the two-way network's BANK error fell from 11.365080 to 5.456911 ft, while its complete field route still missed the incumbent. This follow-up asks whether a separately calibrated amplitude and family choice transfer to wells excluded from neural fitting. It measures original 308-member paths, before the full ensemble, geographic registration and GR readout. Its errors do not share the final field forecast definition.

## Source and roles

Each of nine original 308 predictors excluded an entire well bank, as well as its outer-query wells. Within that single bank, outcome-blind hashes assign five whole-well groups. Each of five rotations reserves one group for TEST, the next for CAL and the remaining three for neural FIT. Across 45 rotations there are 81–84 FIT wells and 27–28 wells in each held role. Every one of 619 development wells appears twice in TEST, for 1,238 member/well cases and 6,068,854 original-station appearances.

Both sibling predictors trained on every well in this bank. Their outputs and the earlier pooled scalers are excluded from this study. Each rotation uses only its own upstream member's unscaled observations and creates a fresh FIT-only scaler. Its two fresh 60,704-parameter networks use the same one-way and two-way architectures, masks, initial parameter stream and whole-well order. Neither the eight-pass nor 64-pass trained neural weights enter these fits. The [first-party design rule](../AGENTS.md) applies.

Each model receives 32 complete FIT passes, with at most eight wells per batch: 352 AdamW updates, learning rate .001, weight decay .01 and seed 42. Ninety models therefore receive 31,680 updates. This smaller FIT population and budget are distinct from the earlier full-outer-bank fits. All final checkpoints are retained; there is no validation checkpoint or epoch selection.

## Exact native objective and learned amplitude

For a predicted physical endpoint pair `u`, the native correction remains `Φu`, with `Φ=[r(1−p), rp]`, `r=1−exp(−s/512)` and `p=s/L`. Define `A=uᵀMu` and `C=bᵀu` from the keyed original-station quadratic. An amplitude `α` gives:

`L(α) = SSE0 + 2αC + α²A`.

For any authorized CAL subset, `α = clip(−sum(C)/sum(A), 0, 1)`, with zero for absent or invalid support. During CAL comparison each well uses the amplitude estimated from the other CAL wells. Its own target terms cannot set its amplitude. The [public operator selector](../src/rwt/operator_selection.py) compares OFF, one-way and two-way using these paired leave-one-well-out native SSEs. The unique lowest pooled error must improve on OFF by more than one whole-well standard error, with at least twelve CAL wells. Ties and insufficient evidence retain OFF.

After choosing a family, all CAL wells estimate its final amplitude for TEST. The neural weights remain fixed. This is one learned amplitude and one admission decision; no second merit factor reuses the same calibration losses. It is distinct from the visible-heel reliability supplied as a neural input.

```mermaid
flowchart LR
  B[One upstream-excluded bank] --> F[Fresh neural FIT and scaler]
  F --> P[Seal all CAL and TEST forecasts]
  P --> C[Release CAL and estimate held merit]
  C --> A[Seal family and amplitude for every TEST well]
  A --> T[Release TEST and score original stations]
```

The matched controls are OFF, unconditional one-way at full amplitude, unconditional two-way at full amplitude, and the same admitted family at amplitude 0.5. The admitted family at full amplitude is also reported. The fixed primary gate requires lower pooled native error and improvements in at least two of three outer contexts against each of the four matched controls. No substitute action or selected-epoch result can replace a failed primary.

## Execution and verification scope

All 135 FIT/CAL/TEST input packs and 45 FIT-only scalers are sealed before numeric objectives are issued. FIT workers receive only their own physically separate FIT objective files. The custodian decodes authorized float64 rows directly from the original stored NPY members: two `b` values and one `SSE0` per well, with role checks before numerical access. Geometry, row counts, original CSV rows and global station indices are checked independently by the feature producer and issuer.

All ninety model forecasts must be sealed before CAL objectives are released. All 45 family/amplitude decisions and their complete TEST actions must be sealed before TEST objectives are released. Positive read allowlists, source-byte and AST identities, process receipts, actual loaded checkpoints and original observation footprints support each boundary. These are application-level data-flow checks, without an operating-system sandbox claim.

Independent pre-fit verification covered all 6,190 rotated well-role appearances and all 31,320 scaler columns. Finite counts matched; independently calculated population deviations differed by at most 4.53×10⁻¹¹. Synthetic native gradients matched to 6.66×10⁻¹⁶, and direct least squares reproduced the leave-own-out calibration and errors to 1.35×10⁻¹³. The FIT issuer disclosed 3,714 role/well objective records, exactly 89,136 payload bytes. Models and private operators remain local.

Uncertainty resamples 619 original well IDs, grouping their two TEST records, with 4,000 PCG64(42) draws. Per-bank and per-rotation results accompany outer-context totals. Shared upstream models and overlapping fitting sets limit the interpretation of these intervals. Previous studies used these development outcomes; the new information boundaries establish prospective decisions within this study, without historical blindness or independent confirmation.

The final native pipeline would require a separate nested reconstruction. Existing sibling members, full heads, coherent decoders, donor registrations and GR profiles can include these bank labels. They cannot be reused as a clean common CAL/TEST endpoint merely because the new neural fit excluded its test wells. All 154 confirmation wells remain sealed.

Local frozen identities:

- Design/role logical root: `fb681b6bfb1b45b2023b683826a2bbef70394385b0762abe78343b06e4332345`.
- Neural contract: `73e661b608930f8fe047e18aa8f9b46dff6e9f10dfc2f376f5fd0e548885abf0`.
- Feature/scaler seal: `62744dc95850de1f1e1a6e0cd356a3b27d22516e25364b11927ab8134bbbdff1`.
- Custody contract: `f7be838d46786ed5eb115e06706518fd0e055e4dc8559c78af5c8a02349fa27c`.
- Independent pre-fit audit: `12c9e4ecd9a820b47cc005bc1694ea19268449da37466715df119657a502288c`.

## TEST result

Calibration retained OFF in 41 rotations, admitted one-way in one rotation and two-way in three. The four admitted amplitudes were 0.615205, 1.000000, 0.507702 and 0.647796. Their median was 0.631501. These choices assigned nonzero actions to 109 of 1,238 TEST records. Every record remained in the score.

| Sealed TEST readout | Native RMSE (ft) | Adaptive minus control | Winning outer contexts |
| --- | ---: | ---: | ---: |
| OFF | 12.322270 | −0.000775 | 1/3 |
| Always one-way, amplitude 1 | 13.492799 | −1.171303 | 3/3 |
| Always two-way, amplitude 1 | 13.339241 | −1.017746 | 3/3 |
| Admitted family, amplitude 0.5 | 12.309841 | +0.011654 | 0/3 |
| Admitted family, amplitude 1, diagnostic | 12.373406 | −0.051911 | 2/3, one tie |
| Adaptive amplitude, primary | **12.321495** | — | — |

The primary gate failed. Adaptive weighting's difference from OFF had a paired whole-well 95% interval of **−0.054302 to +0.048426 ft**. Its difference from admitted-half had interval **−0.007214 to +0.030137 ft**. Against the always-one and always-two controls those intervals were −1.618239 to −0.719306 ft and −1.438215 to −0.595001 ft. The admission rule prevented most unconditional neural actions; fitting the amplitude on CAL did not establish an advantage over OFF or fixed half strength.

Adaptive error in the three outer contexts was 12.743638, 12.405493 and 11.796649 ft. OFF scored 12.731247, 12.402734 and 11.815342 ft. The complete nine-bank and 45-rotation tables remain in the local scored receipt. This member-only comparison leaves the retained **9.988991-ft** complete field route unchanged. No hybrid was assembled from these held outcomes.

All ninety checkpoints independently reproduced 4,952 CAL/TEST endpoint pairs exactly. Independently reconstructed FIT forecasts recovered the saved losses for all 7,428 arm/FIT endpoint vectors. All 9,180 optimizer state tensors were finite, and every parameter state recorded step 352. The ninety final parameter hashes were distinct. Measured fit/evaluation work totalled 599.84 seconds; the 45 rotation workers totalled 734.97 seconds. Peak PyTorch tensor allocation was 184.80 MiB; total process and device memory were not measured.

After the exact TEST-choice seal was approved, a separate audit rebuilt all 1,238 original member curves from saved block predictions, original centers, MD/Z and visible heel. Every native baseline hash matched. Direct evaluation of **36,413,124 action–station residuals** reproduced all six arms' native errors, with at most 2.61×10⁻⁸ ft² discrepancy from the per-well quadratic. Only each rotation's authorized TEST TVT rows were parsed, with off-role negative checks and no confirmation reads. The independent audit also recovered every fold and the 4,000 clustered bootstrap intervals.

The complete closure rehashed 2,146 unique bound artifacts and all 225 custody/calibration/scoring workers plus 57 training-stage processes and their logs. The numerical issuer decoded exactly 89,136 FIT, 29,712 CAL and 29,712 TEST payload bytes under the recorded stage order. The result is evidence for this fixed selector and these fresh small-bank neural candidates. It establishes no general advantage of adaptive weighting, independent validation or external rank.

- All-forecast seal: `919c9c01b77b1a74f6a8df0e3e68058e997c1614e3550d314e9565dca981bf7b`.
- Independent checkpoint replay: `b5ee4c2cf7a456357059cd3270775b8709cba4dcb9387e7356b99c02042b6e64`.
- Complete family/amplitude TEST-action seal: `b3d58a88d2c129cf841e921aca12a45a7d030857c6e0c2a29d18a45fbf47280a`.
- Custody/scoring logical root: `0cc53d232d371ec1f63ac500794ca6451774efa8e335adebbc677b4634b72872`.
- Independent literal-row score audit: `811e455f96ca86cafd41e8748395b7af14ca5d028d72067a462b42f3fb6099b5`.
- Full union closure: `97d3ef97ec802e2ed6b5b4d8334fbccb03c430041de875dd8e51bee87a38f9f5`.
