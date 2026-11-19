# Intake review of the supplied OODA v0.4 assistance

This review authenticates archive identities, cohort metadata and arithmetic in the supplied report. It does not independently measure the operator's accuracy. The active nested calibration study retains its frozen sources, parameters and controls. [First-party design rule](../AGENTS.md) · [Current experiment](NESTED_FIELD_CALIBRATION.md)

The supplied `rwt-ooda-value-alignment-v0.4.zip` has SHA-256 `f9d4b053b5b7a1d5b5731a3c541ec8638d95088f1953b72a9cf897a3768c7b0d`. Its report describes the first 36 IDs in the original development manifest. Those IDs are disjoint from the 154 confirmation IDs. Its three claimed 12-well partitions contain 54,833, 60,394 and 58,329 toe stations, totaling **173,556**. The full stored station counts also match the original census: 233,944 across the 36 wells.

## Reported results and what they establish

| Supplied report scope | Wells | Standalone RMSE, ft | Coupled RMSE, ft |
| --- | ---: | ---: | ---: |
| First development slice | 12 | 7.124 | 6.088 |
| Second development slice | 12 | 6.078 | 5.977 |
| Last slice, described by its author as blind | 12 | 10.199 | 6.832 |
| Pooled supplied subset | 36 | 7.991 | 6.311 |

An independent metadata check recovered the pooled values from the supplied per-well RMSEs and original toe counts. The coupled value uses the printed standalone error on 26 accepted wells and the printed repository-geometry error on ten abstentions; the last slice uses nine and three, respectively. This verifies arithmetic in the report, not predictions against original targets.

The supplied last-slice interval for coupled minus its archived ordinal comparator is **−14.380 to +0.472 ft**. It includes zero. The supplied pooled interval is **−8.734 to −1.808 ft**. Both intervals and the comparator scores remain the author's reported measurements. They refer to this subset and its historical baseline, rather than the current 619-well development forecast.

## Evidence boundaries

| Item | Intake finding |
| --- | --- |
| Frozen source bytes | The two module hashes and one test hash in the supplied freeze contract match the archive. The study runner is outside that freeze. |
| Prediction artifact | The supplied seal matches one NPZ, SHA-256 `b68c69f6966b9be7d8910250c6c0d0fa0274bff81fe1d12c85df8cf341572d55`. Its 36 float64 matrix headers match the original full-station census. |
| Coupled forecast | The seal says `conditioned_mode=false` and `fallback_npz_mode=false`. It supplies no separate hash for the fallback native array, admission vector or final coupled forecast. |
| Raw-source and station mapping | The archive has no independent raw-input hash/read-role record or explicit original-station/prediction-column manifest for the supplied matrices. Matching dimensions alone does not establish those mappings. |
| Blindness and confirmation access | Freeze/download timestamps and zero-target/confirmation statements are self-reported. Supplied execution and access receipts do not independently establish acquisition order or actual reads. The 36 reported IDs themselves are valid development IDs. |
| Design origin | Module headers assert first-party origins in authorized measurements and standard mathematics. A byte comparison found no nonempty file identical to the earlier supplied v0.2.1 archive. That observation does not establish independent design provenance or authorize a transfer of implementation. |

The metadata audit imported no supplied module and decoded no prediction payload, raw target or first-party forecast to validate these scores. It supplies no admission decision for a new candidate. Further merit evidence requires authenticated first-party design and dependencies, a frozen source/configuration and actual source-bound forecasts on a declared population, followed by separate original-row evaluation. The 36-well headline cannot establish a 619-well result or an external rank.

Independent intake receipt: `f96c7878605ba38f00313b902225287a70348672cc8c632c2b79b70710e2b167`. Raw artifacts and the supplied implementation remain local.
