# Network Engine integration

The initial integration provides a reproducible measurement boundary between this public wellbore harness and an optional, separately licensed Network Engine. It exports measurements for offline relational diagnostics. It does not modify the TVT predictor, introduce a new prior, or establish an accuracy improvement.

## Measurement contract

`rwt.network_bridge.build_network_input` accepts one lateral with at least four stations. `MD` must be finite on every row. The source depth unit must be declared as `ft` or `m`; MD and Z are converted to metres. GR retains its source instrument unit.

The exported channels are exactly `MD`, `Z`, and `GR`, with a presence mask for each channel. Existing RWT curve-null handling normalizes LAS null sentinels in Z and GR before masks are built. Missing Z or GR produces a masked channel. Missing support remains missing rather than being interpolated. Acquisition order, repeated MD values, and reversals are preserved; the bridge does not sort or resample.

MD is the analysis axis. The native container also carries a sample ordinal named `t` to preserve station order. That ordinal is not measured time or seconds. Names inherited from the native relation schema do not establish a temporal or causal interpretation of the wellbore data.

The bridge never reads or exports `TVT`, `TVT_true`, `TVT_input`, formation tops, priors, trained-model features, or proprietary feature columns. Mutating those columns must leave the input identity unchanged. Proprietary engine sources are neither bundled nor required by the public preparation path.

## Prepare inputs

```bash
# Practice fixture only; its depth unit is metres
python scripts/measure_network_dynamics.py --synthetic

# Raw lateral CSV; use the actual source depth unit
python scripts/measure_network_dynamics.py --csv path/to/lateral.csv --well-id WELL_ID --depth-unit ft

# Choose a fresh output location
python scripts/measure_network_dynamics.py --synthetic --out artifacts/network_practice_001
```

`--csv` requires `--depth-unit`. The synthetic fixture accepts metres only and is explicitly marked synthetic in its manifest. It checks the integration workflow; it is not field evidence.

The script creates a new timestamped directory under `artifacts/` unless `--out` is supplied. An existing output directory is rejected so an earlier receipt cannot be overwritten.

| Artifact | Contents |
| --- | --- |
| `data.npz` | Measurements, observation masks, and the station ordinal |
| `spec.json` | Channels, declared units, MD axis, and relation requests |
| `mode.json` | Native slice detector configuration; legacy windowed dynamics disabled |
| `manifest.json` | Input identity, data/settings hashes, preprocessing policy, support counts, label policy, and public-code provenance seal |

The input hashes describe the exported data and settings. Provenance seals also record the public code digest and Git revision. Runs remain local; this command does not submit competition predictions.

## Optional native diagnostics

Install the separately licensed engine in the Python environment used for this command, then add `--run-local`:

```bash
python scripts/measure_network_dynamics.py --synthetic --run-local
python scripts/measure_network_dynamics.py --csv path/to/lateral.csv --well-id WELL_ID --depth-unit ft --run-local
```

The adapter invokes the native slice API and requires the requested detector and successful native property checks. It refuses unsupported execution rather than substituting another detector. The older windowed dynamics level remains disabled. Public preparation and boundary tests work without an engine installation; native integration tests are conditional on its availability.

Native outputs are saved beneath `native/`. Its sealed `bridge_receipt.json` links to the input hash and records engine identity, native hashes, diagnostic record counts, property-check status, and save warnings. These outputs describe relationships in the supplied measurements; they are not TVT predictions or evidence of physical cause.

**Known native export limitation:** the optional viewer file `involvement.json` can fail to save even when slice diagnostics and native property checks complete. The receipt records `save_warnings` and `viewer_involvement_available`. The bridge preserves that distinction; it does not repair the separate viewer exporter or claim the optional file exists.

## Spatial observation policy

The public neighbor graph has a separate label boundary. `build_spatial_index` and `well_spatial_node_from_lateral` default to `label_policy="observed_only"`: finite `TVT_input` values are the only donor observations. Missing or empty `TVT_input` never falls back to full `TVT`.

Target geometry can query registered donors without being inserted into their index. A target cannot transfer its own observations to itself. Missing XY remains missing and cannot create a fabricated origin node or station transfer.

`label_policy="full_tvt"` explicitly permits full training-donor labels. This option is unsafe for features assembled before GroupKFold unless donor visibility is reconstructed inside each fold: held-out toe labels must never contribute to either training or validation features. The default observed-only path avoids this full-label dependency. Node and index metadata record the selected policy.

**Historical OOF results require remeasurement.** The earlier spatial builder preferred full TVT before well-group splitting, and unseen prediction wells received no neighbor transfer. The corrected policy changes both label visibility and serving behavior. Existing reported scores cannot certify the corrected pipeline.

## Path to predictive integration

1. Validate measurement support, units, acquisition order, and reproducible input identities on real lateral data.
2. Establish new held-out-well baselines under the corrected spatial policy. Keep all target labels outside diagnostic inputs.
3. Evaluate a proposed dynamics-derived prior or confidence gate against those baselines, with train/serve parity and synthetic falsifiers for missing support and misleading relationships.
4. Promote a predictive change only after new held-out-well results and Gate3 justify it. Record the provider identity, observation policy, metrics, and provenance with that result.

Until those measurements exist, the integration remains a diagnostic capability. Advanced engine implementations stay behind the separately installed provider boundary described in [IP governance](IP_GOVERNANCE.md).
