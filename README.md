# rogii-wellbore-topology

**Late entry scaffold** for the [ROGII Wellbore Geology Prediction](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction) Kaggle competition.

**Lineage:** geometric / middle-out / holonomy components from  
[`ZuluYokohama/topology-sees-sequence`](https://github.com/ZuluYokohama/topology-sees-sequence)  
(adapted for **open wellbore curves**, not cyclic peptides).

| TSS idea | Wellbore use here |
|----------|-------------------|
| Middle-out (known → residual) | Known heel / `TVT_input` section → predict masked toe |
| Bishop holonomy / transport | **Open-curve** parallel transport features on XYZ trajectory |
| E_ref as **prior** (not forced P2) | Structural / geometric prior for TVT residual; never claim OPEN |
| Hard domain reduction | Geometry + GR match first; trees second |
| Peptide Ramachandran / Rosetta | **Out of scope** — not cargo-culted into GR |

## Competition law (short)

- **Target:** TVT (stratigraphic position on typewell column), not Z alone  
- **Metric:** RMSE of dTVT on prediction zone  
- **CV:** GroupKFold by well — never random station rows  
- **Trap:** global TVT↔Z is strong; **within-lateral** slope is ~0  

## Layout

```text
src/rwt/           # package: load, geometry, features, baselines, train, submit
scripts/           # download + synthetic smoke
tests/             # geometry + middle-out unit tests
data/              # raw/ processed/ (gitignored except README)
docs/              # lineage + protocol
notebooks/         # optional exploration
```

## Quick start

```bash
cd rogii-wellbore-topology
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt

# synthetic smoke (no Kaggle data yet)
python scripts/smoke_synthetic.py

# unit tests
python -m unittest discover -s tests -v

# after Kaggle accept + data download into data/raw/
# python -m rwt.train --data data/raw --out artifacts/run1
# python -m rwt.submit --model artifacts/run1/model.joblib --out submissions/late1.csv
```

### Download data (when ready)

```bash
# requires: pip install kaggle  +  ~/.kaggle/kaggle.json  + competition rules accepted
kaggle competitions download -c rogii-wellbore-geology-prediction -p data/raw
# unzip into data/raw/
```

## Package install (editable)

```bash
pip install -e .
```

## Status

| Phase | Status |
|-------|--------|
| Repo scaffold + TSS bridge geometry | **OPEN** (this tree) |
| Kaggle data ingest | **NEED_INFO** (download) |
| Baselines + LGBM GroupKFold | **READY** (code paths) |
| First late submit | **pending data** |

## Attribution

- Geometry transport adapted from `topology-sees-sequence` (`holonomy_extract.bishop_holonomy` / rotation helpers)  
- Competition context: ROGII × Kaggle  
- Domain feature ideas (tortuosity, NCC): public competition discussion / toolkits (cite in submit notes)

## License

Research / competition entry code — use with attribution to TSS lineage.
