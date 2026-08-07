# Abstraction modes — late entry thesis

**Competition:** [ROGII Wellbore Geology Prediction](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction)  
**Lineage:** [topology-sees-sequence](https://github.com/ZuluYokohama/topology-sees-sequence)  
**Claim class:** **GO_MEASURE** of a *mode of computation* — not a claim that topology alone wins RMSE.

---

## What we are demonstrating

Most leaderboard entries treat the problem as:

```text
flatten stations → big feature matrix → gradient-boosted trees → TVT
```

This entry demonstrates a **layered abstraction pipeline** borrowed from topology-sees-sequence:

| Mode | Abstraction | Wellbore operator | TSS analogue |
|------|-------------|-------------------|--------------|
| **M0 · RESTRICT** | Bound the problem | TVT RMSE, GroupKFold-by-well, no station leakage | RESTRICT / law |
| **M1 · GEOM** | Trajectory as open curve in ℝ³ | Tangents, curvature, open Bishop transport, tortuosity proxy | Bishop / frame transport |
| **M2 · PRIOR** | Known section as reference connection | Flat-formation C−Z, last TVT, MD slope on *known only* | E_ref as prior (not forced P2) |
| **M3 · MATCH** | Log as sequence matched to typewell | Multi-scale GR NCC → typewell lag / TVT hint | Sequence–structure coupling (domain-reduced) |
| **M4 · MIDDLE_OUT** | Slice known → residual toe | Prediction-start split; predict only mask | middle_out_v2 slice 0 |
| **M5 · RESIDUAL** | Learn correction, not absolute TVT | LGBM on (TVT − prior_flat) | Residue never forced as OPEN |
| **M6 · CERTIFY** | Measure, don’t market | CV report, baseline beat-table, submit only after MEASURE | AUDIT / certificate |

**Novel angle for a late entry write-up:**

> Topology does not replace geosteering correlation; it **orders the computation** — geometry and priors first, match second, residual learner last — so that the model is constrained by the same middle-out / hard-domain-reduction discipline that made TSS claims corrigible.

---

## What we are *not* claiming

| Non-claim | Why |
|-----------|-----|
| Peptide Ramachandran applies to GR | Wrong domain |
| Closed holonomy of a well is zero ⇒ TVT | Wells are open curves |
| Sheaf λ₁ is the Kaggle metric | Metric is RMSE(dTVT) |
| Abstraction modes alone beat all trees | Trees still own residual fit |
| Production geosteering OPEN | Competition MEASURE only |

---

## Submission narrative (for Kaggle description / discussion)

1. **Problem reduction:** stratigraphic position (TVT), not depth Z.  
2. **Within-well TVT–Z decoupling** respected via residual-to-prior, not raw Z regression.  
3. **Open-curve holonomy features** from TSS rotation transport, re-derived for polylines.  
4. **Middle-out:** heel is the reference section; toe is the residual problem.  
5. **GroupKFold by well** as the only honest CV for late entry.  
6. **Artifact trail:** `artifacts/*/cv_report.json` + baseline RMSE table = certificate of MEASURE.

---

## Mode graph

```text
        RESTRICT (M0)
             │
             ▼
     ┌─── GEOM (M1) ◄── XYZ/MD trajectory
     │
     ├── PRIOR (M2) ◄── known TVT_input / pre-PS
     │
     ├── MATCH (M3) ◄── GR × typewell
     │
     └── MIDDLE_OUT (M4) ── split heel | toe
             │
             ▼
        RESIDUAL (M5) ── tree on (y − prior)
             │
             ▼
        CERTIFY (M6) ── CV + baselines + submit
```

Each mode has a pure function surface under `src/rwt/` so the “abstraction stack” is executable, not only a blog post.
