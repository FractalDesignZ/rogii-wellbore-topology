# IP governance & competition isolation (operational)

**Status:** architectural law for this repository  
**Not legal advice.** Patent filings, commercial contracts, and claim language require a registered IP attorney.

---

## 1. Open-core / dual-layer pattern

```text
┌─────────────────────────────────────────────────────────────────┐
│ PUBLIC (this GitHub repo · AGPLv3)                              │
│  • Pydantic/CLI · CSV loaders · abstraction mode harness        │
│  • Baselines · GroupKFold train · submit CSV builders           │
│  • Public geometry baseline (open-curve transport lite)         │
│  • Benchmarks · tests · documentation                           │
├─────────────────────────────────────────────────────────────────┤
│ PROPRIETARY KERNEL (private_kernel/ · NEVER public push)        │
│  • Advanced holonomy / sheaf features beyond published baseline │
│  • λ₁ / coboundary / crystal-style gates if used                │
│  • Middle-out proprietary residual maps                         │
│  • Watermark secrets · enterprise adapters                      │
└─────────────────────────────────────────────────────────────────┘
```

| Layer | Location | License | Push to public GitHub? |
|-------|----------|---------|-------------------------|
| Public interface | `src/rwt/*` (except private loader secrets) | **AGPLv3-or-later** | Yes |
| Proprietary kernel | `private_kernel/` (gitignored) or separate **private** repo | Closed / commercial | **No** |
| Competition export | `submissions/*.csv`, optional feature arrays | Outputs only | Yes (predictions) |

**Load rule:** `RWT_KERNEL=public|private` (default `public`).  
Private kernel is imported only if present and env allows — public CI never requires it.

---

## 2. Licensing governance

| Use case | License path |
|----------|----------------|
| Hobby / research / AGPL-compatible SaaS | AGPLv3 (this repo) |
| Enterprise closed SaaS without source | **Commercial dual license** (contact copyright holder) |
| Cloud service using this code | AGPL network-use obligation applies to AGPL code |

Copyright remains with the copyright holder for dual-licensing.

**Patents:** provisional applications (if any) are outside this repo. Example *themes* for attorney drafting only:

1. Quality-gated synthesis using spectral-gap style gates (if productized in software systems).  
2. Subsurface trajectory alignment via fiber-style parallel transport features (system/method claims).  

Pure abstract math is not patentable; concrete systems may be — attorney required.

---

## 3. Cryptographic provenance & prior art

| Mechanism | Implementation |
|-----------|----------------|
| Git commit history | Public GitHub timestamps |
| Content seals | `rwt.provenance.seal_artifact` → SHA-256 over payload + public code digest |
| Domain / late-entry certificates | `artifacts/*_certificate.json` with digests |
| Optional OpenTimestamps | `RWT_OPENTIMESTAMPS=1` (if `ots` client installed) |
| Feature watermark | `rwt.watermark` deterministic column / LSB-style mark |

These establish **technical prior art trails**, not a substitute for patents.

---

## 4. Competition isolation (law-safe)

```text
PRIVATE MACHINE
  private_kernel (if any) → features / TVT_hat
        │
        ▼
  export_competition_bundle.py
        │  only: submission.csv · public feature matrix · provenance seal
        ▼
PUBLIC / KAGGLE
  notebook or CLI that trains trivial head OR uploads submission.csv
  NEVER upload private_kernel source
```

Rules:

1. Run custom harnesses on private hardware.  
2. Compute proprietary features only in private workspace.  
3. Export **predictions** or **public-schema features** only.  
4. Keep mathematical OS (private kernel) out of competition notebooks.  

---

## 5. What already went public (honest)

Open-curve Bishop-lite transport and middle-out public priors were published in early commits of this repo. That is **prior art** (good for provenance) and **not secret**. Future novel kernel advances go only into `private_kernel/` or a private repo.

---

## 6. Operator checklist

- [ ] Never `git add private_kernel/`  
- [ ] Set `RWT_WATERMARK_KEY` in private env (not committed)  
- [ ] Before Kaggle upload: `python scripts/export_competition_bundle.py`  
- [ ] Seal CV reports with provenance  
- [ ] Commercial inbound → dual-license path, not AGPL-only assumption  
