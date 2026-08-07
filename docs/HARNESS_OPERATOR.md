# How the harness drives this engine (read this)

You have **two machines glued by one law**. They are not the same process.

```text
┌─────────────────────────────────────────────────────────────┐
│  HARNESS (supagen / prime-session MCP)                      │
│  Law: restrict → measure → audit → OPEN | STOP              │
│  Role: decide WHAT is allowed, certify claims, never force  │
│  Silicon: agent + (optional) LMS SCOUT/PRESERVE + jina NLI  │
└──────────────────────────┬──────────────────────────────────┘
                           │ measures / claims / certificates
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  DOMAIN ENGINE (rwt / rogii-wellbore-topology)              │
│  Law: M0–M6 abstraction · Gate3 residual · GroupKFold       │
│  Role: TVT RMSE · beat 5 ft · geometry+PF+trees             │
│  Silicon: CPU 24c (trees/geom) · GPU optional nets · NPU n/a│
└─────────────────────────────────────────────────────────────┘
```

## What each silicon owns (this PC)

| Unit | Owns | Does **not** own |
|------|------|------------------|
| **CPU Ultra9 24c** | Geometry, PF, NCC, LGBM, GroupKFold, CV | LMS chat |
| **GPU 5060 8GB** | LMS fibers (LFM/frankenstein), future neural residual, embeds | Default TVT score path |
| **NPU AI Boost** | Future always-on Job2 measures | Kaggle metric authority |
| **DDR5 32GB** | Q8 models, station matrices, long folds | — |

**Max precision ≠ thrash GPU on trees.**  
**Max precision = float64 geom + rich priors + Gate3 honesty + deep LGBM when residual earns it.**

## Proper operator loop (every serious enter)

1. **RESTRICT** (prime MCP or human sheet)  
   Goal: beat **5 ft** RMSE · non-goal: force OPEN · success: OOF selected &lt; 5  
2. **MEASURE** (domain)  
   ```powershell
   cd D:\sandbox\rogii-wellbore-topology
   .\.venv\Scripts\Activate.ps1
   $env:OMP_NUM_THREADS=20; $env:RWT_NUM_THREADS=20
   python scripts\train_max_precision.py --max-wells 64   # pilot
   python scripts\train_max_precision.py --max-wells 0    # all 773
   ```  
3. **MEASURE** (harness instruments, when LMS up)  
   `supagen ensure` · jina aboutness · dual_enter face (cosine never OPEN)  
4. **AUDIT**  
   - selected OOF &lt; 5 ft → candidate **GO_MEASURE**  
   - Gate3 fallback pure prior if residual loses floors (residue never forced)  
   - production OPEN only after submit + external process  
5. **CERT** write JSON under `artifacts/` + optional `cert_write` MCP  

## Slash commands (optional)

| Slash | Use |
|-------|-----|
| *(none)* | Preferred: talk natural language; agent calls prime MCP |
| `/prime` | Explicit session door |
| `/harness-pipeline` | Multiplane field packs (Filmore ops), not Kaggle RMSE |
| `/harness-smoke` | Offline harness green |

**You do not need slashes** for the Kaggle engine. You need **MEASURE scripts + honest AUDIT**.

## Why Task Manager is quiet during “harness work”

Harness MCP = control plane (JSON, law).  
GPU lights when LMS/jina/torch run.  
CPU lights when `train_max_precision` / feature assembly runs.  

If only chat is open → quiet meters. That is correct.

## Target

**Beat 5′** = `oof_rmse_selected < 5.0` on GroupKFold-by-well for prediction-zone TVT.
