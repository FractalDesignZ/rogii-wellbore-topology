# Data

Place Kaggle competition files under `raw/` (gitignored).

Expected layout (adjust after inspecting the zip):

```text
data/raw/
  train/
  test/
  sample_submission.csv   # if provided
```

Download:

```bash
kaggle competitions download -c rogii-wellbore-geology-prediction -p data/raw
```
