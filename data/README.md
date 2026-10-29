# Data

Datasets are intentionally **not distributed** in this repository.

Run:

```bash
python scripts/download_data.py
```

This downloads and normalizes the external datasets needed by the real-tabular experiments:

- Adult Income — UCI Machine Learning Repository
- German Credit — UCI Machine Learning Repository
- COMPAS — ProPublica `compas-analysis`
- Glass Identification — UCI Machine Learning Repository
- Wine Quality (red) — UCI Machine Learning Repository

Breast Cancer Wisconsin and Digits are loaded directly from `scikit-learn` and therefore require no local data file.

Please consult the original dataset pages for licenses, citations, and terms of use.
