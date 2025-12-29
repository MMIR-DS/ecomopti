```markdown
# Phase 1: Data Engineering


**Prerequisites**: Python 3.9+, install dependencies via `pip install -r requirements.txt`


**Run:** `python -m ecomopti.phase1.pipeline`

**Input:**  
- Raw dataset expected at `data/raw/telco.csv` (or pass a different filename to `run_pipeline(raw_filename="...")`)

**Produced Artifacts:**
- `data/splits/train.csv`, `val.csv`, `test.csv`  
  Processed, encoded splits with metadata columns (`E`, `T`, `customerID`) + one-hot encoded features
- `artifacts/phase1/preprocessor.pkl` → fitted `ColumnTransformer` (full feature pipeline)
- `artifacts/phase1/encoder.pkl` → fitted one-hot encoder (categorical part only)
- `artifacts/phase1/scaler.pkl` → fitted median imputer (numeric part only)
- `artifacts/phase1/manifest.json` → lineage & reproducibility metadata (hash, split sizes, churn rates, etc.)

**Configuration:** `src/ecomopti/phase1/config.py`

**Tests:** `pytest -q`

---

## Overview

Phase 1 performs end-to-end data preparation for downstream survival/CLV modeling on the Telco churn dataset:

1. Loads raw CSV
2. Cleans data (type conversion, explicit drop of `TotalCharges`)
3. Creates survival targets `E` (event/churn indicator) and `T` (observed time/tenure)
4. Validates cleaned data with Pandera schemas
5. Stratified train/val/test split (70/15/15) on `E`
6. Fits preprocessing pipeline **only on training data**
7. Transforms all splits consistently
8. Validates processed splits
9. Saves processed CSVs + fitted artifacts + manifest

---

## Key Design Decisions

### Leakage Prevention (Critical for CLV)
- `TotalCharges` is **explicitly dropped** after conversion to prevent future-information leakage into lifetime value predictions.
- All preprocessing (imputation medians, one-hot categories) is fitted **exclusively on the training set**.

### Target Definition
- `E`: binary churn event (1 = "Yes", 0 = "No")
- `T`: observed tenure in months (missing → 0 with warning logged)

### Feature Handling
| Type       | Columns                                                                 | Preprocessing                          | Scaling? |
|------------|-------------------------------------------------------------------------|----------------------------------------|----------|
| Metadata   | `E`, `T`, `customerID`                                                  | Passed through unchanged               | No       |
| Numeric    | `tenure`, `MonthlyCharges`, `SeniorCitizen` (0/1)                       | Median imputation                      | **No** (preserves monetary interpretation) |
| Categorical| `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`,<br>`InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`,<br>`TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`,<br>`PaperlessBilling`, `PaymentMethod` | Most-frequent imputation → One-hot encoding<br>(`handle_unknown="ignore"`) | N/A      |

### Reproducibility & Robustness
- Fixed `RANDOM_STATE = 42`
- Raw data hash stored in manifest for versioning
- Comprehensive error handling with automatic cleanup of partial artifacts on failure
- Path traversal protection on input filename
- Pandera schema validation at cleaned & processed stages

### Directory Structure Created
```
data/
├── raw/                ← place telco.csv here
└── splits/             ← train.csv, val.csv, test.csv
artifacts/phase1/
├── preprocessor.pkl
├── encoder.pkl
├── scaler.pkl
└── manifest.json
```

Run the pipeline once and all downstream phases can safely load the processed splits + fitted preprocessor.
```