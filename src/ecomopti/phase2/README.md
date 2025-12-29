```markdown
# Phase 2: Survival Modeling & Customer Lifetime Value (CLV) Prediction

**Prerequisites**: Python 3.9+, install dependencies via `pip install -r src/ecomopti/phase2/requirements.txt`

**Run Full Pipeline (Recommended):**  
`python -m ecomopti.phase2.build`

**Alternative (manual):**  
```bash
python -m ecomopti.phase2.train_cox
python -m ecomopti.phase2.eval_cox
python -m ecomopti.phase2.train_rsf
python -m ecomopti.phase2.eval_rsf
python -m ecomopti.phase2.train_clv_model
python -m ecomopti.phase2.eval_clv
python -m ecomopti.phase2.plots
```

**Prerequisite:** Phase 1 must be completed first (`data/splits/` and `artifacts/phase1/preprocessor.pkl` must exist).

**Produced Artifacts:**
- `models/phase2/cox_model.pkl` → fitted Cox Proportional Hazards model  
- `models/phase2/cox_summary.csv` → hazard ratios and coefficients  
- `models/phase2/rsf_model.pkl` → fitted Random Survival Forest  
- `models/phase2/rsf_feature_importance.csv` → permutation importance (optional)  
- `models/phase2/clv_model.pkl` → fitted CLV regression model (LightGBM)  
- `models/phase2/*_metrics.json` → evaluation metrics for all models  
- `artifacts/phase2/preprocessor_clv.pkl` → leakage-free feature selector for CLV  
- `artifacts/phase2/clv_*_predictions.csv` → actual vs predicted 6-month CLV for train/val/test  
- `artifacts/phase2/customer_segments_test.csv` → risk vs CLV customer segmentation  
- `plots/phase2/*.png` → all diagnostic and interpretability visualizations  

**Configuration:** `src/ecomopti/phase2/config.py`

**Tests:** `pytest -q`

---

## Overview

Phase 2 builds and evaluates survival and CLV models using the processed splits from Phase 1:

1. Trains Cox Proportional Hazards model (interpretable baseline)
2. Trains Random Survival Forest (high-performance ensemble)
3. Computes 6-month survival probabilities (cached from Cox)
4. Constructs leakage-free 6-month CLV target:  
   `CLV₆ = Survival_Probability(6 months) × MonthlyCharges × 6`
5. Trains CLV regression model (LightGBM) using only non-leakage features
6. Evaluates all models (C-index, Brier/IBS, MAE, R², MAPE, calibration)
7. Generates comprehensive plots and diagnostics

---

## Key Design Decisions

### Leakage Prevention (Critical for Valid CLV)
- `MonthlyCharges`, `tenure`, and derived survival probabilities are **explicitly excluded** from CLV model features.
- Automated leakage audit runs before CLV training (fails fast if violation detected).
- Survival probabilities are computed using Cox model and cached for consistent CLV target creation.

### Model Overview
| Model                  | Purpose                          | Key Features                              | Evaluation Metrics                     |
|------------------------|----------------------------------|-------------------------------------------|----------------------------------------|
| Cox PH                 | Interpretable survival baseline  | Linear, hazard ratios                     | C-index (with bootstrap 95% CI), Brier  |
| Random Survival Forest | High-performance survival        | Non-linear, permutation importance        | C-index, Integrated Brier Score (IBS)  |
| CLV (LightGBM)         | 6-month revenue prediction       | Leakage-free features only                | MAE, R², MAPE                          |

### Performance (Run: 2025-12-29)
| Model       | Split | C-index (95% CI)       | IBS     | MAE     | R²     | MAPE   |
|-------------|-------|------------------------|---------|---------|--------|--------|
| Cox PH      | Val   | 0.8441 (0.8235–0.8646) | –       | –       | –      | –      |
|             | Test  | 0.8576 (0.8392–0.8744) | –       | –       | –      | –      |
| RSF         | Val   | 0.8337                 | 0.0930  | –       | –      | –      |
|             | Test  | 0.8437                 | 0.0948  | –       | –      | –      |
| CLV         | Train | –                      | –       | $3.93   | 0.9991 | 1.33%  |
|             | Val   | –                      | –       | $5.17   | 0.9982 | 1.65%  |
|             | Test  | –                      | –       | $5.17   | 0.9982 | 1.71%  |

### Reproducibility & Robustness
- Fixed `RANDOM_STATE = 42`
- Strict dependency checks (Phase 1 required, survival models before CLV)
- Comprehensive logging and error handling
- Cached survival probabilities for speed and consistency
- Bootstrap confidence intervals for C-index

### Directory Structure Created
```
models/phase2/                  ← trained models & metrics
artifacts/phase2/               ← CLV preprocessor, predictions, segments
plots/phase2/                   ← KM curves, hazard ratios, importance, calibration, PDP, etc.
```

Run `build.py` once to complete Phase 2. The resulting models and insights are ready for reporting, deployment, or further business analysis.
```