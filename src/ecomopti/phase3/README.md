# Phase 3: Causal Uplift Modeling

**Prerequisites**: Python 3.9+, install dependencies via `pip install -r src/ecomopti/phase3/requirements.txt`

**Run Full Pipeline (Recommended):**  
`python -m ecomopti.phase3.build`

**Alternative (manual):**  
```bash
python -m ecomopti.phase3.generate_semi_synthetic
python -m ecomopti.phase3.estimate_propensity
python -m ecomopti.phase3.train_uplift
python -m ecomopti.phase3.eval_uplift
python -m ecomopti.phase3.plots
```

**Prerequisite:** Phase 2 must be completed first (`data/splits/`, `models/phase2/cox_model.pkl`, `rsf_model.pkl`, `clv_model.pkl` must exist).

**Produced Artifacts:**
- `artifacts/phase3/processed/{train,val,test}_uplift.csv` → semi-synthetic uplift datasets with true_uplift & clv
- `models/phase3/propensity_model.pkl` → calibrated propensity score model
- `models/phase3/s_learner_fitted.pkl`, `dr_learner_fitted.pkl` → trained uplift meta-learners (XGBoost + Optuna)
- `models/phase3/{s_learner,dr_learner,ensemble}_pred_{train,val,test}.npy` → predicted uplift scores
- `artifacts/phase3/uplift_metrics.json` → AUOC & correlation metrics
- `artifacts/phase3/revenue_impact.json` → business simulation (top-20% targeting)
- `artifacts/phase3/feature_importance_*.{csv,json}` → S-Learner & DR-Learner outcome model importances
- `plots/phase3/*.png` → propensity, uplift curves, calibration, policy, lift, feature importance, etc.

**Configuration:** `src/ecomopti/phase3/config.py`

**Tests:** `pytest -q`

---

## Overview

Phase 3 builds production-grade causal uplift models to identify customers most responsive to retention interventions:

1. Generates realistic **semi-synthetic uplift data** using Phase 2 CLV and survival insights
2. Estimates **calibrated propensity scores** (Logistic + isotonic calibration)
3. Trains **S-Learner** and **DR-Learner** with XGBoost (hyper-tuned via Optuna)
4. Combines into a **weighted ensemble** (30% S + 70% DR)
5. Evaluates using **ground-truth AUOC** and **revenue impact simulation**
6. Generates comprehensive diagnostics and interpretability plots

Ground-truth uplift enables rigorous model comparison (rare in observational settings).

---

## Key Design Decisions

### Semi-Synthetic DGP (Data Generating Process)
- Realistic heterogeneous effects: ~80% positive responders, ~20% negative (do-not-disturb)
- Mean uplift ≈ +3%, Cohen’s d ≈ 1.0 (strong but plausible signal)
- Confounding via tenure, charges, contract type

### Model Overview
| Model     | Learner Type       | Base Model | Key Strength                         | Test AUOC | Test Correlation |
|-----------|--------------------|------------|--------------------------------------|-----------|------------------|
| S-Learner | Single model       | XGBoost    | Simple, fast, explicit interactions  | **0.1742**| 0.095            |
| DR-Learner| Doubly robust      | XGBoost    | Better weak-signal detection         | **0.2422**| 0.134            |
| Ensemble  | Weighted average   | –          | Combines strengths (70% DR weight)   | **0.2430**| 0.135            |

### Business Simulation (Top 20% Targeting)
| Model     | Targeted Revenue | Lift vs Random | Efficiency vs Oracle | Z-Score |
|-----------|------------------|----------------|----------------------|---------|
| S-Learner | $1,347           | **+35.8%**     | 35.0%                | +2.8σ   |
| DR-Learner| $1,535           | **+58.6%**     | 39.9%                | +5.0σ   |
| Ensemble  | **$1,525**       | **+64.3%**     | **39.6%**            | +4.6σ   |
| Random    | $992 ± $128      | –              | 25.8%                | –       |
| Oracle    | $3,850           | 100%           | 100%                 | –       |

### Reproducibility & Robustness
- Fixed `RANDOM_STATE = 42`
- Optuna hyperparameter search with 25 trials per component
- Positivity enforcement (propensity clipped [0.05, 0.95])
- Master diagnostics with correlation, AUOC debug, and DGP validation
- Strict Phase 2 dependency checks

### Directory Structure Created
```
artifacts/phase3/
├── processed/                  ← uplift CSVs with true_uplift
├── uplift_metrics.json
├── revenue_impact.json
├── feature_importance_*.{csv,json}
models/phase3/                  ← fitted uplift & propensity models + predictions
plots/phase3/                   ← propensity, uplift curves, calibration, policy, lift, importance
```

Run `build.py` once to complete Phase 3. Results are ready for stakeholder reporting, model selection, or integration into a recommendation engine.