# src/ecomopti/phase3/config.py
"""
Phase 3 Configuration: Pydantic-based settings management.

NOTE: Uses pydantic-settings for BaseSettings (separate package from pydantic v2+)
"""

from pathlib import Path
import numpy as np
from pydantic import Field, BaseModel 
from pydantic_settings import BaseSettings  # Required for Python 3.9+ with pydantic v2


ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS_DIR = ROOT / "artifacts" / "phase3"
PROCESSED_DIR = ARTIFACTS_DIR / "processed"
MODELS_DIR = ROOT / "models" / "phase3"
PLOTS_DIR = ROOT / "plots" / "phase3"

# Causal columns
TARGET_COL = "Y"
TREATMENT_COL = "A"
TRUE_UPLIFT_COL = "true_uplift"

# Business constants
class BusinessConfig:
    TREATMENT_COST = 5.0            # Cost per targeted customer
    TARGET_FRACTION = 0.2           # Top-20% targeting
    RANDOM_BASELINE_RUNS = 20       # Statistical power for revenue simulation
    MIN_SEGMENT_SIZE = 5            # For stratified analysis

# Data generation parameters (calibrated to industry benchmarks)
TREATMENT_RATE = 0.35               # 35% of customers receive treatment
NOISE_STD = 0.2                     # Realistic noise in treatment assignment
RANDOM_STATE = 42
RNG = np.random.default_rng(RANDOM_STATE)

# XGBoost hyperparameter search spaces (Optuna)
# NOTE: These ranges tuned for Telco dataset—adjust for other domains
PROPENSITY_RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "min_samples_leaf": 30,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

OUTCOME_RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 8,
    "min_samples_leaf": 20,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# S-Learner needs more capacity
S_LEARNER_RF_PARAMS = {
    "n_estimators": 400,
    "max_depth": 8,
    "min_samples_leaf": 20,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# DR-Learner final stage
DR_LEARNER_RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "min_samples_leaf": 25,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# Ensemble configuration (weights empirically validated)
ENSEMBLE_CONFIG = {
    "models": ["s_learner", "dr_learner"],
    "weights": [0.3, 0.7],
    "method": "weighted_average",
}

# Columns to exclude from feature set (prevent leakage)
UPLIFT_EXCLUDE_COLS = {
    "customerID", "A", "Y", "true_uplift", "predicted_clv",
    "estimated_propensity","clv",
}

# Production models for evaluation
UPLIFT_MODELS = ["s_learner", "dr_learner", "ensemble"]
PRODUCTION_MODELS = ["s_learner", "dr_learner", "ensemble"]

def ensure_dirs():
    for d in [ARTIFACTS_DIR, PROCESSED_DIR, MODELS_DIR, PLOTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)