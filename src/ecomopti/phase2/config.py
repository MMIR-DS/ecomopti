# src/ecomopti/phase2/config.py
"""
Phase 2 Configuration: Centralized paths and hyperparameters.

Directory Structure:
- data/splits/: Processed train/val/test.csv from Phase 1
- models/phase2/: Serialized survival and CLV models
- artifacts/phase2/: Predictions, CLV preprocessor, customer segments
- plots/phase2/: All visualizations (KM curves, calibration, PDP)

Key Parameters:
- HORIZON_MONTHS: 6-month CLV horizon (aligned with business cycle)
- PENALIZER: Cox regularization (0.1 = moderate shrinkage)
- BOOTSTRAP_N: 200 samples for stable C-index CI
"""

from pathlib import Path
import pandas as pd

# Phase 2 paths
ROOT = Path(__file__).resolve().parents[3]
PHASE1_ARTIFACTS_DIR = ROOT / "artifacts" / "phase1"  # Required Phase 1 dependency
SPLITS_DIR = ROOT / "data" / "splits"                 # Processed splits from Phase 1
MODELS_DIR = ROOT / "models" / "phase2"
ARTIFACTS_DIR = ROOT / "artifacts" / "phase2"
PLOTS_DIR = ROOT / "plots" / "phase2"

# === Hyperparameters  (tuned for Telco dataset)  ===

RANDOM_STATE = 42
PENALIZER = 0.1          # CoxPH regularization: higher = more conservative
HORIZON_MONTHS = 6       # Matches typical quarterly business review cycle
BOOTSTRAP_N = 200        # Sufficient for stable 95% CI (law of large numbers)
RSF_N_ESTIMATORS = 100   # Balances performance vs. training time

# === Performance & Behavior Flags ===
COMPUTE_PERMUTATION_IMPORTANCE = True  # ~1-2 min overhead; set False for rapid iteration

# === Visualization Constants ===
KM_DECILES = 10                    # 10 risk groups for Kaplan-Meier curves
CALIBRATION_BINS = 10              # 10 bins for calibration assessment
PDP_TOP_FEATURES = [
    "Contract_Two year",           # High-impact categorical (long-term commitment)
    "PaymentMethod_Credit card",   # Payment method effect
    "SeniorCitizen",               # Key demographic
    "TechSupport_Yes"              # Service feature affecting retention
]
# === Ensure directories exist ===
def ensure_dirs():
    for p in [MODELS_DIR, ARTIFACTS_DIR, PLOTS_DIR,]:
        p.mkdir(parents=True, exist_ok=True)

# Also validate processed splits exist
def validate_phase1():
    """Ensure Phase 1 preprocessor exists."""
    if not (PHASE1_ARTIFACTS_DIR / "preprocessor.pkl").exists():
        raise FileNotFoundError(
            f"Phase 1 preprocessor not found at {PHASE1_ARTIFACTS_DIR}. Run Phase 1 first!"
        )
    
