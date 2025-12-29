"""Phase 4 Configuration: Budget optimization with business constraints."""

from pathlib import Path
from pydantic import BaseModel, Field, validator
import numpy as np
import pandas as pd  # ✅ Required for compatibility checker
from typing import List, Optional
ROOT = Path(__file__).resolve().parents[3]

# Directory Structure
PHASE2_ARTIFACTS_DIR = ROOT / "artifacts" / "phase2"
PHASE3_ARTIFACTS_DIR = ROOT / "artifacts" / "phase3"
PHASE3_MODELS_DIR = ROOT / "models" / "phase3"
ARTIFACTS_DIR = ROOT / "artifacts" / "phase4"
PLOTS_DIR = ROOT / "plots" / "phase4"
REPORTS_DIR = ROOT / "reports" / "phase4"
DATA_DIR = ROOT / "data" / "phase4"
CACHE_DIR = ROOT / "data" / "cache"

# Create directories
for p in [ARTIFACTS_DIR, PLOTS_DIR, REPORTS_DIR, DATA_DIR, CACHE_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Business Parameters
class BusinessConfig(BaseModel):
    """Business rules for optimization."""
    default_treatment_cost: float = Field(5.0, description="Base cost per treatment")
    min_roi_threshold: float = Field(0.0, description="Minimum ROI ratio to be eligible")
    target_budget: float = Field(50000.0, description="Total marketing budget")
    max_customers: Optional[int] = Field(None, description="Maximum customers to treat")
    min_customers_per_segment: int = Field(1, description="Fairness: min per segment")
    clv_horizon_months: int = Field(6, description="Must match Phase 2 config")
    treatment_effect_type: str = Field("relative", description="Phase 3 tau is percentage points (0.01 = 1%)")
    max_budget_utilization: float = Field(0.15, description="Budget sanity check threshold")
    min_uplift_threshold: float = Field(0.005, description="0.5 percentage points minimum")
    
    cost_by_tenure: dict = Field(
        {"new": 10.0,     # Short tenure customers cost more to retain
          "mid": 8.0,     # Medium tenure
         "long": 5.0},    # Loyal customers are cheaper to retain
        description="Cost varies by customer tenure segment"
    )
    
    @validator('treatment_effect_type')
    def validate_effect_type(cls, v):
        if v not in ["relative", "absolute"]:
            raise ValueError("treatment_effect_type must be 'relative' or 'absolute'")
        return v

BUSINESS_CONFIG = BusinessConfig()

# Optimization Parameters
ILP_TIME_LIMIT: int = 60
ILP_MAX_ROWS: int = 50000
GREEDY_BATCH_SIZE: int = 1000

# Budget Sweep Analysis
BUDGET_SWEEP_RANGE = np.linspace(200, 5000, 24).tolist()

# Column Names
COLS = {
    "customer_id": "customerID",
    "tau_hat": "tau_hat",  # Percentage points from Phase 3
    "base_churn": "base_churn_probability",  # Probability (0-1)
    "clv": "clv",
    "cost": "treatment_cost",
    "net_value": "net_value",
    "segment": "tenure_segment",
    "incremental_value": "incremental_value",
    "bang_per_buck": "bang_per_buck",
    "tau_se": "tau_standard_error",
}

# Compatibility checker
def verify_phase_compatibility(split: str, model_hint: str) -> dict:
    """
    Ensure Phase 2 & 3 artifacts are compatible before optimization.
    Checks for existence and format of required files.
    Returns dict with 'errors' and 'warnings' lists.
    """
    errors = []
    warnings = []
    
    # Check Phase 2 CLV exists
    clv_path = PHASE2_ARTIFACTS_DIR / f"clv_{split}_predictions.csv"
    if not clv_path.exists():
        errors.append(f"Phase 2 CLV missing: {clv_path}")
    
    # Check Phase 3 model exists
    npy_path = PHASE3_MODELS_DIR / f"{model_hint}_pred_{split}.npy"
    csv_path = PHASE3_ARTIFACTS_DIR / f"{split}_uplift_pred_{model_hint}.csv"
    if not npy_path.exists() and not csv_path.exists():
        errors.append(f"Phase 3 model predictions missing: {model_hint} for {split}")
    
    # Check for DGP features needed to compute base churn
    uplift_data_path = PHASE3_ARTIFACTS_DIR / "processed" / f"{split}_uplift.csv"
    if uplift_data_path.exists():
        try:
            df_check = pd.read_csv(uplift_data_path, nrows=1)
            required_dgp_cols = ["tenure", "is_monthly_contract", "usage_intensity", "price_elasticity"]
            missing = [col for col in required_dgp_cols if col not in df_check.columns]
            if missing:
                errors.append(f"Phase 3 missing DGP features needed for base churn: {missing}")
        except Exception as e:
            warnings.append(f"Error checking Phase 3 data: {e}")
    else:
        errors.append(f"Phase 3 uplift data not found at {uplift_data_path}")
    
    return {"errors": errors, "warnings": warnings}