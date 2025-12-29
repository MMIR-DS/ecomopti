"""Phase 4 Loader: Safely load Phase 2/3 outputs with validation and effect conversion."""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from .config import (
    PHASE2_ARTIFACTS_DIR, PHASE3_ARTIFACTS_DIR, PHASE3_MODELS_DIR, 
    COLS, BUSINESS_CONFIG, verify_phase_compatibility
)
from ecomopti.phase1.config import SPLITS_DIR

logger = logging.getLogger("phase4.loader")

class DataLoadingError(Exception):
    pass

def load_phase2_clv(split: str = "val") -> pd.DataFrame:
    """Load CLV predictions from Phase 2 with horizon verification."""
    clv_path = PHASE2_ARTIFACTS_DIR / f"clv_{split}_predictions.csv"
    
    if not clv_path.exists():
        raise DataLoadingError(f"Phase 2 CLV not found at {clv_path}")
    
    clv_df = pd.read_csv(clv_path, usecols=["customerID", "predicted_clv"])
    clv_df = clv_df.rename(columns={"predicted_clv": COLS["clv"]})
    
    # Verify it's numeric
    clv_df[COLS["clv"]] = pd.to_numeric(clv_df[COLS["clv"]], errors="coerce")
    if clv_df[COLS["clv"]].isnull().any():
        raise DataLoadingError("Phase 2 CLV contains non-numeric values")
    
    logger.info(f"✓ Loaded {len(clv_df)} CLV predictions (Phase 2, {BUSINESS_CONFIG.clv_horizon_months}-month horizon)")
    return clv_df

def compute_base_churn_probability(df: pd.DataFrame) -> np.ndarray:
    """
    Compute base churn probability using Phase 3 DGP formula.
    We recalculate instead of loading to avoid data leakage and ensure
    reproducibility. Formula source: generate_semi_synthetic.py (Phase 3)
    Recomputing base churn is CRITICAL for small datasets:
    - Saves memory (don't load entire Phase 3 file)
    - Ensures reproducibility
    - For 1K rows, computation cost is negligible (<1ms)
    Component breakdown:
    - 0.05: Base churn rate
    - 0.02 × (tenure < 12): New customers churn more
    - 0.03 × is_monthly_contract: Monthly contracts are riskier
    - 0.01 × (1 - usage_intensity): Low engagement → higher churn
    - 0.02 × price_elasticity: Price-sensitive customers churn more
    """
    required_cols = ["tenure", "is_monthly_contract", "usage_intensity", "price_elasticity"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required DGP features for base churn calculation: {missing}")
    
    # Vectorized calculation
    tenure_lt_12 = (df["tenure"] < 12).astype(int)
    is_monthly = df["is_monthly_contract"].astype(int)
    
    base_p = (
        0.05
        + 0.02 * tenure_lt_12
        + 0.03 * is_monthly
        + 0.01 * (1 - df["usage_intensity"])
        + 0.02 * df["price_elasticity"]
    )
    
    return np.clip(base_p, 0.01, 0.30)

def load_phase3_uplift_with_base_churn(split: str = "val", model_hint: str = "dr_learner") -> pd.DataFrame:
    """Load uplift predictions AND compute base churn probability on-the-fly."""
    # Get customer IDs from original split (Phase 1)
    raw_split_path = SPLITS_DIR / f"{split}.csv"
    if not raw_split_path.exists():
        raise DataLoadingError(f"Original split data not found: {raw_split_path}")
    
    customers_df = pd.read_csv(raw_split_path, usecols=["customerID"])
    
    # Load predictions (try .npy first, then .csv)
    npy_path = PHASE3_MODELS_DIR / f"{model_hint}_pred_{split}.npy"
    if npy_path.exists():
        uplift_predictions = np.load(npy_path)
        logger.info(f"✓ Loaded {len(uplift_predictions)} predictions from {npy_path}")
    else:
        csv_path = PHASE3_ARTIFACTS_DIR / f"{split}_uplift_pred_{model_hint}.csv"
        if csv_path.exists():
            pred_df = pd.read_csv(csv_path)
            uplift_predictions = pred_df[COLS["tau_hat"]].values
            logger.info(f"✓ Loaded {len(uplift_predictions)} predictions from {csv_path}")
        else:
            raise DataLoadingError(f"No uplift predictions at {npy_path} or {csv_path}")
    
    if len(uplift_predictions) != len(customers_df):
        raise DataLoadingError(
            f"Count mismatch: predictions ({len(uplift_predictions)}) vs customers ({len(customers_df)})"
        )
    
    # Compute base churn probability instead of loading
    uplift_data_path = PHASE3_ARTIFACTS_DIR / "processed" / f"{split}_uplift.csv"
    if not uplift_data_path.exists():
        raise DataLoadingError(f"Phase 3 uplift data not found at {uplift_data_path}")
    
    # Read only the DGP feature columns needed for calculation
    dgp_features_df = pd.read_csv(
        uplift_data_path, 
        usecols=["customerID", "tenure", "is_monthly_contract", "usage_intensity", "price_elasticity"]
    )
    
    # Align with customers_df order
    dgp_features_df = dgp_features_df.set_index("customerID").reindex(customers_df["customerID"]).reset_index()
    
    # Compute base churn probability
    base_churn = compute_base_churn_probability(dgp_features_df)
    
    logger.info(f"✓ Computed base churn probabilities (range: {base_churn.min():.3f} - {base_churn.max():.3f})")
    
    uplift_df = pd.DataFrame({
        COLS["customer_id"]: customers_df["customerID"].values,
        COLS["tau_hat"]: uplift_predictions,
        COLS["base_churn"]: base_churn
    })
    
    return uplift_df

def load_phase3_uplift_uncertainty(split: str = "val", model_hint: str = "dr_learner") -> pd.DataFrame:
    """Load bootstrap standard errors from Phase 3 (if available)."""
    se_path = PHASE3_ARTIFACTS_DIR / f"{model_hint}_se_{split}.npy"
    
    if se_path.exists():
        se_values = np.load(se_path)
        logger.info(f"✓ Loaded {len(se_values)} standard errors from Phase 3 bootstrap")
        return pd.DataFrame({
            COLS["customer_id"]: pd.read_csv(SPLITS_DIR / f"{split}.csv", usecols=["customerID"])["customerID"].values,
            COLS["tau_se"]: se_values
        })
    else:
        logger.warning("Phase 3 uncertainty estimates not found. Using generic 20% SE.")
        customers_df = pd.read_csv(SPLITS_DIR / f"{split}.csv", usecols=["customerID"])
        return pd.DataFrame({
            COLS["customer_id"]: customers_df["customerID"].values,
            COLS["tau_se"]: np.zeros(len(customers_df))
        })

def merge_and_validate(
    clv_df: pd.DataFrame,
    uplift_df: pd.DataFrame,
    uncertainty_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Merge data and validate quality with effect conversion."""
    merged = uplift_df.merge(clv_df, on=COLS["customer_id"], how="inner")
    
    if merged.empty:
        raise DataLoadingError("No overlapping customerIDs between Phase 2 and Phase 3")
    
    # Add default cost column
    merged[COLS["cost"]] = BUSINESS_CONFIG.default_treatment_cost
    
    # ✅ FIXED: No conversion needed - tau_hat is already in percentage points
    if BUSINESS_CONFIG.treatment_effect_type == "relative":
         
        max_allowed = merged[COLS["base_churn"]] 
        merged[COLS["tau_hat"]] = np.clip(
            merged[COLS["tau_hat"]],
            BUSINESS_CONFIG.min_uplift_threshold,
            max_allowed
        )
        logger.info("✓ Validated relative treatment effects (no conversion needed)")
    
    # Filter out "sleeping dogs" (negative uplift)
    positive_uplift_mask = merged[COLS["tau_hat"]] >= BUSINESS_CONFIG.min_uplift_threshold
    merged = merged[positive_uplift_mask].copy()
    
    if merged.empty:
        raise DataLoadingError(
            "No customers with positive uplift after filtering. "
            "Check: (1) tau_hat range, (2) min_uplift_threshold is reasonable"
        )
    logger.info(f"Filtered to {len(merged)} customers with positive uplift")
    
    # Pre-compute incremental value (τ × CLV) - ranking metric
    merged[COLS["incremental_value"]] = merged[COLS["tau_hat"]] * merged[COLS["clv"]]
    merged[COLS["bang_per_buck"]] = merged[COLS["incremental_value"]] / (merged[COLS["cost"]] + 1e-8)
    
    return merged

def load_data_for_optimization(
    split: str = "val",
    model_hint: str = "dr_learner"
) -> pd.DataFrame:
    """Main entry point with full validation chain."""
    logger.info(f"Loading data for split={split}, model={model_hint}")
    
    # Verify compatibility
    compat = verify_phase_compatibility(split, model_hint)
    if compat["errors"]:
        raise DataLoadingError(f"Phase compatibility errors: {compat['errors']}")
    for w in compat["warnings"]:
        logger.warning(w)
    
    clv_df = load_phase2_clv(split)
    uplift_df = load_phase3_uplift_with_base_churn(split, model_hint)
    uncertainty_df = load_phase3_uplift_uncertainty(split, model_hint)
    
    df = merge_and_validate(clv_df, uplift_df, uncertainty_df)
    
    # Create segments
    df[COLS["segment"]] = pd.cut(
        df[COLS["clv"]],
        bins=[0, 200, 400, float("inf")],
        labels=["new", "mid", "long"]
    )
    
    # Assign variable costs
    df[COLS["cost"]] = df[COLS["segment"]].map(BUSINESS_CONFIG.cost_by_tenure)
    df[COLS["cost"]] = df[COLS["cost"]].astype(float).fillna(BUSINESS_CONFIG.default_treatment_cost)
    
    logger.info(f"✓ Loader ready: {len(df)} customers")
    logger.info(f"✓ CLV range: ${df[COLS['clv']].min():.2f} - ${df[COLS['clv']].max():.2f}")
    logger.info(f"✓ Incremental value range: ${df[COLS['incremental_value']].min():.2f} - ${df[COLS['incremental_value']].max():.2f}")
    
    return df