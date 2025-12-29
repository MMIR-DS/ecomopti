"""Phase 4 Cost Modeling: Variable costs, risk adjustment, scenario analysis."""

import logging
import numpy as np
import pandas as pd
from typing import Dict
from .config import BUSINESS_CONFIG, COLS

logger = logging.getLogger("phase4.cost_model")

def compute_risk_adjusted_uplift(
    df: pd.DataFrame,
    confidence_level: float = 0.8
) -> pd.DataFrame:
    """Adjust tau_hat for uncertainty using Phase 3 bootstrap standard errors."""
    if COLS["tau_se"] not in df.columns:
        logger.warning("Missing uncertainty estimates, using generic SE")
        df[COLS["tau_se"]] = df[COLS["tau_hat"]] * 0.2
    
    # Conservative estimate (lower bound)
    z_score = np.abs(np.percentile(np.random.normal(0, 1, 10000), (1-confidence_level)*100))
    df["tau_conservative"] = df[COLS["tau_hat"]] - z_score * df[COLS["tau_se"]]
    df["tau_conservative"] = np.maximum(df["tau_conservative"], BUSINESS_CONFIG.min_uplift_threshold)
    
    return df

def compute_scenario_net_value(
    df: pd.DataFrame,
    scenario: str = "base_case"
) -> pd.Series:
    """Compute net value under different business scenarios."""
    incremental_value = df[COLS["incremental_value"]].copy()
    
    if scenario == "base_case":
        pass
    elif scenario == "optimistic":
        incremental_value *= 1.2
    elif scenario == "pessimistic":
        incremental_value *= 0.8
    elif scenario == "best_case":
        # Only high-CLV customers (top 20%)
        high_clv_mask = df[COLS["clv"]] >= df[COLS["clv"]].quantile(0.8)
        incremental_value = incremental_value.where(high_clv_mask, 0)
        # Also boost their uplift by 50%
        incremental_value[high_clv_mask] *= 1.5
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    
    return incremental_value - df[COLS["cost"]]

def calculate_roi_metrics(selected_df: pd.DataFrame) -> dict:
    """Calculate comprehensive ROI metrics for selected customers."""
    if len(selected_df) == 0:
        return {
            "total_customers": 0,
            "total_cost": 0,
            "incremental_clv_impact": 0,
            "net_value": 0,
            "roi_ratio": 0,
            "cost_per_customer": 0,
            "value_per_dollar": 0,
            "segment_distribution": {}
        }
    
    total_cost = selected_df[COLS["cost"]].sum()
    
    # TRUE incremental impact: uplift × CLV
    incremental_impact = (selected_df[COLS["tau_hat"]] * selected_df[COLS["clv"]]).sum()
    
    # Net value = benefit - cost
    net_value = incremental_impact - total_cost
    
    return {
        "total_customers": len(selected_df),
        "total_cost": total_cost,
        "incremental_clv_impact": incremental_impact,
        "net_value": net_value,
        "roi_ratio": net_value / (total_cost + 1e-9),
        "cost_per_customer": total_cost / len(selected_df),
        "value_per_dollar": incremental_impact / (total_cost + 1e-9),
        "segment_distribution": selected_df[COLS["segment"]].value_counts().to_dict() if COLS["segment"] in selected_df.columns else {}
    }