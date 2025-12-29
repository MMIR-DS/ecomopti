"""
Uplift Metrics: FIXED AUOC calculation for negative mean uplift.
"""

import numpy as np
import logging
from typing import Optional

logger = logging.getLogger("phase3.metrics")

__all__ = ["compute_auuc", "compute_auoc", "compute_qini"]

def compute_auuc(
    *,
    tau_hat: np.ndarray,
    true_uplift: np.ndarray,
    treatment: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    propensity: Optional[np.ndarray] = None,
) -> float:
    """
    Compute Area Under Uplift Curve - FIXED for semi-synthetic data.
    Handles negative mean uplift correctly.
    """
    if np.max(np.abs(true_uplift)) < 1e-8:
        logger.warning("All true uplift values are zero - returning 0.0")
        return 0.0
    
    # Rank predictions high to low
    order = np.argsort(-tau_hat)
    sorted_true = true_uplift[order]
    
    # Cumulative uplift curve
    cum_uplift = np.cumsum(sorted_true)
    
    # X-axis: fraction of population
    n = len(cum_uplift)
    x = np.arange(1, n + 1) / n
    
    # Area under curve (trapezoidal integration)
    area = np.trapz(cum_uplift, x)
    
    # Perfect ranking area (all positive uplift first, then negative)
    perfect_order = np.argsort(-true_uplift)
    perfect_cum = np.cumsum(true_uplift[perfect_order])
    perfect_area = np.trapz(perfect_cum, x)
    
    # Random baseline area (straight line from 0 to total uplift)
    total_uplift = np.sum(true_uplift)
    random_area = total_uplift * 0.5  # Area of triangle: base=1, height=total_uplift
    
    # Debug info
    logger.debug(f"AUOC Debug: area={area:.4f}, perfect={perfect_area:.4f}, random={random_area:.4f}")
    
    # Handle edge cases
    denom = perfect_area - random_area
    if abs(denom) < 1e-8:
        logger.warning("Denominator near zero in AUOC normalization - returning 0.0")
        return 0.0
    
    # ✅ FIXED: Proper normalization that works for negative mean
    auoc = (area - random_area) / denom
    
    return float(np.clip(auoc, 0.0, 1.0))


def compute_auoc(tau_hat: np.ndarray, true_uplift: np.ndarray) -> float:
    """Convenience wrapper: AUOC = AUUC."""
    return compute_auuc(tau_hat=tau_hat, true_uplift=true_uplift)


def compute_qini(tau_hat: np.ndarray, true_uplift: np.ndarray) -> float:
    """Compute Qini coefficient (same as AUOC for our purposes)."""
    return compute_auuc(tau_hat=tau_hat, true_uplift=true_uplift)