"""
Survival Metrics: Evaluation functions for Cox and RSF models.
"""

import logging
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

def compute_brier_at_horizon(cox, df, horizon):
    """Compute Brier score at time horizon."""
    X = df.drop(columns=["E", "T", "customerID"], errors="ignore")
    try:
        surv = cox.predict_survival_function(X, times=[horizon])
        preds = 1.0 - surv.loc[horizon].values
    except Exception as e:
        logger.warning(f"Brier computation failed: {e}")
        preds = np.zeros(len(df))
    
    obs = ((df["T"] <= horizon) & (df["E"] == 1)).astype(int).values
    brier = float(np.mean((obs - preds) ** 2))
    return brier, preds, obs

def calibration_table(preds, obs, n_bins=10):
    """Create calibration table."""
    df = pd.DataFrame({"pred": preds, "obs": obs})
    df["bin"] = pd.qcut(df["pred"].rank(method="first"), q=n_bins, labels=False, duplicates="drop")
    return df.groupby("bin").agg(
        pred_mean=("pred", "mean"),
        obs_rate=("obs", "mean"),
        n=("obs", "size")
    ).reset_index()

def bootstrap_c_index(model, df, n_bootstrap=200, random_state=42) -> Tuple[float, Tuple[float, float]]:
    """Bootstrap confidence interval for C-index."""
    rng = np.random.default_rng(random_state)
    n, c_inds = len(df), []
    
    logger.info(f"Bootstrap C-index: {n_bootstrap} samples")
    for i in range(n_bootstrap):
        if i % 50 == 0:
            logger.info(f"  {i}/{n_bootstrap}...")
        
        samp_idx = rng.choice(n, size=n, replace=True)
        samp = df.iloc[samp_idx].reset_index(drop=True)
        X = samp.drop(columns=["E", "T", "customerID"], errors="ignore")
        
        try:
            if hasattr(model, 'predict_partial_hazard'):
                pred = model.predict_partial_hazard(X).squeeze()
            else:
                pred = model.predict(X)
            c = concordance_index(samp["T"], -pred, samp["E"])
            c_inds.append(c)
        except:
            continue
    
    if not c_inds:
        return 0.0, (0.0, 0.0)
    
    arr = np.array(c_inds)
    return float(arr.mean()), (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))