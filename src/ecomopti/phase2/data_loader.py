"""
Data Loader: Loads processed splits from Phase 1 with memoization and survival caching.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from functools import lru_cache
from .config import SPLITS_DIR, MODELS_DIR, HORIZON_MONTHS
from .utils import safe_load_model

# === CACHE ===
_PROCESSED_CACHE = {}
logger = logging.getLogger("phase2.data_loader")

def load_processed(split: str) -> pd.DataFrame:
    """Load processed split from Phase 1 with caching."""
    if split not in _PROCESSED_CACHE:
        path = SPLITS_DIR / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Processed split not found: {path}")
        _PROCESSED_CACHE[split] = pd.read_csv(path)
        logger.info(f"Loaded {split} into cache: {len(_PROCESSED_CACHE[split])} rows")
    
    return _PROCESSED_CACHE[split].copy()

def load_train_processed() -> pd.DataFrame:
    """Convenience function for training data."""
    return load_processed("train")

@lru_cache(maxsize=3)
def get_cached_survival(split: str, horizon: int) -> np.ndarray:
    """
    Cache SURVIVAL MODEL predictions from Cox for CLV targets.
    """
    logger.info(f"Computing survival probabilities for {split} at {horizon} months...")
    
    df = load_processed(split)
    cox = safe_load_model(MODELS_DIR / "cox_model.pkl", "Cox")
    
    X = df.drop(columns=["E", "T", "customerID","tenure"])
    surv_df = cox.predict_survival_function(X, times=[horizon])
    surv_probs = surv_df.iloc[0].values
    surv_probs = np.clip(surv_probs, 0.01, 0.99)
    
    logger.info(f"  Mean survival probability: {surv_probs.mean():.3f}")
    return surv_probs