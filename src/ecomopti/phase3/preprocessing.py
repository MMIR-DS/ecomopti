"""Preprocessing for Phase 3 (bypass for one-hot data)."""

from ecomopti.phase3.config import UPLIFT_EXCLUDE_COLS

def fit_and_save_preprocessor(df, exclude_cols):
    """
    For one-hot data: just extract feature columns, no fitting.
    Returns: (None, feature_cols) since no preprocessing needed.
    """
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    return None, feature_cols