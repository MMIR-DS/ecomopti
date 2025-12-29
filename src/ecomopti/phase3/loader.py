"""
Centralized loader: Handles one-hot encoded data directly.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from ecomopti.phase3.config import PROCESSED_DIR, UPLIFT_EXCLUDE_COLS

def load_processed_split(split: str) -> dict:
    """
    Load uplift data for a given split.
    For one-hot data: returns numpy array directly.
    """
    df = pd.read_csv(PROCESSED_DIR / f"{split}_uplift.csv")
    
    # Extract feature columns
    feature_cols = [c for c in df.columns if c not in UPLIFT_EXCLUDE_COLS]
    
    # Convert to numpy array directly
    X = df[feature_cols].values
    
    # Ensure clv is an array
    clv_series = df.get("clv", pd.Series(np.ones(len(df))))
    
    return {
        "X": X,
        "A": df["A"].values,
        "Y": df["Y"].values,
        "propensity": df["estimated_propensity"].values,
        "meta": {
            "true_uplift": df["true_uplift"].values,
            "clv": clv_series.values,  # Ensure it's an array
        }
    }