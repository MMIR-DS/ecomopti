"""
CLV Model Training: Predicts CLV Score (derived from Cox survival × charges).
IMPORTANT: MonthlyCharges and tenure are EXCLUDED from features to prevent leakage.
"""

import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
# Suppress feature name warnings from sklearn/LightGBM
warnings.filterwarnings(action='ignore', category=UserWarning, message="X does not have valid feature names")

from pathlib import Path
from .config import (
    MODELS_DIR, ARTIFACTS_DIR, HORIZON_MONTHS, RANDOM_STATE,
    RSF_N_ESTIMATORS, PHASE1_ARTIFACTS_DIR
)
from .data_loader import load_processed, get_cached_survival
from .utils import safe_load_model

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

logger = logging.getLogger("phase2.train_clv")

def main():
    """Train CLV model with leakage-free features and save predictions."""
    logger.info("=" * 60)
    logger.info("TRAINING CLV MODEL")
    logger.info("=" * 60)
    
    # Load fitted Phase 1 preprocessor (only used for reference)
    phase1_pre = joblib.load(PHASE1_ARTIFACTS_DIR / "preprocessor.pkl")
    
    # Load data
    train = load_processed("train")
    test = load_processed("test")
    val = load_processed("val")
    
    # Compute survival probabilities (cached)
    train["survival_prob"] = get_cached_survival("train", HORIZON_MONTHS)

    # Compute CLV targets
    train["base_clv"] = train["survival_prob"] * train["MonthlyCharges"] * HORIZON_MONTHS
    
    # === DROP LEAKAGE COLUMNS DIRECTLY (Surgical Strike) ===
    # These columns are used for target computation but must NOT be features
    drop_leakage_cols = ["E", "T", "customerID", "survival_prob", "MonthlyCharges","tenure",'base_clv']
    
    # Drop from training data
    X_train_raw = train.drop(columns=drop_leakage_cols)
    y_train = train["base_clv"].values
    
    # Get remaining feature names
    selected_features = X_train_raw.columns.tolist()

    logger.info(f"CLV will use {len(selected_features)} features (dropped {len(drop_leakage_cols)} leakage columns)")
    
    # Create ColumnTransformer that passes through all remaining features
    feature_selector = ColumnTransformer(
        [("selector", "passthrough", selected_features)],
        remainder="drop"
    )
    
    # Fit on training data only
    feature_selector.fit(X_train_raw)
    
    # Save the fitted selector
    joblib.dump(feature_selector, ARTIFACTS_DIR / "preprocessor_clv.pkl")
    
    # Transform training data
    X_train = feature_selector.transform(X_train_raw)
    X_train_df = pd.DataFrame(X_train, columns=selected_features, index=train.index)
    
    # Train model (LightGBM or fallback)
    if HAS_LGB:
        model = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    else:
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(
            n_estimators=RSF_N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            max_depth=10
        )
    
    logger.info(f"Training {model.__class__.__name__}...")
    model.fit(X_train_df, y_train)
    joblib.dump(model, MODELS_DIR / "clv_model.pkl", compress=3)
    logger.info(f"Saved CLV model: {MODELS_DIR / 'clv_model.pkl'}")
    
    # === GENERATE PREDICTIONS FOR ALL SPLITS ===
    for split in ["train", "val", "test"]:
        df = load_processed(split)
        df["survival_prob"] = get_cached_survival(split, HORIZON_MONTHS)
        df["base_clv"] = df["survival_prob"] * df["MonthlyCharges"] * HORIZON_MONTHS
        
        # Drop leakage columns
        X_raw = df.drop(columns=drop_leakage_cols + ["base_clv"])
        X = feature_selector.transform(X_raw)
        X_df = pd.DataFrame(X, columns=selected_features, index=df.index)
        
        pred = model.predict(X_df)
        
        pd.DataFrame({
            "customerID": df["customerID"],
            "actual_clv": df["base_clv"],
            "predicted_clv": pred
        }).to_csv(ARTIFACTS_DIR / f"clv_{split}_predictions.csv", index=False)
        logger.info(f"Saved CLV predictions for {split}: {len(pred)} rows")
    
    logger.info("=" * 60)
    logger.info("✅ CLV TRAINING COMPLETE")
    logger.info("=" * 60)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()