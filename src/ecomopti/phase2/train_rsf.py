"""
RSF Training: Fit Random Survival Forest on processed data.
Fixed: Made permutation importance configurable.
"""

import logging
import joblib
import pandas as pd
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sklearn.inspection import permutation_importance
from .config import MODELS_DIR, RANDOM_STATE, RSF_N_ESTIMATORS, COMPUTE_PERMUTATION_IMPORTANCE
from .data_loader import load_train_processed

logger = logging.getLogger("phase2.train_rsf")

def main():
    """Train RSF model."""
    logger.info("Training Random Survival Forest...")
    
    df = load_train_processed()
    y = Surv.from_dataframe(event="E", time="T", data=df)
    X = df.drop(columns=["E", "T", "customerID","tenure"])
    
    model = RandomSurvivalForest(
        n_estimators=RSF_N_ESTIMATORS,
        min_samples_split=10,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    
    model.fit(X, y)
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / "rsf_model.pkl", compress=3)
    
    logger.info(f"Saved RSF model: {MODELS_DIR / 'rsf_model.pkl'}")
    
    # === CONDITIONAL PERMUTATION IMPORTANCE ===
    if COMPUTE_PERMUTATION_IMPORTANCE:
        logger.info("Computing permutation importance (may take 1-2 minutes)...")
        result = permutation_importance(
            model, X, y,
            n_repeats=5,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        
        importance_df = pd.DataFrame({
            "feature": X.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std
        }).sort_values("importance_mean", ascending=False)
        
        importance_df.to_csv(MODELS_DIR / "rsf_feature_importance.csv", index=False)
        logger.info(f"Saved permutation importance: {MODELS_DIR / 'rsf_feature_importance.csv'}")
    else:
        logger.info("Skipping permutation importance (config.COMPUTE_PERMUTATION_IMPORTANCE=False)")
    
    logger.info("✅ RSF training complete")

if __name__ == "__main__":
    main()