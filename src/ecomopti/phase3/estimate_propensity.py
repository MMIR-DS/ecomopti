"""Estimate propensity scores (regularized, calibrated, tuned)."""

import joblib, logging, pandas as pd, numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score
from ecomopti.phase3.config import PROCESSED_DIR, MODELS_DIR, UPLIFT_EXCLUDE_COLS
from ecomopti.phase3.preprocessing import fit_and_save_preprocessor
from ecomopti.phase3.diagnostics import check_propensity_overlap

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("phase3.estimate_propensity")

def main():
    train_df = pd.read_csv(PROCESSED_DIR / "train_uplift.csv")
    pre, feature_cols = fit_and_save_preprocessor(train_df, UPLIFT_EXCLUDE_COLS)
    X_train = train_df[feature_cols].values
    
    # ✅ FIXED: Hyperparameter tuning for C
    logger.info("Tuning propensity model hyperparameter C...")
    best_c, best_score = 0.1, 0.0
    for C in [0.01, 0.1, 1.0, 10.0]:
        model = LogisticRegression(C=C, random_state=42, max_iter=5000, penalty='l2', solver='saga')
        scores = cross_val_score(model, X_train, train_df["A"], cv=3, scoring='roc_auc')
        logger.info(f"  C={C}: AUROC={scores.mean():.3f}")
        if scores.mean() > best_score:
            best_c, best_score = C, scores.mean()
    
    logger.info(f"Best C selected: {best_c} (AUROC: {best_score:.3f})")
    
    # Train final model with best C
    base_model = LogisticRegression(C=best_c, random_state=42, max_iter=5000, penalty='l2', solver='saga')
    propensity_model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
    propensity_model.fit(X_train, train_df["A"])
    joblib.dump(propensity_model, MODELS_DIR / "propensity_model.pkl")
    
    for split in ["train", "val", "test"]:
        df = pd.read_csv(PROCESSED_DIR / f"{split}_uplift.csv")
        X = df[feature_cols].values
        df["estimated_propensity"] = propensity_model.predict_proba(X)[:, 1]
        df["estimated_propensity"] = np.clip(df["estimated_propensity"], 0.05, 0.95)
        
        overlap_stats = check_propensity_overlap(df["estimated_propensity"].values)
        logger.info(f"{split} overlap: {overlap_stats}")
        
        df.to_csv(PROCESSED_DIR / f"{split}_uplift.csv", index=False)
        logger.info(f"{split}: propensity mean={df['estimated_propensity'].mean():.3f}")

if __name__ == "__main__":
    main()