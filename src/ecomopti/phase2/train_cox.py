"""
Cox PH Training: Fit proportional hazards model on processed data.
"""

import logging
import joblib
import pandas as pd
from pathlib import Path
from lifelines import CoxPHFitter
from .config import MODELS_DIR, ARTIFACTS_DIR, PENALIZER, RANDOM_STATE
from .data_loader import load_train_processed

logger = logging.getLogger("phase2.train_cox")

def main():
    """Train Cox model."""
    logger.info("Training Cox PH model...")
    
    train = load_train_processed()
    
    # Separate features from targets
    X = train.drop(columns=["E", "T", "customerID","tenure"])
    df_cox = pd.concat([train[["E", "T"]], X], axis=1)
    
    # Fit Cox model
    cox = CoxPHFitter(penalizer=PENALIZER)
    cox.fit(df_cox, duration_col="T", event_col="E", show_progress=True)
    
    # Save model
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(cox, MODELS_DIR / "cox_model.pkl")
    
    # Save summary
    summary = cox.summary.copy()
    summary["hazard_ratio"] = summary["exp(coef)"]
    summary.to_csv(MODELS_DIR / "cox_summary.csv", index=False)
    
    logger.info(f"Saved Cox model: {MODELS_DIR / 'cox_model.pkl'}")

if __name__ == "__main__":
    main()