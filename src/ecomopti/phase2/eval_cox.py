"""
Cox Model Evaluation: Compute metrics using external metrics.py.
"""

import logging
import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path
from lifelines.utils import concordance_index
from .config import MODELS_DIR, HORIZON_MONTHS, BOOTSTRAP_N, RANDOM_STATE
from .metrics import compute_brier_at_horizon, calibration_table, bootstrap_c_index
from .data_loader import load_processed

logger = logging.getLogger("phase2.eval_cox")

def evaluate_split(split: str):
    """Evaluate Cox on a single split."""
    logger.info(f"Evaluating on {split}")
    
    df = load_processed(split)
    X = df.drop(columns=["E", "T", "customerID"])
    
    cox = joblib.load(MODELS_DIR / "cox_model.pkl")
    pred = cox.predict_partial_hazard(X).squeeze()
    
    # C-index
    c_index = concordance_index(df["T"], -pred, df["E"])
    
    # Bootstrap CI
    bmean, (lo, hi) = bootstrap_c_index(cox, df, n_bootstrap=BOOTSTRAP_N)
    
    # Brier & calibration
    brier, preds, obs = compute_brier_at_horizon(cox, df, HORIZON_MONTHS)
    cal_tab = calibration_table(preds, obs)
    cal_tab.to_csv(MODELS_DIR / f"{split}_calibration.csv", index=False)
    
    metrics = {
        "c_index": float(c_index),
        "c_index_bootstrap_mean": bmean,
        "c_index_ci_lower": lo,
        "c_index_ci_upper": hi,
        "brier_at_horizon": float(brier) if brier else None
    }
    
    (MODELS_DIR / f"{split}_metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info(f"{split.upper()}: C-index={c_index:.4f} (CI: {lo:.4f}-{hi:.4f})")

def main():
    np.random.seed(RANDOM_STATE)
    
    if not (MODELS_DIR / "cox_model.pkl").exists():
        raise FileNotFoundError("Cox model not found. Run train_cox.py first.")
    
    for split in ["val", "test"]:
        evaluate_split(split)

if __name__ == "__main__":
    main()