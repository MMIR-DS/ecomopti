"""
RSF Evaluation: C-index and Integrated Brier Score.
Fixed: Ensure time points count matches predictions exactly.
"""

import logging
import joblib
import numpy as np
import json
from .config import MODELS_DIR, RANDOM_STATE
from .data_loader import load_processed
import warnings
warnings.filterwarnings(
    "ignore", 
    category=DeprecationWarning, 
    module="sksurv.metrics",
    message="`trapz` is deprecated"
)
from sksurv.metrics import concordance_index_censored, integrated_brier_score
from sksurv.util import Surv

logger = logging.getLogger("phase2.eval_rsf")

def evaluate_split(split: str):
    """Evaluate RSF on a single split."""
    logger.info(f"Evaluating RSF on {split}")
    
    df = load_processed(split)
    y = Surv.from_dataframe(event="E", time="T", data=df)
    X = df.drop(columns=["E", "T", "customerID","tenure"])
    
    model = joblib.load(MODELS_DIR / "rsf_model.pkl")
    pred = model.predict(X)
    
    # C-index
    event_field, time_field = y.dtype.names
    c_index = concordance_index_censored(
        y[event_field], y[time_field], pred
    )[0]
    
    # === FIXED: Compute time points and deduplicate ===
    event_times = df.loc[df["E"] == 1, "T"]
    
    # Generate time points
    if len(event_times) >= 15:
        times = np.percentile(event_times, np.linspace(5, 95, 15))
    else:
        times = np.unique(event_times)
    
    # === CRITICAL: Remove duplicates after clipping ===
    max_time = df["T"].max() - 1e-5  # Ensure strictly less than max
    times = np.unique(np.clip(times, 0, max_time))  # DEDUPLICATE & SORT
    
    # Predict survival functions at these EXACT times
    surv_funcs = model.predict_survival_function(X)
    preds = np.vstack([fn(times) for fn in surv_funcs])
    
    # IBS
    df_train = load_processed("train")
    y_train = Surv.from_dataframe(event="E", time="T", data=df_train)
    ibs = integrated_brier_score(y_train, y, preds, times)
    
    logger.info(f"{split.upper()}: C-index={c_index:.4f} | IBS={ibs:.4f} | n_times={len(times)}")
    return c_index, ibs

def main():
    """Evaluate RSF on val and test splits."""
    if not (MODELS_DIR / "rsf_model.pkl").exists():
        raise FileNotFoundError("RSF model not found. Run train_rsf.py first.")
    
    results = {}
    for split in ["val", "test"]:
        c_index, ibs = evaluate_split(split)
        results[split] = {"c_index": float(c_index), "ibs": float(ibs)}
    
    # Save results
    (MODELS_DIR / "rsf_metrics.json").write_text(json.dumps(results, indent=2))
    logger.info(f"Saved RSF metrics: {MODELS_DIR / 'rsf_metrics.json'}")

if __name__ == "__main__":
    main()