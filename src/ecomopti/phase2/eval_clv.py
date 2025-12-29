"""
CLV Evaluation: Compute metrics on test set.
"""

import logging
import joblib
import json
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from .config import MODELS_DIR, ARTIFACTS_DIR

logger = logging.getLogger("phase2.eval_clv")

def main():
    """Evaluate CLV model on test set."""
    logger.info("=" * 60)
    logger.info("EVALUATING CLV MODEL")
    logger.info("=" * 60)
    
    # Load model and preprocessor
    model = joblib.load(MODELS_DIR / "clv_model.pkl")
    pre = joblib.load(ARTIFACTS_DIR / "preprocessor_clv.pkl")
    
    # Load test predictions (created by train_clv_model.py)
    test_pred = pd.read_csv(ARTIFACTS_DIR / "clv_test_predictions.csv")
    train_pred = pd.read_csv(ARTIFACTS_DIR / "clv_train_predictions.csv")
    val_pred = pd.read_csv(ARTIFACTS_DIR / "clv_val_predictions.csv")
    
    # Compute metrics for all splits
    metrics = {}
    for split, df in [("train", train_pred), ("val", val_pred), ("test", test_pred)]:
        mae = mean_absolute_error(df["actual_clv"], df["predicted_clv"])
        r2 = r2_score(df["actual_clv"], df["predicted_clv"])
        mape = mean_absolute_percentage_error(df["actual_clv"], df["predicted_clv"])
        
        metrics[split] = {
            "mae": float(mae),
            "r2": float(r2),
            "mape": float(mape),
            "mean_actual": float(df["actual_clv"].mean()),
            "mean_predicted": float(df["predicted_clv"].mean())
        }
        
        logger.info(f"{split.upper()}: MAE=${mae:.2f} | R²={r2:.4f} | MAPE={mape:.2%}")
    
    # Save metrics
    (MODELS_DIR / "clv_metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info(f"Saved CLV metrics: {MODELS_DIR / 'clv_metrics.json'}")
    
    logger.info("=" * 60)
    logger.info("✅ CLV EVALUATION COMPLETE")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()