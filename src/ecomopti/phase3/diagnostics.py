"""
Phase 3 Diagnostics: Fixed propensity validation and DR pseudo-outcomes.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from pathlib import Path
import logging
from typing import Any

from ecomopti.phase3.loader import load_processed_split
from ecomopti.phase3.metrics_uplift import compute_auoc
from ecomopti.phase3.config import MODELS_DIR

logger = logging.getLogger("phase3.diagnostics")

def check_propensity_overlap(propensity: np.ndarray) -> dict:
    """Check for positivity violations in propensity scores."""
    # FIXED: Correct validation logic
    return {
        "min": float(propensity.min()),
        "max": float(propensity.max()),
        "pct_below_05": float((propensity < 0.05).mean() * 100),
        "pct_above_95": float((propensity > 0.95).mean() * 100),
        "is_valid": propensity.min() >= 0.05 and propensity.max() <= 0.95,  # FIXED
    }

def validate_generated_effects(df: pd.DataFrame, split_name: str) -> float:
    """Validate semi-synthetic data generation quality."""
    logger.info(f"\n=== {split_name} DGP VALIDATION ===")
    
    logger.info(f"True uplift range: {df['true_uplift'].min():.4f} to {df['true_uplift'].max():.4f}")
    logger.info(f"Mean true uplift: {df['true_uplift'].mean():.4f}")
    logger.info(f"Std true uplift: {df['true_uplift'].std():.4f}")
    
    # Signal-to-noise ratio
    cohens_d = df['true_uplift'].mean() / df['true_uplift'].std()
    logger.info(f"Cohen's d (effect size): {cohens_d:.4f}")
    
    if abs(cohens_d) < 0.2:
        logger.warning("⚠️ Small effect size - consider increasing treatment effect magnitude")
    
    return df['true_uplift'].std()

def validate_model_signal(model: Any, X: np.ndarray, Y: np.ndarray, T: np.ndarray, 
                         split_name: str, model_name: str) -> np.ndarray:
    """Check if model captures uplift signal."""
    logger.info(f"\n=== {model_name} {split_name} SIGNAL VALIDATION ===")
    
    tau_hat = model.effect(X)
    logger.info(f"Predicted std: {tau_hat.std():.4f}")
    
    # ✅ DEBUG: Check S-Learner treatment interaction strength
    if model_name == "s_learner":
        interaction_strength = np.corrcoef(tau_hat, T)[0, 1]
        logger.info(f"S-Learner treatment interaction: {interaction_strength:.3f}")
        if abs(interaction_strength) < 0.05:
            logger.warning("⚠️ S-Learner shows weak treatment interaction - max_depth may be too low")
    
    # For DR-Learner: Load actual propensity scores
    elif hasattr(model, 'model_final'):
        data = load_processed_split(split_name.lower())
        p = data.get("propensity", np.ones(len(Y)) * 0.5)
        
        pseudo_true = np.zeros(len(Y))
        pseudo_true[T == 1] = (Y[T == 1] - Y.mean()) / p[T == 1]
        pseudo_true[T == 0] = -(Y[T == 0] - Y.mean()) / (1 - p[T == 0])
        
        corr, p_val = spearmanr(tau_hat, pseudo_true)
        logger.info(f"Correlation with pseudo-outcome: {corr:.4f} (p={p_val:.4f})")
        
        if corr < 0.1:
            logger.warning("⚠️ Very weak signal capture")
    
    return tau_hat

def debug_auoc(tau_hat: np.ndarray, true_uplift: np.ndarray) -> tuple[float, float, float]:
    """Debug AUOC with detailed logging."""
    from scipy.stats import spearmanr
    
    # Temporarily increase logging level
    logging.getLogger("phase3.metrics").setLevel(logging.DEBUG)
    
    auoc = compute_auoc(tau_hat, true_uplift)
    corr, p_val = spearmanr(tau_hat, true_uplift)
    
    # Top-20% overlap
    perfect_order = np.argsort(-true_uplift)
    model_order = np.argsort(-tau_hat)
    top_20_idx = set(perfect_order[:len(perfect_order)//5])
    model_top_20 = set(model_order[:len(model_order)//5])
    overlap = len(top_20_idx.intersection(model_top_20))
    overlap_pct = overlap / len(top_20_idx) * 100 if top_20_idx else 0
    
    logger.info(f"AUOC: {auoc:.4f}")
    logger.info(f"Correlation: {corr:.4f} (p={p_val:.4f})")
    logger.info(f"Top-20% overlap: {overlap_pct:.1f}%")
    logger.info(f"Mean true uplift: {true_uplift.mean():.4f}")
    
    # Reset logging level
    logging.getLogger("phase3.metrics").setLevel(logging.INFO)
    
    return corr, overlap_pct, auoc

def run_phase3_diagnostics():
    """Master diagnostic function for Phase 3 pipeline."""
    logger.info("🔍 RUNNING PHASE 3 MASTER DIAGNOSTICS")
    
    # Load data
    train_data = load_processed_split("train")
    val_data = load_processed_split("val")
    test_data = load_processed_split("test")
    
    # Load predictions
    predictions = {}
    for model in ["dr_learner", "s_learner"]:
        for split in ["train", "val", "test"]:
            path = MODELS_DIR / f"{model}_pred_{split}.npy"
            predictions[f"{model}_{split}"] = np.load(path)
    
    # DGP Quality Check
    logger.info("\n=== 1. DATA GENERATION QUALITY ===")
    for split in ["train", "val", "test"]:
        df = pd.read_csv(f"artifacts/phase3/processed/{split}_uplift.csv")
        validate_generated_effects(df, split.upper())
    
    # Model Performance
    logger.info("\n=== 2. MODEL PERFORMANCE CHECK ===")
    for model_name in ["dr_learner", "s_learner"]:
        for split, data in [("TRAIN", train_data), ("VAL", val_data), ("TEST", test_data)]:
            tau_hat = predictions[f"{model_name}_{split.lower()}"]
            corr, _ = spearmanr(tau_hat, data["meta"]["true_uplift"])
            logger.info(f"{model_name} {split}: corr={corr:.3f}")
    
    # AUOC Debug
    logger.info("\n=== 3. AUOC METRIC DEBUG ===")
    debug_auoc(predictions["dr_learner_test"], test_data["meta"]["true_uplift"])
    debug_auoc(predictions["s_learner_test"], test_data["meta"]["true_uplift"])
    
    logger.info("\n✅ MASTER DIAGNOSTICS COMPLETE")

if __name__ == "__main__":
    run_phase3_diagnostics()