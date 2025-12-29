"""
Evaluation: AUOC metric with revenue impact simulation.
"""

import logging
import json
import numpy as np
from pathlib import Path
from typing import Dict
from ecomopti.phase3.config import (
    ARTIFACTS_DIR, MODELS_DIR, PRODUCTION_MODELS, 
    BusinessConfig, ENSEMBLE_CONFIG
)
from ecomopti.phase3.metrics_uplift import compute_auoc
from ecomopti.phase3.loader import load_processed_split  # ✅ ADDED: SplitData import
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("phase3.eval_uplift")

def calculate_revenue_impact() -> Dict[str, dict]:
    """Simulate revenue impact with detailed reporting."""
    logger.info("\n💰 REVENUE IMPACT SIMULATION (Targeting Top 20%)")
    
    data = load_processed_split("test")
    clv = data["meta"]["clv"]
    true_uplift = data["meta"]["true_uplift"]
    
    results = {}
    
    for model_name in PRODUCTION_MODELS:
        pred = np.load(MODELS_DIR / f"{model_name}_pred_test.npy")
        
        # Target top fraction
        k = int(len(pred) * BusinessConfig.TARGET_FRACTION)
        top_k_idx = np.argsort(-pred)[:k]
        
        # Revenue calculation
        targeted_uplift = true_uplift[top_k_idx]
        revenue = (clv[top_k_idx] * targeted_uplift).sum() - k * BusinessConfig.TREATMENT_COST
        
        # Random baseline
        random_revenues = []
        for _ in range(BusinessConfig.RANDOM_BASELINE_RUNS):
            random_idx = np.random.choice(len(pred), k, replace=False)
            random_uplift = true_uplift[random_idx]
            random_revenue = (clv[random_idx] * random_uplift).sum() - k * BusinessConfig.TREATMENT_COST
            random_revenues.append(random_revenue)
        
        random_revenue = np.mean(random_revenues)
        random_std = np.std(random_revenues)
        
        # Oracle
        oracle_idx = np.argsort(-true_uplift)[:k]
        oracle_uplift = true_uplift[oracle_idx]
        oracle_revenue = (clv[oracle_idx] * oracle_uplift).sum() - k * BusinessConfig.TREATMENT_COST
        
        # Metrics
        lift_vs_random = ((revenue - random_revenue) / abs(random_revenue) * 100) if random_revenue != 0 else 0
        z_score = (revenue - random_revenue) / random_std if random_std > 0 else 0
        
        # For negative revenues, "closer to zero" is better
        if oracle_revenue < 0:
            efficiency = (oracle_revenue / revenue * 100) if revenue != 0 else 0
        else:
            efficiency = (revenue / oracle_revenue * 100) if oracle_revenue != 0 else 0
        
        logger.info(f"\n📊 {model_name.upper()}:")
        logger.info(f"   Targeted Revenue: ${revenue:,.0f}")
        logger.info(f"   Random Revenue:   ${random_revenue:,.0f} ± ${random_std:,.0f}")
        logger.info(f"   Oracle Revenue:   ${oracle_revenue:,.0f}")
        logger.info(f"   Lift vs Random:   {lift_vs_random:+.1f}% ({z_score:+.1f}σ)")
        logger.info(f"   Efficiency:       {efficiency:.1f}%")
        
        results[model_name] = {
            "targeted_revenue": float(revenue),
            "random_revenue": float(random_revenue),
            "random_std": float(random_std),
            "oracle_revenue": float(oracle_revenue),
            "lift_vs_random_pct": float(lift_vs_random),
            "efficiency_pct": float(efficiency),
            "z_score": float(z_score)
        }
    
    return results


def evaluate_model(model_name: str, split: str, data: dict) -> Dict[str, float]:
    """Evaluate a single model on a split."""
    tau_hat = np.load(MODELS_DIR / f"{model_name}_pred_{split}.npy")
    
    # Revenue-weighted AUOC
    weighted_uplift = data["meta"]["true_uplift"] * data["meta"]["clv"]
    auoc = compute_auoc(tau_hat=tau_hat, true_uplift=weighted_uplift)
    
    # Correlation
    corr, _ = spearmanr(tau_hat, data["meta"]["true_uplift"])
    
    return {"auoc": auoc, "correlation": corr}


def main():
    logger.info("Evaluating production models...")
    results = {}
    
    for split in ["val", "test"]:
        data = load_processed_split(split)
        
        for model_name in PRODUCTION_MODELS:
            metrics = evaluate_model(model_name, split, data)
            results[f"{model_name}_{split}"] = metrics["auoc"]
            logger.info(f"{model_name} {split}: AUOC={metrics['auoc']:.4f}, corr={metrics['correlation']:.3f}")
    
    # Save metrics
    (ARTIFACTS_DIR / "uplift_metrics.json").write_text(json.dumps(results, indent=2))
    logger.info("✅ Evaluation complete")
    
    # Calculate revenue impact
    revenue_results = calculate_revenue_impact()
    (ARTIFACTS_DIR / "revenue_impact.json").write_text(json.dumps(revenue_results, indent=2))
    logger.info("✅ Revenue impact simulation saved")

if __name__ == "__main__":
    main()