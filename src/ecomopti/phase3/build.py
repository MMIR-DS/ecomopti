# src/ecomopti/phase3/build.py
"""
Phase 3 Build Script: Complete uplift pipeline with ensemble.
Enforces strict dependency order and runs comprehensive diagnostics.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ecomopti.phase3 import generate_semi_synthetic, estimate_propensity, train_uplift, eval_uplift, plots
from ecomopti.phase3.config import ensure_dirs
from ecomopti.phase3.diagnostics import run_phase3_diagnostics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("phase3.build")

def check_phase_dependencies():
    """Ensure Phase 2 artifacts exist."""
    required = [
        Path("data/splits/train.csv"),
        Path("models/phase2/cox_model.pkl"),
        Path("models/phase2/rsf_model.pkl"),
        Path("models/phase2/clv_model.pkl"),
    ]
    missing = [f for f in required if not f.exists()]
    if missing:
        logger.error(f"Missing dependencies: {missing}")
        raise FileNotFoundError(f"Run Phase 2 first. Missing: {missing}")
    logger.info("✅ Phase 2 dependencies verified")

def main():
    """Orchestrate complete Phase 3 pipeline with 5 steps."""
    logger.info("=" * 60)
    logger.info("PHASE 3 UPLIFT PIPELINE STARTED")
    logger.info("=" * 60)
    
    check_phase_dependencies()
    ensure_dirs()

    # STEP 1: Generate semi-synthetic data with ground-truth uplift
    logger.info("\n🔬 STEP 1: Generating Semi-Synthetic Data...")
    generate_semi_synthetic.main()
    
    # STEP 2: Estimate calibrated propensity scores
    logger.info("\n📊 STEP 2: Estimating Propensity Scores...")
    estimate_propensity.main()
    
    # STEP 3: Train uplift models (S-Learner, DR-Learner, Ensemble)
    logger.info("\n🌲 STEP 3: Training Uplift Models...")
    train_uplift.main()
    
    # STEP 4: Run master diagnostics (DGP validation, signal check, AUOC debug)
    logger.info("\n📈 STEP 4: Evaluating Production Models...")
    eval_uplift.main()

    # STEP 5: Run master diagnostics (DGP validation, signal check, AUOC debug)
    logger.info("\n🔍 STEP 5: Running Master Diagnostics...")
    run_phase3_diagnostics()

    # STEP 6: Evaluate production models and revenue impact
    logger.info("\n🎨 STEP 6: Generating Plots...")
    plots.main()
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ PHASE 3 COMPLETE")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()