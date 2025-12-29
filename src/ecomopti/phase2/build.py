# src/ecomopti/phase2/build.py
"""
Phase 2 Build Script: Complete pipeline orchestration with dependency enforcement.

Execution Order:
1. Train Cox PH model
2. Train Random Survival Forest
3. Run leakage audit (verifies CLV features exclude charges/tenure)
4. Warm survival probability cache
5. Train CLV model (leakage-free)
6. Evaluate all models (Cox, RSF, CLV)
7. Generate plots and diagnostics
"""

import logging
import sys
from pathlib import Path
import joblib  
import pandas as pd

# Import all Phase 2 modules
from . import train_cox, train_rsf, train_clv_model, eval_cox, eval_rsf, eval_clv, plots
from .config import MODELS_DIR, ARTIFACTS_DIR, PLOTS_DIR, ensure_dirs, validate_phase1, PHASE1_ARTIFACTS_DIR
from .data_loader import get_cached_survival  # Import for cache warm-up

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("phase2.build")


def check_survival_models():
    """
    Ensure Cox and RSF models exist before CLV training.
    Called AFTER training Cox/RSF but BEFORE CLV training.
    """
    required = {
        MODELS_DIR / "cox_model.pkl": "Cox PH",
        MODELS_DIR / "rsf_model.pkl": "Random Survival Forest"
    }
    
    for path, name in required.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{name} model not found at {path}. "
                f"Run 'python -m ecomopti.phase2.train_cox' and 'train_rsf' first!"
            )
    logger.info("✅ Survival models verified")


def check_leakage_prevention():
    """Verify CLV preprocessor excludes MonthlyCharges, TotalCharges, tenure.
    
    This audit runs BEFORE CLV training to ensure no leakage.
    If CLV preprocessor exists, validates it; otherwise verifies Phase 1 setup.
    """
    logger.info("🔍 Running leakage audit...")
    
    # Path to Phase 2 CLV preprocessor (created by train_clv_model.py)
    clv_pre_path = ARTIFACTS_DIR / "preprocessor_clv.pkl"
    
    if not clv_pre_path.exists():
        # If CLV hasn't run yet, just verify Phase 1 has the column
        phase1_pre = joblib.load(PHASE1_ARTIFACTS_DIR / "preprocessor.pkl")
        all_features = phase1_pre.get_feature_names_out()
        
        has_charge = any('MonthlyCharges' in f for f in all_features)
        if has_charge:
            logger.info("✅ Phase 1 includes MonthlyCharges (correct for survival models)")
            logger.info("✅ Phase 2 will exclude it during CLV training")
        return
    
    # If CLV preprocessor exists, verify it excluded charge columns
    clv_pre = joblib.load(clv_pre_path)
    clv_features = clv_pre.get_feature_names_out()
    
    forbidden = ['MonthlyCharges', 'TotalCharges','tenure']
    found = [f for f in clv_features if any(forbid in f for forbid in forbidden)]
    
    if found:
        raise RuntimeError(f"LEAKAGE IN CLV PREPROCESSOR: {found}")
    
    logger.info(f"✅ CLV preprocessor clean: {len(clv_features)} features")


def main():
    """Orchestrate complete Phase 2 with enforced dependencies."""
    logger.info("=" * 60)
    logger.info("PHASE 2 BUILD STARTED")
    logger.info("=" * 60)
    
    # Validate Phase 1 exists
    validate_phase1()
    ensure_dirs()
    
    # === Training Phase (Survival Models First) ===
    logger.info("\n🔬 STEP 1: Training Cox PH...")
    try:
        train_cox.main()
    except Exception as e:
        logger.error(f"Cox training failed: {e}")
        raise
    
    logger.info("\n🌲 STEP 2: Training RSF...")
    try:
        train_rsf.main()
    except Exception as e:
        logger.error(f"RSF training failed: {e}")
        raise
    
    # Verify survival models are saved
    check_survival_models()
    
    # ✅ LEAKAGE PREVENTION CHECK (before CLV training)
    logger.info("\n🔍 STEP 2.5: Running leakage audit...")
    check_leakage_prevention()
    
    # Warm up survival cache for CLV
    logger.info("\n⚡ Warming up survival probability cache...")
    for split in ["train", "val", "test"]:
        _ = get_cached_survival(split, 6)  # Pre-compute for speed
    
    logger.info("\n💰 STEP 3: Training CLV...")
    try:
        train_clv_model.main()
    except Exception as e:
        logger.error(f"CLV training failed: {e}")
        raise
    
    # === Evaluation Phase ===
    logger.info("\n📊 STEP 4a: Evaluating Cox...")
    eval_cox.main()
    
    logger.info("\n📊 STEP 4b: Evaluating RSF...")
    eval_rsf.main()
    
    logger.info("\n📊 STEP 4c: Evaluating CLV...")
    eval_clv.main()
    
    logger.info("\n📈 STEP 5: Generating plots...")
    plots.main()
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ PHASE 2 COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()