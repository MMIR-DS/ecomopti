# src/ecomopti/phase1/pipeline.py

"""
Phase 1 Pipeline: Orchestrates preprocessing end-to-end with robust error handling.

DATA FLOW:
raw CSV → clean_data() → validate_cleaned() → stratified split → 
fit preprocessor on train → transform all splits → validate_processed() → 
save CSVs + artifacts + manifest
"""

import logging
import joblib
import pandas as pd
import shutil
import hashlib
import json
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from .config import RAW_DIR, SPLITS_DIR, ARTIFACTS_DIR, RANDOM_STATE, METADATA_COLS, ensure_dirs
from .preprocess import clean_data, get_feature_preprocessor, transform_split
from .schemas import validate_cleaned, validate_processed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("phase1.pipeline")

def run_pipeline(raw_filename: str = "telco.csv"):
    """Execute complete Phase 1 pipeline."""
    # P0 #3: Sanitize filename to prevent directory traversal
    from pathlib import PurePath
    if PurePath(raw_filename).name != raw_filename:
        raise ValueError(f"Invalid filename (path traversal detected): {raw_filename}")
    
    logger.info("=" * 60)
    logger.info("PHASE 1 PIPELINE STARTED")
    logger.info("=" * 60)
    
    # P1 #4: Initialize manifest for data versioning
    manifest = {
        "run_timestamp": datetime.utcnow().isoformat(),
        "raw_filename": raw_filename,
        "random_state": RANDOM_STATE
    }
    
    # P0 #2: Wrap pipeline in try/except for robust error handling
    try:
        ensure_dirs()

        # 1. Load raw data
        raw_path = RAW_DIR / raw_filename
        logger.info(f"Loading raw data: {raw_path}")
        
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw data not found: {raw_path}")
        
        # # Calculate data hash for versioning and lineage tracking
        with open(raw_path, 'rb') as f:
            manifest["raw_data_hash"] = hashlib.md5(f.read()).hexdigest()
            
        df_raw = pd.read_csv(raw_path)

         # 2. Clean data (drops TotalCharges, creates E/T targets)
        logger.info("Cleaning data (converting types, creating E/T targets)")
        df = clean_data(df_raw)
        
        # 3. Validate cleaned data before splitting
        validate_cleaned(df)
        
        # 4. Stratified split on E to preserve churn rate across splits
        logger.info("Splitting data (70/15/15, stratified on E)")
        train, temp = train_test_split(df, test_size=0.3, random_state=RANDOM_STATE, stratify=df["E"])
        val, test = train_test_split(temp, test_size=0.5, random_state=RANDOM_STATE, stratify=temp["E"])
        
        logger.info(f"Train: {len(train)} rows | Val: {len(val)} rows | Test: {len(test)} rows")
        logger.info(f"Churn rate: {train['E'].mean():.2%} (train), {val['E'].mean():.2%} (val), {test['E'].mean():.2%} (test)")
        
        # 5. Fit preprocessing on TRAIN only to prevent data leakage
        logger.info("Fitting feature preprocessor on training data...")
        preprocessor = get_feature_preprocessor()
        preprocessor.fit(train.drop(columns=METADATA_COLS))
        
        # 6. Transform all splits consistently using same preprocessor
        for split_name, df_split in [("train", train), ("val", val), ("test", test)]:
            logger.info(f"Processing {split_name}...")
            
            processed = transform_split(df_split, preprocessor)
            validate_processed(processed)
            
            out_path = SPLITS_DIR / f"{split_name}.csv"
            processed.to_csv(out_path, index=False)
            
            logger.info(f"  Saved: {out_path} ({processed.shape[1]} columns)")
        
        # 7. Save preprocessing artifacts for downstream phases
        logger.info("Saving preprocessing artifacts...")
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(preprocessor, ARTIFACTS_DIR / "preprocessor.pkl")
        joblib.dump(preprocessor.named_transformers_["cat"], ARTIFACTS_DIR / "encoder.pkl")
        joblib.dump(preprocessor.named_transformers_["num"], ARTIFACTS_DIR / "scaler.pkl")
        
        # P1 #4: Save manifest with lineage metadata for reproducibility
        manifest.update({
            "train_rows": len(train),
            "val_rows": len(val),
            "test_rows": len(test),
            "churn_rate_train": float(train['E'].mean()),
            "churn_rate_val": float(val['E'].mean()),
            "churn_rate_test": float(test['E'].mean()),
            "feature_count": len(preprocessor.get_feature_names_out())
        })
        
        manifest_path = ARTIFACTS_DIR / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Manifest saved: {manifest_path}")
        
        logger.info("=" * 60)
        logger.info("✅ PHASE 1 COMPLETE")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        # Cleanup partial artifacts on failure
        if ARTIFACTS_DIR.exists():
            shutil.rmtree(ARTIFACTS_DIR, ignore_errors=True)
            logger.info(f"Cleaned up partial artifacts at {ARTIFACTS_DIR}")
        raise

if __name__ == "__main__":
    run_pipeline()