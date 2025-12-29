# src/ecomopti/phase1/preprocess.py
"""
Preprocessing: All transformation logic in ONE place (DRY principle).

CRITICAL DESIGN:
- Numeric features: median imputation (NO scaling to preserve monetary interpretation)
- Categorical features: impute THEN encode to handle missing values
- Metadata columns: passed through unchanged (E, T, customerID)
- Leakage prevention: TotalCharges dropped BEFORE any modeling
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline  # NEW IMPORT
import logging
from .config import METADATA_COLS, RAW_NUMERIC_COLS, RAW_CATEGORICAL_COLS

logger = logging.getLogger("phase1.preprocess")

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw data: create targets, convert types, handle missing values.
    
    Tenure can be missing in raw data due to data entry errors or new customers
    without tenure history. We log a warning and set to 0 for survival analysis.
    """
    df = df.copy()
    
    # Convert TotalCharges to numeric (handles blanks and strings)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
    
    # P0 #1: EXPLICITLY drop TotalCharges to prevent CLV leakage (future information)
    df = df.drop(columns=["TotalCharges"])
    
    # Create survival targets from Churn column
    df["E"] = (df["Churn"].str.lower() == "yes").astype(int)
    
    # P1 #7: Log warning for missing tenure (data quality issue)
    original_missing_tenure = df["tenure"].isna().sum()
    if original_missing_tenure > 0:
        logger.warning(f"Found {original_missing_tenure} missing tenure values, setting to 0")
    
    df["T"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(0).astype(int)
    
    logger.info(f"Cleaned data: {len(df)} rows, TotalCharges dropped, E/T targets created")
    
    return df

def get_feature_preprocessor() -> ColumnTransformer:
    """
    Define preprocessing pipeline for FEATURES ONLY (not metadata).
    
    Returns:
        Unfitted ColumnTransformer ready for fitting on training data.
        Must be fitted BEFORE transforming any splits.
    """
    # CRITICAL: Only impute numeric features (never scale monetary values like MonthlyCharges)
    numeric_transformer = SimpleImputer(strategy="median")
    
    # CRITICAL: Impute THEN encode categorical features to handle missing categories
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),  # Handles NaNs in categorical
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, RAW_NUMERIC_COLS),
            ("cat", categorical_transformer, RAW_CATEGORICAL_COLS)
        ],
        verbose_feature_names_out=False
    )

def transform_split(df: pd.DataFrame, preprocessor: ColumnTransformer) -> pd.DataFrame:
    """
    Transform a raw split into processed format.
    
    Steps:
    1. Separate metadata (E, T, customerID) from features
    2. Transform features only using fitted preprocessor
    3. Recombine metadata + processed features
    """
    # Separate metadata (preserve exactly as-is for survival analysis)

    metadata = df[METADATA_COLS]  # These pass through unchanged
    features = df.drop(columns=METADATA_COLS)  # These get encoded
    
    # Transform features only
    X_processed = pd.DataFrame(
        preprocessor.transform(features),
        columns=preprocessor.get_feature_names_out(),
        index=metadata.index
    )
    
    # Recombine: metadata first, then processed features (survival libraries expect E,T first)
    return pd.concat([metadata, X_processed], axis=1)