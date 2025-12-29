# src/ecomopti/phase1/config.py

"""
Phase 1 Configuration: Centralized paths, constants & column definitions.
"""

from pathlib import Path

# Project root (3 levels up from phase1/config.py)
ROOT = Path(__file__).resolve().parents[3]

# Data paths
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SPLITS_DIR = DATA_DIR / "splits"

# Artifact paths
ARTIFACTS_DIR = ROOT / "artifacts" / "phase1"

# Reproducibility seed
RANDOM_STATE = 42

# === CENTRALIZED COLUMN DEFINITIONS ===

# Metadata columns: preserved as-is, never used as features or targets
METADATA_COLS = ["E", "T", "customerID"]

# Raw numeric features (will be imputed but NOT scaled)
# P1 #6: SeniorCitizen is numeric (0/1), not categorical
RAW_NUMERIC_COLS = [ "MonthlyCharges", "SeniorCitizen","tenure"]

# Raw categorical features (will be one-hot encoded)
RAW_CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract", 
    "PaperlessBilling", "PaymentMethod"
]

def ensure_dirs():
    """Create directories at runtime (not on import)."""
    for p in [RAW_DIR, SPLITS_DIR, ARTIFACTS_DIR]:
        p.mkdir(parents=True, exist_ok=True)