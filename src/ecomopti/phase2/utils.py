"""
Utility functions for Phase 2 (actually used).
"""

import json
import joblib
from pathlib import Path
from typing import List, Dict, Any

def save_model_artifact(
    artifact_path: Path,
    data: Dict[str, Any],
    filename: str = "artifact.json"
):
    """Save a JSON artifact (e.g., metrics, feature list)."""
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    (artifact_path.parent / filename).write_text(json.dumps(data, indent=2))

def get_feature_names_from_preprocessor(preprocessor) -> List[str]:
    """Safely extract feature names."""
    try:
        return preprocessor.get_feature_names_out().tolist()
    except:
        return ["feature_placeholder"]

def safe_load_model(model_path: Path, model_name: str):
    """Load model with descriptive error."""
    if not model_path.exists():
        raise FileNotFoundError(f"{model_name} model not found at {model_path}")
    return joblib.load(model_path)