"""Utility functions for Phase 3 Uplift Modeling."""

import joblib
import json
import numpy as np
from pathlib import Path
from typing import Any

def safe_load_model(model_path: Path, model_name: str) -> Any:
    """Load a serialized model with descriptive error handling."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_name} model not found at {model_path}. Run the training pipeline first!"
        )
    try:
        return joblib.load(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load {model_name} from {model_path}: {e}")

def save_json(data: dict, path: Path) -> None:
    """Save dictionary as formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)

def load_json(path: Path) -> dict:
    """Load JSON file as dictionary."""
    with open(path, "r") as f:
        return json.load(f)

def ensure_dir(path: Path) -> None:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)