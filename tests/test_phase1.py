# tests/test_phase1.py
"""
Phase 1 Unit Tests
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from ecomopti.phase1.preprocess import clean_data, get_feature_preprocessor, transform_split
from ecomopti.phase1.config import METADATA_COLS, RAW_NUMERIC_COLS, RAW_CATEGORICAL_COLS

import logging
logging.basicConfig(level=logging.WARNING)

@pytest.fixture
def sample_raw_data():
    """Create sample raw data mimicking telco.csv structure."""
    return pd.DataFrame({
        "customerID": ["C1", "C2", "C3", "C4"],
        "gender": ["Male", "Female", "Male", "Female"],
        "SeniorCitizen": [0, 1, 0, 1],
        "Partner": ["Yes", "No", "Yes", "No"],
        "Dependents": ["No", "Yes", "No", np.nan],  # Add NaN for testing
        "tenure": [1, 12, 24, np.nan],
        "PhoneService": ["Yes", "Yes", "No", "Yes"],
        "MultipleLines": ["No", "Yes", np.nan, "No"],
        "InternetService": ["DSL", "Fiber optic", "DSL", "No"],
        "OnlineSecurity": ["No", "Yes", "No", np.nan],
        "OnlineBackup": ["Yes", "No", np.nan, "Yes"],
        "DeviceProtection": ["No", "Yes", "No", "No"],
        "TechSupport": ["No", "No", "Yes", "Yes"],
        "StreamingTV": ["Yes", "No", "Yes", np.nan],
        "StreamingMovies": ["No", "Yes", "No", "Yes"],
        "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month"],
        "PaperlessBilling": ["Yes", "No", "Yes", "No"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        "MonthlyCharges": [29.85, 56.95, np.nan, 70.35],
        "TotalCharges": ["29.85", "657.3", "", "892.1"],
        "Churn": ["No", "Yes", "No", "Yes"]
    })

class TestCleanData:
    """Test the clean_data function."""
    
    def test_creates_targets_and_drops_total_charges(self, sample_raw_data):
        """P0 #1: Verify TotalCharges is explicitly dropped and targets created."""
        cleaned = clean_data(sample_raw_data)
        
        # Targets created
        assert "E" in cleaned.columns
        assert "T" in cleaned.columns
        
        # TotalCharges explicitly dropped (leakage prevention)
        assert "TotalCharges" not in cleaned.columns
        
        # Target values correct
        assert cleaned["E"].tolist() == [0, 1, 0, 1]
        assert cleaned["T"].tolist() == [1, 12, 24, 0]  # NaN -> 0
    
    def test_logs_warning_for_missing_tenure(self, sample_raw_data, caplog):
        """P1 #7: Verify warning is logged for missing tenure values."""
        with caplog.at_level(logging.WARNING):
            clean_data(sample_raw_data)
        
        assert "missing tenure values" in caplog.text
        assert "setting to 0" in caplog.text

class TestSeniorCitizenEncoding:
    """P1 #6: Test that SeniorCitizen is treated as numeric."""
    
    def test_senior_citizen_in_numeric_cols(self):
        assert "SeniorCitizen" in RAW_NUMERIC_COLS
        assert "SeniorCitizen" not in RAW_CATEGORICAL_COLS

class TestPreprocessor:
    """Test preprocessor without fitting (configuration checks)."""
    
    def test_preprocessor_configuration(self):
        """Verify preprocessor has correct transformers."""
        preprocessor = get_feature_preprocessor()
    
        from sklearn.compose import ColumnTransformer
        assert isinstance(preprocessor, ColumnTransformer)
    
        # Get transformer names (these exist before fitting)
        transformer_names = [name for name, _, _ in preprocessor.transformers]
        assert "num" in transformer_names
        assert "cat" in transformer_names
    
        # Check numeric transformer is median imputer (not scaler)
        from sklearn.impute import SimpleImputer
        num_transformer = preprocessor.transformers[0][1]  # Access directly by index
        assert isinstance(num_transformer, SimpleImputer)
        assert num_transformer.strategy == "median"
    
        # Check categorical transformer is a Pipeline
        from sklearn.pipeline import Pipeline
        cat_transformer = preprocessor.transformers[1][1]  # Access directly by index
        assert isinstance(cat_transformer, Pipeline)
        assert cat_transformer.steps[0][0] == "imputer"
        assert cat_transformer.steps[1][0] == "encoder"

class TestTransformSplit:
    """Test transformation with fitted preprocessor."""
    
    @pytest.fixture
    def fitted_preprocessor(self, sample_raw_data):
        """Create and fit a preprocessor on cleaned data."""
        cleaned = clean_data(sample_raw_data)
        preprocessor = get_feature_preprocessor()
        preprocessor.fit(cleaned.drop(columns=METADATA_COLS))
        return preprocessor
    
    def test_transform_split_preserves_metadata(self, sample_raw_data, fitted_preprocessor):
        """Verify metadata columns pass through transformation unchanged."""
        cleaned = clean_data(sample_raw_data)
        processed = transform_split(cleaned, fitted_preprocessor)
        
        # Metadata columns preserved
        for col in METADATA_COLS:
            assert col in processed.columns
        
        # Metadata values unchanged (compare as strings for safety)
        pd.testing.assert_series_equal(processed["E"], cleaned["E"], check_names=False)
        pd.testing.assert_series_equal(processed["T"], cleaned["T"], check_names=False)
        pd.testing.assert_series_equal(processed["customerID"], cleaned["customerID"], check_names=False)
    
    def test_transform_split_creates_encoded_features(self, sample_raw_data, fitted_preprocessor):
        """Verify transformation creates expected encoded features."""
        cleaned = clean_data(sample_raw_data)
        processed = transform_split(cleaned, fitted_preprocessor)
        
        # Should have more columns than metadata
        assert processed.shape[1] > len(METADATA_COLS)
        
        # All non-metadata columns should be numeric (encoded)
        feature_cols = processed.drop(columns=METADATA_COLS)
        assert all(feature_cols.dtypes.apply(lambda x: np.issubdtype(x, np.number)))
        
        # No NaN values in processed features (thanks to imputation)
        assert not processed.isna().any().any()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])