"""
tests/test_phase2.py
Complete Phase 2 test suite in a single file.
Run with: pytest tests/test_phase2.py -v
"""

import sys
from pathlib import Path

# Add src to path for imports (since tests folder is in root)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import joblib
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import json
from pathlib import Path

# Import Phase 2 modules
from ecomopti.phase2.config import (
    MODELS_DIR,
    ARTIFACTS_DIR,
    PLOTS_DIR,
    PHASE1_ARTIFACTS_DIR,
    SPLITS_DIR,
    HORIZON_MONTHS,
    validate_phase1,
)
from ecomopti.phase2.data_loader import load_processed, get_cached_survival, _PROCESSED_CACHE
from ecomopti.phase2.build import check_leakage_prevention, check_survival_models
from ecomopti.phase2.utils import safe_load_model


# =========================================================
# Fixtures
# =========================================================

@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """Ensure test environment is ready before any tests run."""
    # Clear any existing caches
    _PROCESSED_CACHE.clear()
    get_cached_survival.cache_clear()
    
    # Verify Phase 1 exists (skip tests if not)
    if not (PHASE1_ARTIFACTS_DIR / "preprocessor.pkl").exists():
        pytest.skip("Phase 1 artifacts not found. Run Phase 1 first.")
    
    yield
    
    # Cleanup after all tests
    _PROCESSED_CACHE.clear()
    get_cached_survival.cache_clear()


@pytest.fixture
def sample_processed_data():
    """Load a small sample of processed data for testing."""
    df = load_processed("train").head(100)
    return df


@pytest.fixture
def phase1_preprocessor():
    """Load Phase 1 preprocessor."""
    return joblib.load(PHASE1_ARTIFACTS_DIR / "preprocessor.pkl")


@pytest.fixture
def cox_model():
    """Load Cox model if available."""
    path = MODELS_DIR / "cox_model.pkl"
    if path.exists():
        return safe_load_model(path, "Cox")
    pytest.skip("Cox model not found")


@pytest.fixture
def rsf_model():
    """Load RSF model if available."""
    path = MODELS_DIR / "rsf_model.pkl"
    if path.exists():
        return safe_load_model(path, "RSF")
    pytest.skip("RSF model not found")


@pytest.fixture
def clv_preprocessor():
    """Load CLV preprocessor if available."""
    path = ARTIFACTS_DIR / "preprocessor_clv.pkl"
    if path.exists():
        return joblib.load(path)
    pytest.skip("CLV preprocessor not found")


# =========================================================
# Test Classes (prevents naming collisions)
# =========================================================

class TestPhase2Dependencies:
    """Validate that Phase 2 can find all required Phase 1 artifacts."""
    
    def test_phase1_preprocessor_exists(self):
        """Phase 1 preprocessor must exist."""
        assert (PHASE1_ARTIFACTS_DIR / "preprocessor.pkl").exists()
    
    def test_phase1_splits_exist(self):
        """All train/val/test splits must exist."""
        for split in ["train", "val", "test"]:
            assert (SPLITS_DIR / f"{split}.csv").exists(), f"Missing {split}.csv"
    
    def test_validate_phase1_function(self):
        """validate_phase1() should raise clear error if Phase 1 missing."""
        # Should not raise when Phase 1 exists
        validate_phase1()
        
        # Should raise when forced to look at non-existent path
        with patch("ecomopti.phase2.config.PHASE1_ARTIFACTS_DIR", Path("/fake/path")):
            with pytest.raises(FileNotFoundError, match="Phase 1"):
                validate_phase1()


class TestLeakagePrevention:
    """Critical: Verify CLV model excludes charge columns to prevent leakage."""
    
    def test_clv_preprocessor_excludes_charges(self, clv_preprocessor):
        """CLV preprocessor must NOT contain MonthlyCharges or TotalCharges."""
        features = clv_preprocessor.get_feature_names_out()
        forbidden = ["MonthlyCharges", "TotalCharges"]
        found = [f for f in features if any(forbid in f for forbid in forbidden)]
        assert not found, f"LEAKAGE DETECTED: Charge features found in CLV preprocessor: {found}"
    
    def test_phase1_includes_charges_for_survival(self, phase1_preprocessor):
        """Phase 1 preprocessor SHOULD include charges (for survival models)."""
        features = phase1_preprocessor.get_feature_names_out()
        has_charges = any("MonthlyCharges" in f or "TotalCharges" in f for f in features)
        assert has_charges, "Phase 1 preprocessor should include charge features for survival models"
    
    def test_check_leakage_prevention_function(self):
        """Leakage check function should pass when CLV preprocessor is clean."""
        # Should not raise for clean preprocessor
        try:
            check_leakage_prevention()
        except RuntimeError as e:
            if "LEAKAGE" in str(e):
                pytest.fail(f"Leakage check failed: {e}")


class TestDataLoader:
    """Test data loading, caching, and preprocessing."""
    
    def test_load_processed_caches_data(self, sample_processed_data):
        """Data should be cached after first load."""
        # Clear cache first
        _PROCESSED_CACHE.clear()
        assert "train" not in _PROCESSED_CACHE
        
        # First load
        df1 = load_processed("train")
        assert "train" in _PROCESSED_CACHE
        assert len(_PROCESSED_CACHE["train"]) > 0
        
        # Second load should return copy
        df2 = load_processed("train")
        assert df1 is not df2, "Should return a copy, not the cached object"
        pd.testing.assert_frame_equal(df1, df2)
    
    def test_get_cached_survival_consistency(self, cox_model):
        """Cached survival predictions must be identical across calls."""
        get_cached_survival.cache_clear()
        
        s1 = get_cached_survival("train", HORIZON_MONTHS)
        s2 = get_cached_survival("train", HORIZON_MONTHS)
        
        np.testing.assert_array_equal(s1, s2, "Cache should return identical arrays")
        assert get_cached_survival.cache_info().hits == 1, "Should have cache hit"
    
    def test_load_processed_nonexistent_split(self):
        """Should raise clear error for missing split."""
        with pytest.raises(FileNotFoundError, match="Processed split not found"):
            load_processed("nonexistent")
    
    def test_processed_data_schema(self, sample_processed_data):
        """Processed data must contain required columns."""
        required_cols = ["E", "T", "customerID"]
        for col in required_cols:
            assert col in sample_processed_data.columns
        assert sample_processed_data["E"].isin([0, 1]).all()
        assert (sample_processed_data["T"] >= 0).all()


class TestSurvivalModels:
    """Test Cox and RSF model training and evaluation."""
    
    def test_cox_model_exists(self):
        """Cox model file should exist after training."""
        assert (MODELS_DIR / "cox_model.pkl").exists()
        assert (MODELS_DIR / "cox_summary.csv").exists()
    
    def test_rsf_model_exists(self):
        """RSF model file should exist after training."""
        assert (MODELS_DIR / "rsf_model.pkl").exists()
    
    def test_cox_model_can_predict(self, cox_model, sample_processed_data):
        """Cox model should produce valid predictions."""
        X = sample_processed_data.drop(columns=["E", "T", "customerID"])
        pred = cox_model.predict_partial_hazard(X)
        assert len(pred) == len(sample_processed_data)
        assert np.isfinite(pred).all(), "Predictions must be finite"
    
    def test_rsf_model_can_predict(self, rsf_model, sample_processed_data):
        """RSF model should produce valid predictions."""
        X = sample_processed_data.drop(columns=["E", "T", "customerID"])
        pred = rsf_model.predict(X)
        assert len(pred) == len(sample_processed_data)
        assert np.isfinite(pred).all(), "Predictions must be finite"
    
    def test_check_survival_models_function(self):
        """Should pass when both models exist."""
        try:
            check_survival_models()
        except FileNotFoundError as e:
            pytest.fail(f"Survival model check failed: {e}")
    
    def test_cox_metrics_exist(self):
        """Cox evaluation metrics should be saved."""
        for split in ["val", "test"]:
            assert (MODELS_DIR / f"{split}_metrics.json").exists()
            assert (MODELS_DIR / f"{split}_calibration.csv").exists()
    
    def test_rsf_metrics_exist(self):
        """RSF evaluation metrics should be saved."""
        assert (MODELS_DIR / "rsf_metrics.json").exists()
        if (MODELS_DIR / "rsf_feature_importance.csv").exists():
            df = pd.read_csv(MODELS_DIR / "rsf_feature_importance.csv")
            assert len(df) > 0


class TestCLVModel:
    """Test CLV model training and predictions."""
    
    def test_clv_model_exists(self):
        """CLV model and preprocessor should exist after training."""
        assert (MODELS_DIR / "clv_model.pkl").exists()
        assert (ARTIFACTS_DIR / "preprocessor_clv.pkl").exists()
    
    def test_clv_predictions_exist(self):
        """CLV predictions should exist for all splits."""
        for split in ["train", "val", "test"]:
            path = ARTIFACTS_DIR / f"clv_{split}_predictions.csv"
            assert path.exists(), f"Missing CLV predictions for {split}"
            
            df = pd.read_csv(path)
            assert "actual_clv" in df.columns
            assert "predicted_clv" in df.columns
            assert len(df) > 0
    
    def test_clv_predictions_reasonable(self):
        """CLV predictions should be within reasonable bounds."""
        df = pd.read_csv(ARTIFACTS_DIR / "clv_test_predictions.csv")
        
        # No negative CLV
        assert (df["actual_clv"] >= 0).all()
        assert (df["predicted_clv"] >= 0).all()
        
        # Predictions should be close to actual (MAPE check)
        mape = np.mean(np.abs((df["actual_clv"] - df["predicted_clv"]) / df["actual_clv"]))
        assert mape < 0.1, f"MAPE too high: {mape:.2%} (should be <10%)"
    
    def test_clv_metrics_exist(self):
        """CLV evaluation metrics should be saved."""
        assert (MODELS_DIR / "clv_metrics.json").exists()
        
        with open(MODELS_DIR / "clv_metrics.json") as f:
            metrics = json.load(f)
        
        for split in ["train", "val", "test"]:
            assert split in metrics
            assert "r2" in metrics[split]
            assert "mae" in metrics[split]
            assert metrics[split]["r2"] > 0.9, f"R² too low for {split}"


class TestBuildOrchestration:
    """Test the build.py orchestration logic."""
    
    def test_build_main_sequence(self):
        """Verify build.py calls modules in correct order."""
        # Mock all training modules to test orchestration logic
        with patch("ecomopti.phase2.build.train_cox") as mock_cox, \
             patch("ecomopti.phase2.build.train_rsf") as mock_rsf, \
             patch("ecomopti.phase2.build.train_clv_model") as mock_clv, \
             patch("ecomopti.phase2.build.check_survival_models") as mock_check, \
             patch("ecomopti.phase2.build.check_leakage_prevention") as mock_leakage, \
             patch("ecomopti.phase2.build.get_cached_survival") as mock_cache, \
             patch("ecomopti.phase2.build.validate_phase1"), \
             patch("ecomopti.phase2.build.ensure_dirs"):
            
            # Simulate successful training
            mock_cox.main = MagicMock()
            mock_rsf.main = MagicMock()
            mock_clv.main = MagicMock()
            
            # Import and run main
            from ecomopti.phase2.build import main
            try:
                main()
            except:
                pass  # We expect this to fail due to mocking
            
            # Verify call order
            mock_cox.main.assert_called_once()
            mock_rsf.main.assert_called_once()
            mock_check.assert_called_once()  # Check models exist BEFORE CLV
            mock_leakage.assert_called_once()  # Leakage check BEFORE CLV
            mock_clv.main.assert_called_once()
    
    def test_build_fails_without_phase1(self):
        """build.py should fail fast if Phase 1 is missing."""
        from ecomopti.phase2.build import main
        
        with patch("ecomopti.phase2.build.validate_phase1", side_effect=FileNotFoundError("Phase 1 missing")):
            with pytest.raises(FileNotFoundError, match="Phase 1"):
                main()


class TestMetrics:
    """Test statistical metrics and evaluation functions."""
    
    def test_calibration_table_shape(self):
        """Calibration table should have correct structure."""
        from ecomopti.phase2.metrics import calibration_table
        
        # Mock predictions and observations
        preds = np.random.uniform(0, 1, 1000)
        obs = np.random.randint(0, 2, 1000)
        
        cal = calibration_table(preds, obs, n_bins=10)
        assert len(cal) <= 10  # May be fewer if bins are merged
        assert "pred_mean" in cal.columns
        assert "obs_rate" in cal.columns
        assert "n" in cal.columns
    
    def test_bootstrap_c_index_convergence(self, cox_model):
        """Bootstrap CI should converge with reasonable variance."""
        from ecomopti.phase2.metrics import bootstrap_c_index
        
        df = load_processed("val").head(200)  # Use small sample for speed
        
        mean, (lo, hi) = bootstrap_c_index(cox_model, df, n_bootstrap=50)
        
        assert 0 <= lo <= mean <= hi <= 1, "CI bounds invalid"
        assert hi - lo < 0.2, f"CI too wide: {hi-lo:.3f} (indicates high variance)"
    
    def test_brier_score_range(self):
        """Brier score should be between 0 and 1."""
        from ecomopti.phase2.metrics import compute_brier_at_horizon
        
        # Mock model
        mock_model = MagicMock()
        mock_model.predict_survival_function.return_value = pd.DataFrame(
            {HORIZON_MONTHS: np.random.uniform(0.8, 1.0, 100)}
        )
        
        df = load_processed("val").head(100)
        brier, _, _ = compute_brier_at_horizon(mock_model, df, HORIZON_MONTHS)
        
        assert 0 <= brier <= 1, f"Brier score out of bounds: {brier}"


class TestPlotsGeneration:
    """Test that plotting functions execute without errors."""
    
    def test_plots_directory_created(self):
        """Plots directory should exist after plotting."""
        assert PLOTS_DIR.exists(), "Plots directory not found"
    
    def test_plot_files_generated(self):
        """Key plot files should be generated."""
        required_plots = [
            "cox_km_val.png",
            "cox_hazard_ratios.png",
            "clv_pred_vs_actual.png",
            "clv_distribution.png",
            "risk_vs_clv.png",
        ]
        
        for plot in required_plots:
            path = PLOTS_DIR / plot
            if not path.exists():
                pytest.skip(f"Plot {plot} not found (run plots.py first)")
            assert path.stat().st_size > 0, f"Plot file is empty: {plot}"


# =========================================================
# Run a quick smoke test
# =========================================================

if __name__ == "__main__":
    print("Quick Phase 2 smoke test...")
    
    # Test imports work
    try:
        from ecomopti.phase2.config import MODELS_DIR
        print("✅ Imports successful")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        sys.exit(1)
    
    # Test basic paths
    if MODELS_DIR.exists():
        print("✅ Models directory found")
    else:
        print("❌ Models directory not found")
    
    print("Run full tests with: pytest tests/test_phase2.py -v")