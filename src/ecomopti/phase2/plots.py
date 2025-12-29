"""
Phase 2 Plotting: All visualizations with calibration, residuals, and partial dependence.
"""

import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from lifelines import KaplanMeierFitter
from sklearn.metrics import roc_curve, auc
from sklearn.inspection import PartialDependenceDisplay
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
# Suppress feature name warnings from sklearn/LightGBM
warnings.filterwarnings(action='ignore', category=UserWarning, message="X does not have valid feature names")

from .config import MODELS_DIR, ARTIFACTS_DIR, PLOTS_DIR, HORIZON_MONTHS, PDP_TOP_FEATURES, RANDOM_STATE
from .data_loader import load_processed, get_cached_survival  # Import from data_loader
from .utils import safe_load_model
# Remove this import - no longer needed
# from .train_clv_model import get_cached_survival

sns.set_style("whitegrid")
PLOTS_DIR.mkdir(exist_ok=True)
logger = logging.getLogger("phase2.plots")

# ========================================
# Define ALL functions BEFORE main()
# ========================================

def plot_cox_km_deciles(split="val"):
    """Kaplan-Meier curves by predicted risk decile."""
    df = load_processed(split)
    cox = safe_load_model(MODELS_DIR / "cox_model.pkl", "Cox")
    X = df.drop(columns=["E", "T", "customerID"])
    
    pred_risk = cox.predict_partial_hazard(X).squeeze()
    df["risk_decile"] = pd.qcut(pred_risk.rank(method="first"), q=10, labels=False, duplicates="drop")
    
    plt.figure(figsize=(12, 7))
    km = KaplanMeierFitter()
    for decile in sorted(df["risk_decile"].unique()):
        sub = df[df["risk_decile"] == decile]
        km.fit(sub["T"], sub["E"], label=f"Decile {int(decile)}")
        km.plot_survival_function(ci_show=False)
    
    plt.title(f"Cox KM Curves ({split})")
    plt.xlabel("Time (months)")
    plt.ylabel("Survival Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"cox_km_{split}.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {PLOTS_DIR / f'cox_km_{split}.png'}")

def plot_cox_hazard_ratios():
    """Hazard ratios for significant features."""
    cox = safe_load_model(MODELS_DIR / "cox_model.pkl", "Cox")
    summary = cox.summary
    sig = summary[summary["p"] < 0.05]
    dfp = sig if len(sig) > 0 else summary.head(20)
    
    hr = dfp["exp(coef)"].sort_values()
    
    plt.figure(figsize=(10, max(6, 0.3 * len(hr))))
    hr.plot(kind="barh", color="steelblue")
    plt.title("Cox Hazard Ratios")
    plt.xlabel("Hazard Ratio (>1 = Higher Risk)")
    plt.axvline(x=1, color="r", linestyle="--")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "cox_hazard_ratios.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {PLOTS_DIR / 'cox_hazard_ratios.png'}")

def plot_rsf_feature_importance():
    """Plot RSF feature importance from CSV (if available)."""
    importance_path = MODELS_DIR / "rsf_feature_importance.csv"
    
    if not importance_path.exists():
        logger.warning("RSF feature importance not found. Set COMPUTE_PERMUTATION_IMPORTANCE=True to generate.")
        return
    
    df_importance = pd.read_csv(importance_path).head(10)
    
    plt.figure(figsize=(12, 8))
    plt.barh(df_importance["feature"], df_importance["importance_mean"], color="darkgreen")
    plt.title("RSF Top 10 Features (Permutation Importance)")
    plt.xlabel("Mean Importance")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "rsf_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {PLOTS_DIR / 'rsf_importance.png'}")

def plot_clv_pred_vs_actual():
    """CLV predictions vs actual."""
    model = safe_load_model(MODELS_DIR / "clv_model.pkl", "CLV")
    pre = safe_load_model(ARTIFACTS_DIR / "preprocessor_clv.pkl", "CLV Preprocessor")
    df = load_processed("val")
    
    df["survival_prob"] = get_cached_survival("val", HORIZON_MONTHS)
    df["base_clv"] = df["survival_prob"] * df["MonthlyCharges"] * HORIZON_MONTHS
    
    X = pre.transform(df.drop(columns=["E", "T", "customerID"]))
    y_pred = model.predict(X)
    
    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(df["base_clv"], y_pred)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(df["base_clv"], y_pred, alpha=0.5, s=15, color="purple")
    plt.plot([df["base_clv"].min(), df["base_clv"].max()], [df["base_clv"].min(), df["base_clv"].max()], "r--")
    plt.title(f"CLV: Pred vs Actual (MAE=${mae:.2f})")
    plt.xlabel("Actual CLV ($)")
    plt.ylabel("Predicted CLV ($)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "clv_pred_vs_actual.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {PLOTS_DIR / 'clv_pred_vs_actual.png'}")

def plot_calibration():
    """Calibration curve for Cox model at 6-month horizon."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, split in enumerate(["val", "test"]):
        cal_path = MODELS_DIR / f"{split}_calibration.csv"
        if not cal_path.exists():
            logger.warning(f"Calibration file not found: {cal_path}")
            continue
            
        cal = pd.read_csv(cal_path)
        
        axes[idx].plot(cal["pred_mean"], cal["obs_rate"], marker='o', label="Model")
        axes[idx].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
        axes[idx].set_title(f"Calibration ({split})")
        axes[idx].set_xlabel("Predicted churn probability")
        axes[idx].set_ylabel("Observed churn rate")
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "cox_calibration.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {PLOTS_DIR / 'cox_calibration.png'}")

def plot_clv_distribution():
    """Histogram of actual vs predicted CLV."""
    # Load predictions which already contains both actual and predicted CLV
    pred_df = pd.read_csv(ARTIFACTS_DIR / "clv_test_predictions.csv")
    
    plt.figure(figsize=(12, 6))
    plt.hist(pred_df["actual_clv"], bins=50, alpha=0.6, label="Actual CLV", color="steelblue")
    plt.hist(pred_df["predicted_clv"], bins=50, alpha=0.6, label="Predicted CLV", color="darkorange")
    plt.title("CLV Distribution: Actual vs Predicted")
    plt.xlabel("CLV ($)")
    plt.ylabel("Number of customers")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "clv_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {PLOTS_DIR / 'clv_distribution.png'}")

def plot_risk_clv_scatter():
    """
    Enhanced Risk vs CLV with 4-quadrant segmentation.
    Adds median lines, quadrant labels, and saves segment assignments.
    """
    df = load_processed("test")
    
    # Load models
    cox = safe_load_model(MODELS_DIR / "cox_model.pkl", "Cox")
    
    # Compute risk and CLV (ensure consistency)
    X = df.drop(columns=["E", "T", "customerID"])
    df["risk"] = cox.predict_partial_hazard(X).squeeze()
    df["survival_prob"] = get_cached_survival("test", HORIZON_MONTHS)
    df["clv"] = df["survival_prob"] * df["MonthlyCharges"] * HORIZON_MONTHS
    
    # 🔷 SEGMENTATION: Compute median boundaries
    risk_median = df["risk"].median()
    clv_median = df["clv"].median()
    
    # Assign quadrant labels
    conditions = [
        (df["risk"] <= risk_median) & (df["clv"] >= clv_median),
        (df["risk"] > risk_median) & (df["clv"] >= clv_median),
        (df["risk"] <= risk_median) & (df["clv"] < clv_median),
        (df["risk"] > risk_median) & (df["clv"] < clv_median)
    ]
    choices = ["Stable Value", "At-Risk High-Value", "Safe Low-Value", "High-Risk Low-Value"]
    df["segment"] = np.select(conditions, choices, default="Uncategorized")
    
    # Plot setup
    plt.figure(figsize=(12, 8))
    
    # Scatter by segment (color and marker)
    colors = {"Stable Value": "steelblue", "At-Risk High-Value": "darkred", 
              "Safe Low-Value": "lightgray", "High-Risk Low-Value": "orange"}
    markers = {"Stable Value": "o", "At-Risk High-Value": "s", 
               "Safe Low-Value": "^", "High-Risk Low-Value": "x"}
    
    for segment in choices:
        seg_data = df[df["segment"] == segment]
        plt.scatter(seg_data["risk"], seg_data["clv"], 
                    c=colors[segment], marker=markers[segment], 
                    alpha=0.6, s=30, label=f"{segment} (n={len(seg_data)})")
    
    # Add quadrant lines (dashed)
    plt.axvline(x=risk_median, color='black', linestyle='--', alpha=0.5, 
                label=f'Median Risk ({risk_median:.2f})')
    plt.axhline(y=clv_median, color='black', linestyle='--', alpha=0.5, 
                label=f'Median CLV (${clv_median:.0f})')
    
    # 🔷 ANNOTATE: Add segment statistics
    stats_text = "\n".join([
        f"{seg}: Avg CLV=${df[df['segment']==seg]['clv'].mean():.0f}, "
        f"Churn Rate={df[df['segment']==seg]['E'].mean():.1%}"
        for seg in choices
    ])
    plt.text(0.98, 0.02, stats_text, transform=plt.gca().transAxes, 
             fontsize=9, verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Formatting
    plt.title("Risk vs CLV: Customer Segmentation", fontsize=14, fontweight='bold')
    plt.xlabel("Cox Risk Score (higher = more likely to churn)", fontsize=12)
    plt.ylabel(f"6-Month CLV ($)", fontsize=12)
    plt.legend(loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(PLOTS_DIR / "risk_vs_clv_segmented.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # 🔷 EXPORT: Save segment assignments for CRM
    export_path = ARTIFACTS_DIR / "customer_segments_test.csv"
    df[["customerID", "risk", "clv", "segment"]].to_csv(export_path, index=False)
    
    logger.info(f"Saved segmented plot: {PLOTS_DIR / 'risk_vs_clv_segmented.png'}")
    logger.info(f"Saved segment assignments: {export_path}")

def plot_time_dependent_roc():
    df = load_processed("val")
    cox = safe_load_model(MODELS_DIR / "cox_model.pkl", "Cox")
    X = df.drop(columns=["E", "T", "customerID"])
    risk = cox.predict_partial_hazard(X).squeeze()
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    for idx, horizon in enumerate([6, 12, 18]):
        observed = ((df["T"] <= horizon) & (df["E"] == 1)).astype(int)
        fpr, tpr, _ = roc_curve(observed, risk)  # ✅ Keep this fixed line
        roc_auc = auc(fpr, tpr)
        
        axes[idx].plot(fpr, tpr, color='steelblue', lw=2, label=f'AUC = {roc_auc:.3f}')
        axes[idx].plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.3)
        axes[idx].set_title(f'{horizon}-month ROC')
        axes[idx].legend()
        axes[idx].set_xlabel('False Positive Rate')
        axes[idx].set_ylabel('True Positive Rate')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "cox_time_dependent_roc.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {PLOTS_DIR / 'cox_time_dependent_roc.png'}")

def plot_clv_residuals():
    """CLV residuals vs fitted to check model assumptions."""
    df = load_processed("test")
    model = safe_load_model(MODELS_DIR / "clv_model.pkl", "CLV")
    pre = safe_load_model(ARTIFACTS_DIR / "preprocessor_clv.pkl", "CLV Preprocessor")
    
    # ✅ FIX: Compute survival_prob first - it's not saved in raw data
    df["survival_prob"] = get_cached_survival("test", HORIZON_MONTHS)
    
    X = pre.transform(df.drop(columns=["E", "T", "customerID", "survival_prob"]))
    fitted = model.predict(X)
    actual = df["survival_prob"] * df["MonthlyCharges"] * HORIZON_MONTHS
    residuals = actual - fitted
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Residuals vs Fitted
    ax1.scatter(fitted, residuals, alpha=0.3, s=10)
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax1.set_xlabel("Fitted CLV ($)")
    ax1.set_ylabel("Residuals ($)")
    ax1.set_title("Residuals vs Fitted")
    
    # Residual Distribution
    ax2.hist(residuals, bins=50, alpha=0.7, color='steelblue', density=True)
    ax2.set_xlabel("Residuals ($)")
    ax2.set_ylabel("Density")
    ax2.set_title("Residual Distribution")
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "clv_residuals.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {PLOTS_DIR / 'clv_residuals.png'}")

def plot_clv_partial_dependence():
    """
    Partial dependence plots for CLV model features.
    Dynamically selects from valid features to avoid leakage.
    """
    model = safe_load_model(MODELS_DIR / "clv_model.pkl", "CLV")
    pre = safe_load_model(ARTIFACTS_DIR / "preprocessor_clv.pkl", "CLV Preprocessor")
    
    # Load test data
    df = load_processed("test")
    df["survival_prob"] = get_cached_survival("test", HORIZON_MONTHS)
    
    # Transform features (validates leakage removal)
    drop_cols = ["E", "T", "customerID", "survival_prob"]
    X_raw = df.drop(columns=drop_cols)
    X = pre.transform(X_raw)
    feature_names = pre.get_feature_names_out()
    
    # 🔷 DYNAMIC FEATURE MATCHING: Find features that actually exist
    available_features = []
    for i, fname in enumerate(feature_names):
        if any(target in fname for target in PDP_TOP_FEATURES):
            available_features.append(i)
    
    # Safety: Ensure we have at least 2 features
    if len(available_features) < 2:
        logger.warning(
            f"Only {len(available_features)} PDP features found. "
            f"Available: {[feature_names[i] for i in available_features]}"
        )
        # Fallback: Use first 2 valid features
        available_features = list(range(min(2, len(feature_names))))
    
    # Generate PDP
    try:
        fig, ax = plt.subplots(figsize=(12, 7))
        display = PartialDependenceDisplay.from_estimator(
            model, X, features=available_features[:3],  # Limit to top 3
            feature_names=feature_names,
            kind="both", subsample=800, random_state=RANDOM_STATE, 
            grid_resolution=20, ax=ax  # Higher resolution for cleaner curves
        )
        ax.set_title("CLV Partial Dependence: Top Non-Leakage Features", 
                     fontsize=14, fontweight='bold')
        ax.set_xlabel("Feature Value", fontsize=12)
        ax.set_ylabel("Partial Dependence (CLV $)", fontsize=12)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "clv_partial_dependence.png", dpi=150, bbox_inches="tight")
        plt.close()
        
        logger.info(f"✅ PDP saved: {PLOTS_DIR / 'clv_partial_dependence.png'}")
        logger.info(f"  Features plotted: {[feature_names[i] for i in available_features[:3]]}")
        
    except Exception as e:
        logger.error(f"PDP generation failed: {e}")
        raise  # Fail fast in CI/CD

def main():
    """Generate all plots including diagnostics."""
    logger.info("=" * 50)
    logger.info("GENERATING PHASE 2 PLOTS & DIAGNOSTICS")
    logger.info("=" * 50)
    
    # Original charts
    plot_cox_km_deciles("val")
    plot_cox_hazard_ratios()
    plot_rsf_feature_importance()
    plot_clv_pred_vs_actual()
    plot_calibration()
    plot_clv_distribution()
    plot_risk_clv_scatter()
    

    # NEW: Diagnostic plots
    logger.info("\n📊 Generating model diagnostics...")
    plot_time_dependent_roc()
    plot_clv_residuals()
    plot_clv_partial_dependence()
    
    logger.info(f"✅ All plots saved to {PLOTS_DIR}")

if __name__ == "__main__":
    main()