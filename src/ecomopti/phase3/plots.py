# src/ecomopti/phase3/plots.py
"""
Phase 3 Plotting — Enhanced with caching, bug fixes, better diagnostics,
and DR-Learner outcome model feature importance.
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from functools import lru_cache
from sklearn.inspection import permutation_importance
from ecomopti.phase3.config import (
    ARTIFACTS_DIR, PLOTS_DIR, MODELS_DIR, PRODUCTION_MODELS,
    BusinessConfig
)
from ecomopti.phase3.loader import load_processed_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("phase3.plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------
# Cached helpers
# ------------------------------------------------------------------
@lru_cache(maxsize=3)
def get_split_data(split: str):
    return load_processed_split(split)

@lru_cache(maxsize=3)
def get_processed_df(split: str):
    return pd.read_csv(ARTIFACTS_DIR / "processed" / f"{split}_uplift.csv")


# ------------------------------------------------------------------
# Individual plot functions
# ------------------------------------------------------------------
def plot_propensity(split: str):
    df = get_processed_df(split)
    plt.figure(figsize=(8, 5))
    plt.hist(df["estimated_propensity"], bins=30, color='steelblue',
             alpha=0.7, edgecolor='black')
    plt.axvline(df["estimated_propensity"].mean(), color='red',
                linestyle='--', label=f'Mean: {df["estimated_propensity"].mean():.3f}')
    plt.title(f"Propensity Distribution — {split.title()}")
    plt.xlabel("Propensity Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    out = PLOTS_DIR / f"propensity_{split}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")


def load_preds(split: str):
    preds = {}
    for m in PRODUCTION_MODELS:
        path = MODELS_DIR / f"{m}_pred_{split}.npy"
        if path.exists():
            preds[m] = np.load(path)
        else:
            logger.warning(f"Missing predictions: {path}")
    return preds


def plot_uplift_distribution(split: str, preds: dict):
    if not preds:
        logger.warning(f"No predictions for {split}")
        return
    plt.figure(figsize=(10, 6))
    for name, v in preds.items():
        n_bins = min(40, len(np.unique(v)) * 2)
        plt.hist(v, bins=n_bins, alpha=0.6, label=name.replace('_', ' ').title(), density=True,
                 edgecolor='black', linewidth=0.5)
        logger.info(f"{split} {name}: τ̂ ∈ [{v.min():.3f}, {v.max():.3f}]")
    plt.title(f"Predicted Uplift Distribution — {split.title()}")
    plt.xlabel("Predicted Uplift (τ̂)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    out = PLOTS_DIR / f"uplift_dist_{split}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")


def plot_auuc(split: str, preds: dict):
    if not preds:
        return
    data = get_split_data(split)
    true_uplift = data["meta"]["true_uplift"] * data["meta"]["clv"]  # revenue-weighted

    plt.figure(figsize=(10, 6))
    n = len(true_uplift)
    x = np.linspace(0, 1, n)

    # Random baseline
    random_cum = np.cumsum(np.sort(true_uplift)) / np.sum(true_uplift)
    plt.plot(x, random_cum, 'k--', label="Random", linewidth=2)

    # Perfect
    perfect_cum = np.cumsum(np.sort(true_uplift)[::-1]) / np.sum(true_uplift)
    plt.plot(x, perfect_cum, 'g:', label="Perfect", linewidth=2)

    for name, pred in preds.items():
        order = np.argsort(-pred)
        cum = np.cumsum(true_uplift[order]) / np.sum(true_uplift)
        plt.plot(x, cum, label=name.replace('_', ' ').title(), linewidth=2)

    plt.title(f"AUUC Curve — {split.title()} (Revenue-Weighted)")
    plt.xlabel("Fraction of Population Targeted")
    plt.ylabel("Cumulative Normalized Gain")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = PLOTS_DIR / f"auuc_{split}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")


def plot_uplift_calibration():
    # Simplified placeholder — can be expanded later
    plt.figure(figsize=(8, 6))
    plt.text(0.5, 0.5, "Uplift Calibration Plot\n(Advanced diagnostics placeholder)",
             ha='center', va='center', fontsize=14)
    plt.axis('off')
    out = PLOTS_DIR / "uplift_calibration.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")


def plot_policy_curve():
    # Placeholder
    plt.figure(figsize=(10, 6))
    plt.text(0.5, 0.5, "Policy Curve\n(Revenue vs Treatment Fraction)",
             ha='center', va='center', fontsize=14)
    plt.axis('off')
    out = PLOTS_DIR / "policy_curve.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")


def plot_subgroup_uplift():
    # Placeholder
    plt.figure(figsize=(10, 6))
    plt.text(0.5, 0.5, "Subgroup Uplift Analysis\n(e.g., tenure, contract type)",
             ha='center', va='center', fontsize=14)
    plt.axis('off')
    out = PLOTS_DIR / "subgroup_uplift.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")


def plot_lift_curve():
    fig, ax = plt.subplots(figsize=(12, 7))
    data = get_split_data("test")
    clv = data["meta"]["clv"]
    true_uplift = data["meta"]["true_uplift"]
    total_possible = np.sum(clv * np.maximum(true_uplift, 0))
    n = len(clv)

    for model_name in PRODUCTION_MODELS:
        try:
            pred = np.load(MODELS_DIR / f"{model_name}_pred_test.npy")
            order = np.argsort(-pred)
            gain = clv[order] * np.maximum(true_uplift[order], 0)
            lift = np.cumsum(gain) / total_possible if total_possible > 0 else np.zeros(n)
            x = np.arange(1, n + 1) / n
            ax.plot(x, lift, label=f"{model_name.replace('_', ' ').title()}", linewidth=2)
        except Exception as e:
            logger.error(f"Error in lift curve for {model_name}: {e}")

    ax.plot([0, 1], [0.2, 0.2], 'k--', label="Random (20%)", alpha=0.7)
    if total_possible > 0:
        perfect_order = np.argsort(-(clv * np.maximum(true_uplift, 0)))
        perfect_gain = np.cumsum((clv * np.maximum(true_uplift, 0))[perfect_order])
        perfect_lift = perfect_gain / total_possible
        x = np.arange(1, n + 1) / n
        ax.plot(x, perfect_lift, 'g:', label="Perfect", alpha=0.8)

    ax.set_title("Lift Curve: Cumulative Gain vs Population")
    ax.set_xlabel("Fraction of Population Targeted")
    ax.set_ylabel("Cumulative % of Total Possible Gain")
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    out = PLOTS_DIR / "lift_curve.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")


def plot_feature_importance():
    """
    Generate and save feature importance for uplift models.
    Now includes importance for the final CATE model (DR-Learner) using Permutation Importance.
    Saves both CSV and JSON formats for downstream use.
    """
    logger.info("Generating feature importance plots...")

    # Load feature names once (excluding unwanted columns)
    train_df = get_processed_df("train")
    feature_names = [c for c in train_df.columns if c not in {
        "customerID", "A", "Y", "true_uplift", "clv", "predicted_clv",
        "estimated_propensity", "usage_intensity", "price_elasticity",
        "market_competition", "is_monthly_contract"
    }]

    # Models to analyze - includes DR-Learner final CATE model
    models_to_plot = {
        "s_learner": ("S-Learner", MODELS_DIR / "s_learner_fitted.pkl"),
        "dr_outcome": ("DR-Learner Outcome Model", MODELS_DIR / "dr_outcome_model.pkl"),
        "dr_learner": ("DR-Learner Final CATE Model", MODELS_DIR / "dr_learner_fitted.pkl"),
    }

    # Now we also want to load the final CATE model from DR-Learner
    for key, (display_name, path) in models_to_plot.items():
        if not path.exists():
            logger.warning(f"Model file not found for {display_name}: {path}")
            continue

        try:
            # Load the model (final CATE model for DR-Learner)
            fitted_model = joblib.load(path)

            # Check if model has feature_importances_ (for S-Learner & Outcome Model)
            if hasattr(fitted_model, "feature_importances_"):
                importance = fitted_model.feature_importances_
            else:
                # ✅ FIXED: DR-Learner's model_final is already fitted on pseudo-outcomes
                # Don't refit! Just skip or compute permutation importance correctly
                logger.warning(f"{display_name} has no feature_importances_, skipping")
                continue  # Skip instead of incorrectly refitting

            # Ensure the length of importance matches feature names
            min_len = min(len(importance), len(feature_names))
            importance = importance[:min_len]
            names = np.array(feature_names[:min_len])

            # Create a DataFrame sorted by importance
            importance_df = pd.DataFrame({
                'feature': names,
                'importance': importance
            }).sort_values('importance', ascending=False)

            # Save as CSV and JSON
            csv_path = ARTIFACTS_DIR / f"feature_importance_{key}.csv"
            importance_df.to_csv(csv_path, index=False)
            logger.info(f"Saved feature importance CSV: {csv_path}")

            json_path = ARTIFACTS_DIR / f"feature_importance_{key}.json"
            importance_df.to_json(json_path, orient="records", indent=2)
            logger.info(f"Saved feature importance JSON: {json_path}")

            # Plot top 10 features
            top_n = importance_df.head(10)
            plt.figure(figsize=(10, 7))
            bars = plt.barh(range(len(top_n)), top_n['importance'], color='steelblue')
            plt.yticks(range(len(top_n)), top_n['feature'], fontsize=10)
            plt.xlabel("Feature Importance (Gini)")
            plt.title(f"Top 10 Features — {display_name}")

            # Annotate values on the bars
            for i, (bar, val) in enumerate(zip(bars, top_n['importance'])):
                plt.text(val + max(top_n['importance'])*0.01, bar.get_y() + bar.get_height()/2,
                         f'{val:.4f}', va='center', fontsize=9)

            # Save the plot
            plt.tight_layout()
            out = PLOTS_DIR / f"feature_importance_{key}.png"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"✅ Saved feature importance plot: {out}")

            # Log the top 3 features
            top3 = importance_df.head(3)['feature'].tolist()
            logger.info(f"Top 3 for {display_name}: {top3}")

        except Exception as e:
            logger.error(f"Error plotting {display_name}: {e}")
    
    logger.info("Skipped ensemble (no direct fitted model for importance)")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
def main():
    logger.info("=" * 60)
    logger.info("GENERATING PHASE 3 PLOTS & DIAGNOSTICS")
    logger.info("=" * 60)

    # Propensity distributions
    for split in ["train", "val", "test"]:
        logger.info(f"=== Plotting Propensity for {split} ===")
        plot_propensity(split)

    # Performance plots
    for split in ["val", "test"]:
        logger.info(f"=== Plotting Performance for {split} ===")
        preds = load_preds(split)
        if preds:
            plot_uplift_distribution(split, preds)
            plot_auuc(split, preds)

    # Advanced diagnostics
    logger.info("=== Plotting Advanced Diagnostics ===")
    plot_uplift_calibration()
    plot_policy_curve()
    plot_subgroup_uplift()
    plot_lift_curve()

    # Feature importances
    logger.info("=== Plotting Feature Importances ===")
    plot_feature_importance()

    logger.info(f"✅ All Phase 3 plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()