# src/ecomopti/phase3/train_uplift.py
"""
train_uplift.py: XGBoost + Optuna hyperparameter tuning for uplift meta-learners.

DR-Learner Component Architecture:
1. Propensity model (XGBClassifier): P(A=1|X)
2. Outcome model (XGBRegressor): E[Y|X,A]
3. Final CATE model (XGBRegressor): τ̂(X) = E[Y¹-Y⁰|X]

S-Learner: Single XGBRegressor on [X,A] with interaction depth.
"""

import json
import joblib
import numpy as np
import optuna
from pathlib import Path
from xgboost import XGBRegressor, XGBClassifier  # ✅ CHANGED: XGBoost Imports
from sklearn.model_selection import cross_val_score
from econml.metalearners import SLearner as EconMLSLearner
from econml.dr import DRLearner as EconMLDRLearner

from ecomopti.phase3.config import (
    MODELS_DIR, PROCESSED_DIR, RANDOM_STATE,
    ENSEMBLE_CONFIG
)
from ecomopti.phase3.loader import load_processed_split

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("phase3.train_uplift")

# ------------------------------------------------------------------
# 1.  Shared Optuna objective (XGBoost Adapted)
# ------------------------------------------------------------------
def _xgb_objective(trial: optuna.Trial, X, y, model_role: str) -> float:
    """
    Objective for XGBoost hyper-parameter tuning.
    Customized search spaces for S-Learner vs DR-Learner components.
    """
    
    # Common XGBoost params
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
        # 'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True), # Optional L1
        # 'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True), # Optional L2
    }

    if model_role == "s_learner":
        # S-Learner needs depth to capture interactions between Treatment (A) and Features (X)
        params["max_depth"] = trial.suggest_int("max_depth", 6, 12)
        params["min_child_weight"] = trial.suggest_int("min_child_weight", 1, 5)
        params["gamma"] = trial.suggest_float("gamma", 0, 0.3)
    elif model_role in ["dr_propensity", "dr_outcome"]:
        # Nuisance models: standard capacity
        params["max_depth"] = trial.suggest_int("max_depth", 3, 6)
        params["min_child_weight"] = trial.suggest_int("min_child_weight", 5, 20)
        
    elif model_role == "dr_final":
        # Final CATE model: Conservative to avoid overfitting noise
        params["max_depth"] = trial.suggest_int("max_depth", 2, 5)
        params["min_child_weight"] = trial.suggest_int("min_child_weight", 5, 20)
        
    else:
        raise ValueError(f"Unknown model_role: {model_role}")

    # Cross-validation
    if "propensity" in model_role:
        # Classifier for Propensity
        model = XGBClassifier(**params, eval_metric='logloss')
        score = -np.mean(cross_val_score(
            model, X, y, cv=3, scoring="neg_log_loss", n_jobs=-1))
    else:
        # Regressor for Outcomes/CATE
        model = XGBRegressor(**params)
        score = -np.mean(cross_val_score(
            model, X, y, cv=3, scoring="neg_root_mean_squared_error", n_jobs=-1))

    return score

# ------------------------------------------------------------------
# 2.  Helper: run one Optuna study
# ------------------------------------------------------------------
def _tune_xgb(X, y, model_role: str, n_trials: int = 25) -> dict:
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    study.optimize(
        lambda trial: _xgb_objective(trial, X, y, model_role),
        n_trials=n_trials,
        show_progress_bar=False
    )
    best = study.best_params
    best.update({"random_state": RANDOM_STATE, "n_jobs": -1})
    logger.info(f"Best {model_role} params: {best}")
    return best

# ------------------------------------------------------------------
# 3.  Train each component with tuned hyper-parameters
# ------------------------------------------------------------------
def main():
    logger.info("Loading training data...")
    train_data = load_processed_split("train")
    X, A, Y = train_data["X"], train_data["A"], train_data["Y"]

    # ----------------------------------------------------------
    # 3a.  S-Learner (XGBoost)
    # ----------------------------------------------------------
    logger.info("Tuning S-Learner (XGBoost)...")
    # ✅ FIXED: Add explicit treatment interaction terms
    X_with_interactions = np.column_stack([X, A])

    # Add A * X interaction terms (critical for S-Learner)
    interaction_terms = X * A.reshape(-1, 1)  # Element-wise product
    X_with_interactions = np.column_stack([X_with_interactions, interaction_terms])

    s_params = _tune_xgb(X_with_interactions, Y, "s_learner")
    
    
    # ✅ Instantiating XGBRegressor
    s_model = XGBRegressor(**s_params)
    s_learner = EconMLSLearner(overall_model=s_model)
    s_learner.fit(Y=Y, T=A, X=X)
    
    joblib.dump(s_learner, MODELS_DIR / "s_learner.pkl")
    joblib.dump(s_learner.overall_model, MODELS_DIR / "s_learner_fitted.pkl")
    (MODELS_DIR / "s_learner_best_params.json").write_text(json.dumps(s_params, indent=2))
    logger.info(f"S-Learner training with {X_with_interactions.shape[1]} features (including interactions)")
    # ----------------------------------------------------------
    # 3b.  DR-Learner (XGBoost)
    # ----------------------------------------------------------
    logger.info("Tuning DR-Learner Propensity (XGBoost)...")
    p_params = _tune_xgb(X, A, "dr_propensity")
    prop_model = XGBClassifier(**p_params, eval_metric='logloss') # ✅ XGBClassifier

    logger.info("Tuning DR-Learner Outcome (XGBoost)...")
    o_params = _tune_xgb(X, Y, "dr_outcome")
    out_model = XGBRegressor(**o_params)
    out_model.fit(X, Y)  # Fit explicitly for feature importance check later
    
    logger.info("Tuning DR-Learner CATE (XGBoost)...")
    c_params = _tune_xgb(X, Y, "dr_final")
    cate_model = XGBRegressor(**c_params)

    dr_learner = EconMLDRLearner(
        model_propensity=prop_model,
        model_regression=out_model,
        model_final=cate_model
    )
    dr_learner.fit(Y=Y, T=A, X=X)
    
    joblib.dump(out_model, MODELS_DIR / "dr_outcome_model.pkl")
    joblib.dump(dr_learner, MODELS_DIR / "dr_learner.pkl")
    joblib.dump(dr_learner.model_final, MODELS_DIR / "dr_learner_fitted.pkl")
    (MODELS_DIR / "dr_learner_best_params.json").write_text(
        json.dumps({"propensity": p_params, "outcome": o_params, "cate": c_params}, indent=2)
    )

    # ----------------------------------------------------------
    # 3c.  Generate predictions
    # ----------------------------------------------------------
    logger.info("Generating predictions...")
    for split in ["train", "val", "test"]:
        data = load_processed_split(split)
        for name, model in {"s_learner": s_learner, "dr_learner": dr_learner}.items():
            tau_hat = model.effect(data["X"])
            np.save(MODELS_DIR / f"{name}_pred_{split}.npy", tau_hat)

    # ----------------------------------------------------------
    # 3d.  Ensemble (weighted average of predictions)
    # Weights validated via revenue simulation - DR-Learner dominates
    # ----------------------------------------------------------
    logger.info("Training Ensemble...")
    weights = np.array(ENSEMBLE_CONFIG["weights"])
    weights = weights / weights.sum()
    for split in ["train", "val", "test"]:
        s_pred = np.load(MODELS_DIR / f"s_learner_pred_{split}.npy")
        d_pred = np.load(MODELS_DIR / f"dr_learner_pred_{split}.npy")
        ensemble_pred = weights[0] * s_pred + weights[1] * d_pred
        np.save(MODELS_DIR / f"ensemble_pred_{split}.npy", ensemble_pred)
    
    logger.info(f"✅ Ensemble trained with weights: {dict(zip(['s', 'dr'], weights))}")
    logger.info("✅ All models trained with XGBoost + Optuna")

if __name__ == "__main__":
    main()