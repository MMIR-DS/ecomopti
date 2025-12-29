"""Phase 4 Evaluator: Budget sweep, scenario analysis, optimization comparison."""

import logging
import time
from typing import List, Dict
from .config import BUSINESS_CONFIG, BUDGET_SWEEP_RANGE, COLS
from .optimizer import greedy_optimizer, ilp_optimizer
import pandas as pd

logger = logging.getLogger("phase4.evaluator")

def run_budget_sweep(
    df: pd.DataFrame,
    budgets: List[float] = None,
    methods: List[str] = None
) -> pd.DataFrame:
    """Evaluate performance across multiple budgets and optimization methods."""
    budgets = budgets or BUDGET_SWEEP_RANGE
    methods = methods or ["greedy", "ilp"]
    
    # Check pulp availability before adding ILP
    if "ilp" in methods:
        try:
            import pulp
        except ImportError:
            logger.warning("pulp not available, removing ILP from sweep")
            methods = ["greedy"]
    
    results = []
    
    for method in methods:
        logger.info(f"Running budget sweep for {method}")
        for budget in budgets:
            if method == "greedy":
                selected_df, metrics = greedy_optimizer(df, budget, enforce_fairness=False)
            else:
                selected_df, metrics = ilp_optimizer(df, budget)
            
            # Calculate segment distribution safely
            segment_dist = metrics.get("segment_distribution", {})
            segment_cols = {f"{seg}_count": segment_dist.get(seg, 0) 
                          for seg in df[COLS["segment"]].unique()}
            
            results.append({
                "method": method,
                "budget": budget,
                "customers_selected": metrics["total_customers"],
                "total_cost": metrics["total_cost"],
                "net_value": metrics["net_value"],
                "roi_ratio": metrics["roi_ratio"],
                **segment_cols
            })
    
    return pd.DataFrame(results)

def run_scenario_analysis(df: pd.DataFrame, budget: float) -> pd.DataFrame:
    """Analyze sensitivity to different business scenarios."""
    
    # Helper to recompute sorting metrics after modifying inputs
    def _recalc_metrics(d):
        d_copy = d.copy()
        d_copy[COLS["incremental_value"]] = d_copy[COLS["tau_hat"]] * d_copy[COLS["clv"]]
        d_copy[COLS["bang_per_buck"]] = d_copy[COLS["incremental_value"]] / (d_copy[COLS["cost"]] + 1e-9)
        return d_copy

    # Each scenario gets its own isolated DataFrame copy
    scenarios = {
        "base_case": _recalc_metrics(df.copy()),
        
        "optimistic_uplift": _recalc_metrics(df.copy().assign(**{
            COLS["tau_hat"]: lambda x: x[COLS["tau_hat"]] * 1.2
        })),
        
        "pessimistic_uplift": _recalc_metrics(df.copy().assign(**{
            COLS["tau_hat"]: lambda x: x[COLS["tau_hat"]] * 0.8
        })),
        
        "high_cost": _recalc_metrics(df.copy().assign(**{
            COLS["cost"]: lambda x: x[COLS["cost"]] * 1.5
        })),
        
        "best_case_segment": _recalc_metrics(
            df[df[COLS["clv"]] >= df[COLS["clv"]].quantile(0.8)].copy()
        )
    }
    
    results = []
    for scenario_name, df_scenario in scenarios.items():
        if len(df_scenario) == 0:
            logger.warning(f"Scenario {scenario_name} has no data, skipping")
            continue
        
        selected_df, metrics = greedy_optimizer(df_scenario, budget, enforce_fairness=False)
        
        results.append({
            "scenario": scenario_name,
            "net_value": metrics["net_value"],
            "customers_selected": metrics["total_customers"],
            "roi_ratio": metrics["roi_ratio"],
            "avg_uplift": df_scenario[COLS["tau_hat"]].mean(),
            "avg_incremental_value": df_scenario[COLS["incremental_value"]].mean()
        })
    
    return pd.DataFrame(results)

def run_method_comparison(df: pd.DataFrame, budget: float) -> pd.DataFrame:
    """Head-to-head comparison of Greedy vs ILP."""
    logger.info("Comparing optimization methods...")
    
    results = []
    for method_name, optimizer in [("greedy", greedy_optimizer), ("ilp", ilp_optimizer)]:
        start_time = time.time()
        try:
            selected_df, metrics = optimizer(df, budget)
            runtime = time.time() - start_time
            
            results.append({
                "method": method_name,
                "customers": metrics["total_customers"],
                "net_value": metrics["net_value"],
                "roi": metrics["roi_ratio"],
                "time_seconds": round(runtime, 3),
                "is_optimal": method_name == "ilp",
                "status": "success"
            })
        except Exception as e:
            logger.error(f"{method_name} failed: {e}")
            results.append({
                "method": method_name,
                "customers": 0,
                "net_value": 0,
                "roi": 0,
                "time_seconds": 0,
                "is_optimal": False,
                "status": f"error: {str(e)}"
            })
    
    return pd.DataFrame(results)