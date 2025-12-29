"""Phase 4 Optimizer: Greedy and ILP with business constraints."""

import logging
import numpy as np
import pandas as pd
from typing import Tuple
from .config import BUSINESS_CONFIG, COLS, ILP_TIME_LIMIT, ILP_MAX_ROWS
from .cost_model import calculate_roi_metrics

logger = logging.getLogger("phase4.optimizer")

class OptimizationError(Exception):
    pass

def greedy_optimizer(
    df: pd.DataFrame,
    budget: float,
    enforce_fairness: bool = True
) -> Tuple[pd.DataFrame, dict]:
    
    # CRITICAL: For small datasets, we often have <1000 customers after filtering
    # Runtime is <1ms, making Greedy ideal for real-time use cases
    """Greedy selection with profitability filter and fairness."""
    if budget <= 0:
        raise OptimizationError("Budget must be positive")
    
    # CRITICAL: Filter profitable customers only
    profitable_mask = df[COLS["incremental_value"]] > df[COLS["cost"]]
    df_profit = df[profitable_mask].copy()
    
    if df_profit.empty:
        logger.warning("No profitable customers found")
        empty_df = pd.DataFrame(columns=df.columns)
        return empty_df, {"net_value": 0, "total_cost": 0, "total_customers": 0, "roi_ratio": 0}
    
    # Sort by bang-for-buck ratio
    df_profit = df_profit.sort_values(COLS["bang_per_buck"], ascending=False).reset_index(drop=True)
    df_profit["cumulative_cost"] = df_profit[COLS["cost"]].cumsum()
    
    # Select within budget
    affordable_mask = df_profit["cumulative_cost"] <= budget
    selected_df = df_profit[affordable_mask].copy()
    
    # Enforce fairness constraints
    if enforce_fairness and COLS["segment"] in selected_df.columns:
        min_per_segment = BUSINESS_CONFIG.min_customers_per_segment
        budget_remaining = budget - selected_df[COLS["cost"]].sum()
        
        # For each segment below minimum, add best remaining customers
        for segment in df_profit[COLS["segment"]].unique():
            current_count = (selected_df[COLS["segment"]] == segment).sum()
            
            if current_count < min_per_segment:
                segment_pool = df_profit[
                    (df_profit[COLS["segment"]] == segment) & 
                    ~df_profit.index.isin(selected_df.index)
                ].head(min_per_segment - current_count)
                
                for _, row in segment_pool.iterrows():
                    if budget_remaining >= row[COLS["cost"]]:
                        selected_df = pd.concat([selected_df, row.to_frame().T], ignore_index=True)
                        budget_remaining -= row[COLS["cost"]]
                        current_count += 1
    
    metrics = calculate_roi_metrics(selected_df)
    logger.info(f"Greedy: {metrics['total_customers']} customers, ${metrics['net_value']:,.2f}")
    return selected_df, metrics

def ilp_optimizer(
    df: pd.DataFrame,
    budget: float,
    time_limit: int = ILP_TIME_LIMIT
) -> Tuple[pd.DataFrame, dict]:
    
    # Strategic sampling: If dataset >50K rows, use top-N by bang-per-buck
    # This preserves the most valuable candidates while making ILP tractable
    # Trade-off: Optimal solution within sample, but global optimum not guaranteed

    """ILP exact optimizer with fairness constraints."""
    try:
        import pulp
    except ImportError:
        logger.warning("pulp not installed, falling back to greedy")
        return greedy_optimizer(df, budget)
    
    # Pre-filter profitable customers
    profitable_mask = df[COLS["incremental_value"]] > df[COLS["cost"]]
    df = df[profitable_mask].copy()
    
    if df.empty:
        logger.warning("No profitable customers for ILP")
        return pd.DataFrame(), {"net_value": 0, "total_cost": 0, "total_customers": 0}
    
    # Strategic sampling: take top N by bang-per-buck
    if len(df) > ILP_MAX_ROWS:
        logger.warning(f"Dataset too large ({len(df)} rows), using top {ILP_MAX_ROWS} customers")
        df = df.nlargest(ILP_MAX_ROWS, COLS["bang_per_buck"]).copy()
    
    # Setup ILP
    prob = pulp.LpProblem("MarketingBudget", pulp.LpMaximize)
    x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in df.index}
    
    # Objective: maximize total net value
    prob += pulp.lpSum([
        (row[COLS["incremental_value"]] - row[COLS["cost"]]) * x[i]
        for i, row in df.iterrows()
    ])
    
    # Constraint: budget
    prob += pulp.lpSum([row[COLS["cost"]] * x[i] for i, row in df.iterrows()]) <= budget
    
    # Fairness constraints
    if COLS["segment"] in df.columns:
        min_per_segment = BUSINESS_CONFIG.min_customers_per_segment
        for segment in df[COLS["segment"]].unique():
            segment_idx = df[df[COLS["segment"]] == segment].index
            prob += pulp.lpSum([x[i] for i in segment_idx]) >= min_per_segment
    
    # Solve
    solver = pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=False)
    result = prob.solve(solver)
    
    if pulp.LpStatus[result] not in ["Optimal", "Integer Feasible"]:
        logger.warning(f"ILP status: {pulp.LpStatus[result]}, using greedy fallback")
        return greedy_optimizer(df, budget)
    
    selected_idx = [i for i in df.index if pulp.value(x[i]) > 0.5]
    selected_df = df.loc[selected_idx].copy()
    
    metrics = calculate_roi_metrics(selected_df)
    logger.info(f"ILP: {metrics['total_customers']} customers, ${metrics['net_value']:,.2f}")
    return selected_df, metrics