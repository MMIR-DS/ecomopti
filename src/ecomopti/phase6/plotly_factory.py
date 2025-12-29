# plotly_factory.py - Complete Visualization Factory
from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import logging

logger = logging.getLogger(__name__)

# ==============================================================================
# HELPER: AUTO-INSIGHTS & NARRATIVE
# ==============================================================================
def generate_insight(metric_name: str, value: float, context: str = "") -> str:
    """Generates one-sentence executive takeaways."""
    if metric_name == "roi":
        if value > 3: return f"💡 High Efficiency: For every $1 spent, you get ${value:.2f} back."
        if value > 1: return f"💡 Profitable: Returns exceed costs by {(value-1)*100:.0f}%."
        return "⚠️ Low Efficiency: Returns are currently below break-even."

    if metric_name == "capture_rate":
        return f"💡 Pareto Principle: This budget captures {value:.1f}% of all available opportunities."

    return context

def generate_recommendation(n_customers: int, primary_segment: str, net_val: float) -> str:
    """Generates the 'So What?' business action sentence."""
    return (f"🚀 Recommended Action: Target these {n_customers:,} customers "
            f"(skewing towards {primary_segment}) to generate an estimated "
            f"${net_val:,.0f} in net profit.")

def empty_fig(title: str, msg: str, height: int = 420) -> go.Figure:
    """Empty figure placeholder for missing data"""
    fig = go.Figure()
    fig.update_layout(
        title=title,
        height=height,
        annotations=[dict(
            text=msg, x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(size=14, color="gray")
        )],
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig

def read_csv_safe(path: Path) -> Optional[pd.DataFrame]:
    """Safely read CSV with existence and emptiness checks"""
    try:
        if path.exists():
            df = pd.read_csv(path)
            return df if not df.empty else None
    except Exception as e:
        logging.warning(f"Failed to read {path}: {e}")
        return None
    return None

def add_value_cols(df: pd.DataFrame, treatment_cost: float) -> pd.DataFrame:
    """
    Add value columns to DataFrame.
    Handles empty DataFrames and invalid cost values.
    """
    out = df.copy()
    out["clv"] = pd.to_numeric(out.get("clv", 0.0), errors="coerce").fillna(0.0)
    out["tau_hat"] = pd.to_numeric(out.get("tau_hat", 0.0), errors="coerce").fillna(0.0)

    # Guard against invalid cost
    cost = float(treatment_cost) if treatment_cost and treatment_cost > 0 else 5.0
    out["treatment_cost"] = cost

    out["incremental_value"] = out["clv"] * out["tau_hat"]
    out["net_value_unit"] = out["incremental_value"] - cost
    return out

# ==============================================================================
# PLOTS
# ==============================================================================

def budget_sweep_figure(sweep_csv: Path, selected_budget: Optional[float] = None) -> go.Figure:
    """Budget sensitivity analysis figure"""
    df = read_csv_safe(sweep_csv)
    if df is None:
        return empty_fig("Budget Sensitivity", "Run Phase 4 sweep to generate data.")

    cols = {c.lower(): c for c in df.columns}
    b_col = cols.get("budget") or cols.get("budgets")
    net_col = cols.get("net_value") or cols.get("net_profit")
    roi_col = cols.get("roi_ratio") or cols.get("roi")

    if not b_col or not net_col:
        return empty_fig("Budget Sensitivity", "Missing required columns.")

    df = df.sort_values(b_col).reset_index(drop=True)
    df[b_col] = pd.to_numeric(df[b_col], errors="coerce")
    df = df[(df[b_col] >= 500) & (df[b_col] <= 10000)]

    max_net = df[net_col].max()
    opt_budget = df.loc[df[net_col].idxmax(), b_col]
    insight = f"💡 Peak Profit: ${max_net:,.0f} at ${opt_budget:,.0f} budget."

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(x=df[b_col], y=df[net_col], name="Net Value ($)", marker_color="#01B8AA", opacity=0.7),
        secondary_y=False
    )

    if roi_col:
        fig.add_trace(
            go.Scatter(x=df[b_col], y=df[roi_col], mode="lines+markers", name="ROI (x)",
                      line=dict(color="#374649", width=3)),
            secondary_y=True
        )
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="Breakeven", secondary_y=True)

    if selected_budget is not None:
        fig.add_vline(x=float(selected_budget), line_dash="dot", line_color="#FD625E")

    # Move insight to left-aligned annotation to prevent overlap
    fig.update_layout(
    title=dict(text="Budget Sensitivity", x=0.02, xanchor="left", font=dict(size=16)),
    margin=dict(t=100, b=40, l=40, r=40),  # Increased top margin
    height=400, 
    legend=dict(orientation="h", y=1.08),
    plot_bgcolor="white",
    annotations=[
        dict(
            text=insight,
            x=0.02, y=1.0,
            xref="paper", yref="paper",
            xanchor="left", yanchor="bottom",
            showarrow=False,
            font=dict(size=11, color="#374649"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(128,128,128,0.3)",
            borderwidth=1
            )
        ]
    )   
    fig.update_yaxes(title_text="Net Value ($)", secondary_y=False, gridcolor="#f0f0f0")
    fig.update_yaxes(title_text="ROI (x)", secondary_y=True, showgrid=False)

    return fig

def cumulative_gain_figure(df_all: pd.DataFrame, budget: float, treatment_cost: float) -> go.Figure:
    """
    Cumulative gain (Qini) curve showing value capture vs. random baseline.
    
    Key features:
    - Handles negative uplift (sleeping dogs) using absolute value normalization
    - Decimates large datasets (>2000 rows) for performance while preserving the budget cutoff point
    - Marks budget cutoff precisely on the decimated curve
    
    Args:
        df_all: DataFrame with 'clv' and 'tau_hat' columns from Phase 2/3
        budget: Maximum campaign budget (e.g., 2500.0)
        treatment_cost: Cost per customer treatment (e.g., 5.0)
    
    Returns:
        Plotly figure object with model curve, random baseline, and budget cutoff marker
    """
    # ── VALIDATION ─────────────────────────────────────────────────────────────
    if df_all.empty:
        logger.warning("Empty DataFrame provided to cumulative_gain_figure")
        return empty_fig("Cumulative Gain", "No data available.")
    
    required_cols = {"clv", "tau_hat"}
    if not required_cols.issubset(df_all.columns):
        missing = required_cols - set(df_all.columns)
        logger.warning(f"Missing required columns: {missing}")
        return empty_fig("Cumulative Gain", f"Missing data: {missing}")

    # ── DATA PREPARATION ───────────────────────────────────────────────────────
    # Calculate incremental value per customer and sort descending (optimal targeting order)
    df = add_value_cols(df_all, treatment_cost=treatment_cost)
    df = df.sort_values("incremental_value", ascending=False).reset_index(drop=True)
    
    # Calculate cumulative value and cost curves
    df["cum_value"] = df["incremental_value"].cumsum()
    df["cum_cost"] = df["treatment_cost"].cumsum()

    # ── NORMALIZATION STRATEGY ────────────────────────────────────────────────
    # Use absolute total value as denominator to properly handle negative uplift values.
    # This allows the curve to dip below zero, visually representing the cost of targeting 
    # "sleeping dogs" (customers with negative treatment effects).
    total_abs_value = df["incremental_value"].abs().sum()
    if total_abs_value <= 0:
        logger.warning("Total absolute value is zero - cannot normalize")
        return empty_fig("Cumulative Incremental Value", "No incremental value in dataset.")

    # ── BUDGET CUTOFF CALCULATION ─────────────────────────────────────────────
    # Find the last customer index we can afford within budget
    # searchsorted returns count of rows where cum_cost <= budget; convert to 0-based index
    affordable_count = df["cum_cost"].searchsorted(budget, side='right')
    affordable_idx = min(affordable_count - 1, len(df) - 1)
    
    logger.info(f"Budget ${budget:,.2f} affords {affordable_count} customers (last index: {affordable_idx})")

    # ── DECIMATION FOR PERFORMANCE ───────────────────────────────────────────
    # For large datasets, sample ~2000 points to improve rendering speed
    if len(df) > 2000:
        step = len(df) // 1999  # Sampling interval
        sampled_positions = list(range(0, len(df), step))
        
        # CRITICAL: Ensure the cutoff point is included in the decimated dataset
        # so we can place the marker accurately
        if affordable_idx not in sampled_positions:
            sampled_positions.append(affordable_idx)
            sampled_positions.sort()
        
        # Create plot DataFrame and track the cutoff's position within it
        plot_df = df.iloc[sampled_positions].reset_index(drop=True)
        cutoff_pos_in_plot = sampled_positions.index(affordable_idx)
    else:
        plot_df = df.copy()
        cutoff_pos_in_plot = affordable_idx

    # ── PERCENTAGE COORDINATES ────────────────────────────────────────────────
    # Convert cumulative value to percentage of total absolute value
    x_pct = np.linspace(0, 100, len(plot_df))
    y_pct = (plot_df["cum_value"] / total_abs_value) * 100
    
    # Extract cutoff coordinates for the marker
    cutoff_x = (cutoff_pos_in_plot / len(plot_df)) * 100
    cutoff_y = y_pct.iloc[cutoff_pos_in_plot]

    # ── VISUALIZATION ─────────────────────────────────────────────────────────
    

    fig = go.Figure()
    
    # Model performance curve
    fig.add_trace(go.Scatter(
        x=x_pct, y=y_pct, mode="lines", name="Model (Qini)",
        line=dict(color="#01B8AA", width=3),
        hovertemplate="%{x:.1f}% targeted<br>%{y:.1f}% value captured<extra></extra>"
    ))
    
    # Random baseline: expected value when targeting randomly
    random_mean_value = df["incremental_value"].mean()
    random_cum_value = random_mean_value * np.arange(1, len(plot_df) + 1)
    y_random = (random_cum_value / total_abs_value) * 100
    
    # Random baseline (diagonal)
    fig.add_trace(go.Scatter(
        x=x_pct, y=y_random, mode="lines", name="Random Baseline",
        line=dict(dash="dash", color="gray"),
        hovertemplate="Random: %{y:.1f}% value<extra></extra>"
    ))

    # Budget cutoff marker
    fig.add_trace(go.Scatter(
        x=[cutoff_x], y=[cutoff_y], mode="markers", name="Budget Cap",
        marker=dict(color="#FD625E", size=10, line=dict(width=2, color="white")),
        hovertemplate=f"Budget: ${budget:,.0f}<br>Capture: {cutoff_y:.1f}%<extra></extra>"
    ))

    # Calculate insight text for left annotation (AFTER y_random is defined)
    insight_text = f"💡 Capture Rate: {cutoff_y:.1f}% vs {y_random[cutoff_pos_in_plot]:.1f}% random"

    # Layout with left-aligned title and separate insight annotation
    fig.update_layout(
        title=dict(
            text="Cumulative Incremental Value",
            x=0.02,  # ✅ Left-aligned (was 0.5/center)
            xanchor="left",
            font=dict(size=16)
        ),
        xaxis_title="% Population Targeted",
        yaxis_title="% Absolute Value Captured",
        height=400,
        margin=dict(t=100, b=40, l=40, r=40),  # ✅ Increased top margin for annotation
        plot_bgcolor="white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        # ✅ Add left-aligned annotation for insight
        annotations=[
            dict(
                text=insight_text,
                x=0.02,  # ✅ Left side (x=0 is far left, x=1 is far right)
                y=1.0,   # ✅ Top of plot area
                xref="paper",
                yref="paper",
                xanchor="left",
                yanchor="bottom",
                showarrow=False,
                font=dict(size=11, color="#374649"),
                bgcolor="rgba(255,255,255,0.8)",  # ✅ Semi-transparent background
                bordercolor="rgba(128,128,128,0.3)",
                borderwidth=1
            )
        ]
    )
    
    # Add subtle gridlines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    
    return fig


def uplift_calibration_figure(df_all: pd.DataFrame) -> go.Figure:
    """Uplift calibration plot (decile-based)"""
    if df_all.empty or "true_uplift" not in df_all.columns:
        return empty_fig("Calibration", "Missing 'true_uplift' data. Using simulated ground truth.")

    df = df_all.copy()
    df["decile"] = pd.qcut(df["tau_hat"], 10, labels=False, duplicates="drop")

    cal = df.groupby("decile").agg({"tau_hat": "mean", "true_uplift": "mean"}).reset_index()

    min_val = min(cal["tau_hat"].min(), cal["true_uplift"].min())
    max_val = max(cal["tau_hat"].max(), cal["true_uplift"].max())

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode="lines", name="Perfect", line=dict(dash="dash", color="gray")
    ))
    fig.add_trace(go.Scatter(
        x=cal["tau_hat"], y=cal["true_uplift"],
        mode="lines+markers", name="Model Actual",
        marker=dict(size=10, color="#01B8AA")
    ))

    corr = np.corrcoef(cal["tau_hat"], cal["true_uplift"])[0,1]

    note = "Note: Uses simulated ground truth for calibration."
    insight = f"💡 Trust Score: {corr:.2f} correlation.<br><span style='font-size:10px;color:red'>{note}</span>"

    fig.update_layout(
        title=dict(text=f"Uplift Calibration (By Decile)<br>{insight}"),
        xaxis_title="Predicted Uplift", yaxis_title="Actual Uplift",
        height=400, margin=dict(t=80), plot_bgcolor="white"
    )
    return fig

def strategy_matrix(df_all: pd.DataFrame, df_sel: Optional[pd.DataFrame]) -> go.Figure:
    """CLV vs Uplift scatter plot with sleeping dogs highlighting"""
    if df_all.empty or not {"clv", "tau_hat"}.issubset(df_all.columns):
        return empty_fig("Strategy Matrix", "Missing CLV/Uplift data.")

    df_dogs = df_all[df_all["tau_hat"] < 0]
    df_wins = df_all[df_all["tau_hat"] >= 0]

    # Stratified Downsampling
    sample_dogs = df_dogs.sample(n=min(len(df_dogs), 500), random_state=42)
    sample_wins = df_wins.sample(n=min(len(df_wins), 2000), random_state=42)
    plot_df = pd.concat([sample_dogs, sample_wins])

    pct_dogs = (len(df_dogs) / len(df_all)) * 100
    insight = f"💡 Strategy: Avoid the {pct_dogs:.1f}% of customers (red) who react negatively."

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_df["clv"], y=plot_df["tau_hat"],
        mode="markers", name="Population",
        marker=dict(color=np.where(plot_df["tau_hat"]<0, "#FD625E", "lightgray"), opacity=0.4, size=6)
    ))

    if df_sel is not None and not df_sel.empty:
        safe_selected = df_sel[df_sel["tau_hat"] >= 0]
        if not safe_selected.empty:
            fig.add_trace(go.Scatter(
                x=safe_selected["clv"], y=safe_selected["tau_hat"],
                mode="markers", name="Selected Targets",
                marker=dict(color="#01B8AA", size=8, line=dict(width=1, color="white"))
            ))

    fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.5)
    fig.update_layout(
        title=dict(text=f"Strategy Matrix<br><span style='font-size:12px;color:gray'>{insight}</span>"),
        xaxis_title="Customer Lifetime Value (CLV)", yaxis_title="Uplift (Persuadability)",
        height=450, margin=dict(t=60), plot_bgcolor="white"
    )
    return fig

def waterfall_figure(incremental_value: float, total_cost: float) -> go.Figure:
    """ROI waterfall chart"""
    net = incremental_value - total_cost
    roi = (incremental_value / total_cost) if total_cost > 0 else 0
    insight = generate_insight("roi", roi)

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["relative", "relative", "total"],
        x=["Invested Budget", "Gross Return", "Net Profit"],
        text=[f"-${total_cost:,.0f}", f"+${incremental_value:,.0f}", f"${net:,.0f}"],
        y=[-total_cost, incremental_value, net],
        connector={"line": {"color": "gray"}},
        decreasing={"marker": {"color": "#FD625E"}},
        increasing={"marker": {"color": "#01B8AA"}},
        totals={"marker": {"color": "#374649"}}
    ))
    fig.update_layout(
        title=dict(text=f"Financial Bridge<br><span style='font-size:12px;color:gray'>{insight}</span>"),
        height=400
    )
    return fig

def feature_importance_figure(df_all: pd.DataFrame, model: str = "s_learner") -> go.Figure:
    """
    Load REAL Phase 3 feature importance for the SELECTED model.
    For Ensemble, uses DR-Learner importance (70% weight) as representative.
    """
    from ecomopti.phase6.app import load_feature_importance
    
    # Map ensemble to dr_learner since ensemble has no direct importance
    actual_model = "dr_learner" if model == "ensemble" else model
    df = load_feature_importance(actual_model)
    
    if df.empty:
        return empty_fig("Feature Importance", f"No importance data found for {model}. Run Phase 3 training first.")
    
    # Use real data from Phase 3
    fig = go.Figure(go.Bar(
        x=df["importance"],
        y=df["feature"],
        orientation="h",
        marker_color="#01B8AA"
    ))

    # Build title components
    top_feature = df.iloc[0]["feature"] if not df.empty else "N/A"
    model_display = model.replace("_", " ").title()
    insight = f"💡 Top Feature: {top_feature}"
    
    # ✅ REMOVE NOTE ENTIRELY - single-line clean title
    fig.update_layout(
        title=dict(
            text=f"Feature Importance ({model_display})<br>"
                 f"<span style='font-size:12px;color:gray'>{insight}</span>"
        ),
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        height=400,
        margin=dict(t=60),  # Reduced top margin since no subtitle
        plot_bgcolor="white"
    )
    
    return fig

def scenario_analysis_figure(df_sel: pd.DataFrame, default_cost:float) -> go.Figure:
    """
    Scenario analysis showing optimistic, base, pessimistic cases
    
    Args:
        df_sel: Selected customers DataFrame
        default_cost: Uniform treatment cost for sensitivity analysis
    """
    if df_sel.empty:
        return empty_fig("Scenario Analysis", "No selected data")

    df = add_value_cols(df_sel.copy(), default_cost)
    base_value = df["incremental_value"].sum()
    base_cost = len(df) *  default_cost
    base_net = base_value - base_cost

    scenarios = {
        "Optimistic (+20%)": base_value * 1.2 - base_cost,
        "Base Case": base_net,
        "Pessimistic (-20%)": base_value * 0.8 - base_cost,
        "High Cost (+50%)": base_value - base_cost * 1.5
    }

    fig = go.Figure(go.Bar(
        x=list(scenarios.keys()),
        y=list(scenarios.values()),
        marker_color=["#01B8AA", "#374649", "#FD625E", "#FFB6C1"]
    ))

    fig.update_layout(
        title="Scenario Analysis",
        yaxis_title="Net Profit ($)",
        height=400,
        margin=dict(t=60),
        plot_bgcolor="white"
    )

    # Add breakeven line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Breakeven")

    return fig