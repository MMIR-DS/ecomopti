"""Phase 4 Plots: Budget curves, ROI waterfalls, segment analysis."""


import matplotlib
matplotlib.use('Agg')  # Use Agg backend for headless/server environments
import logging
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from .config import PLOTS_DIR, COLS

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
logger = logging.getLogger("phase4.plots")

def _verify_plot_saved(filepath: Path, plot_name: str):
    """Verify file was actually created on disk.
    Helps catch matplotlib backend issues in headless environments."""
    if filepath.exists():
        logger.info(f"✓ {plot_name} saved: {filepath}")
    else:
        logger.error(f"✗ {plot_name} FAILED to save")

def plot_cumulative_gain(df: pd.DataFrame, selected_df: pd.DataFrame):
    """
    Plot Cumulative Gain Curve (The 'Qini-style' efficiency curve for Budget).
    Shows percentage of Total Possible Value captured vs. Percentage of Budget Spent.
    """
    # Sort entire population by efficiency (Bang per Buck)
    df = df.copy()
    if COLS["bang_per_buck"] not in df.columns:
        df[COLS["incremental_value"]] = df[COLS["tau_hat"]] * df[COLS["clv"]]
        df[COLS["bang_per_buck"]] = df[COLS["incremental_value"]] / df[COLS["cost"]]
    
    sorted_df = df.sort_values(COLS["bang_per_buck"], ascending=False).reset_index(drop=True)
    
    # Calculate cumulative sums
    sorted_df["cum_cost"] = sorted_df[COLS["cost"]].cumsum()
    sorted_df["cum_value"] = sorted_df[COLS["incremental_value"]].cumsum()
    
    total_cost = sorted_df[COLS["cost"]].sum()
    total_value = sorted_df[COLS["incremental_value"]].sum()
    
    # Normalize to percentages
    sorted_df["pct_cost"] = sorted_df["cum_cost"] / total_cost
    sorted_df["pct_value"] = sorted_df["cum_value"] / total_value
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    # Random selection baseline (diagonal)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Selection")
    
    # Model selection curve
    plt.plot(sorted_df["pct_cost"], sorted_df["pct_value"], 
             color="#2ecc71", linewidth=2.5, label="Model Selection (Greedy Sort)")
    
    # Mark the cutoff point
    current_budget = selected_df[COLS["cost"]].sum()
    current_pct_cost = current_budget / total_cost
    
    idx = (sorted_df["cum_cost"] - current_budget).abs().idxmin()
    current_pct_value = sorted_df.loc[idx, "pct_value"]
    
    plt.scatter(current_pct_cost, current_pct_value, color="red", zorder=5, s=100, label="Current Budget Limit")
    plt.annotate(f"Budget Cutoff\n({current_pct_cost:.1%} spend captures\n{current_pct_value:.1%} value)", 
                 (current_pct_cost, current_pct_value), 
                 xytext=(current_pct_cost + 0.1, current_pct_value - 0.1),
                 arrowprops=dict(arrowstyle="->", color="black"))

    plt.title("Cumulative Gain: Budget Efficiency", fontsize=14, fontweight="bold")
    plt.xlabel("Fraction of Total Budget Spent")
    plt.ylabel("Fraction of Total Potential Value Captured")
    plt.legend()
    plt.grid(alpha=0.3)
    
    output_path = PLOTS_DIR / "cumulative_gain.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    _verify_plot_saved(output_path, "Cumulative Gain Curve")

def plot_uplift_calibration(df: pd.DataFrame):
    """Check if predicted uplift correlates with actual high-value traits (CLV)."""
    df = df.copy()
    
    # Bin predictions into deciles
    df["uplift_decile"] = pd.qcut(df[COLS["tau_hat"]], q=10, labels=False, duplicates='drop')
    
    # Aggregates
    agg = df.groupby("uplift_decile").agg({
        COLS["tau_hat"]: "mean",
        COLS["clv"]: "mean",
        COLS["incremental_value"]: "mean"
    }).reset_index()
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot Uplift (Line)
    sns.lineplot(data=agg, x="uplift_decile", y=COLS["tau_hat"], 
                 marker="o", color="blue", ax=ax1, label="Avg Predicted Uplift")
    ax1.set_ylabel("Predicted Uplift")
    ax1.set_xlabel("Uplift Decile (0=Low, 9=High)")
    
    # Plot Value (Bar) on secondary axis
    ax2 = ax1.twinx()
    sns.barplot(data=agg, x="uplift_decile", y=COLS["incremental_value"], 
                alpha=0.3, color="green", ax=ax2, label="Avg Incremental Value")
    ax2.set_ylabel("Incremental Value ($)")
    
    plt.title("Model Calibration: Predicted Uplift vs Realized Value", fontsize=14, fontweight="bold")
    
    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper left")
    
    output_path = PLOTS_DIR / "model_calibration.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    _verify_plot_saved(output_path, "Model Calibration Plot")

def plot_budget_efficiency_curve(sweep_results: pd.DataFrame):
    """Plot net value vs budget with efficiency frontier."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    for method in sweep_results["method"].unique():
        data = sweep_results[sweep_results["method"] == method]
        ax1.plot(data["budget"], data["net_value"], marker="o", label=method.title())
        ax2.plot(data["budget"], data["roi_ratio"], marker="s", label=method.title())
    
    ax1.set_xlabel("Budget ($)")
    ax1.set_ylabel("Net Value ($)")
    ax1.set_title("Budget Efficiency")
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    ax2.set_xlabel("Budget ($)")
    ax2.set_ylabel("ROI Ratio")
    ax2.set_title("ROI vs Budget")
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = PLOTS_DIR / "budget_efficiency.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    _verify_plot_saved(output_path, "Budget efficiency curve")

def plot_segment_distribution(selected_df: pd.DataFrame):
    """Show distribution of selected customers across segments."""
    if COLS["segment"] not in selected_df.columns:
        logger.warning("No segment column found, skipping segment distribution plot")
        return
    
    segment_counts = selected_df[COLS["segment"]].value_counts()
    
    plt.figure(figsize=(10, 6))
    segment_counts.plot(kind="bar", color="steelblue", alpha=0.7)
    plt.title("Selected Customers by Segment")
    plt.xlabel("Customer Segment")
    plt.ylabel("Number of Customers")
    plt.xticks(rotation=0)
    plt.grid(axis="y", alpha=0.3)
    
    for i, v in enumerate(segment_counts.values):
        plt.text(i, v + max(segment_counts) * 0.01, str(v), ha="center")
    
    plt.tight_layout()
    output_path = PLOTS_DIR / "segment_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    _verify_plot_saved(output_path, "Segment distribution")

def plot_scenario_comparison(scenario_results: pd.DataFrame):
    """Compare net value across scenarios."""
    plt.figure(figsize=(12, 6))
    
    scenarios = scenario_results["scenario"].values
    net_values = scenario_results["net_value"].values
    
    colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99"]
    bars = plt.bar(scenarios, net_values, color=colors, alpha=0.7)
    
    for bar, value in zip(bars, net_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(net_values) * 0.01,
            f"${value:,.0f}",
            ha="center"
        )
    
    plt.title("Scenario Analysis: Net Value Sensitivity")
    plt.xlabel("Scenario")
    plt.ylabel("Net Value ($)")
    plt.xticks(rotation=45)
    plt.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    output_path = PLOTS_DIR / "scenario_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    _verify_plot_saved(output_path, "Scenario comparison")

def plot_roi_waterfall(metrics: dict):
    """Financial 'Bridge' Chart: Investment -> Returns -> Net Profit."""
    try:
        categories = ["Investment\n(Cost)", "Gross Return\n(Uplift × CLV)", "Net Profit"]
        
        cost = -1 * metrics["total_cost"]
        gross_return = metrics["incremental_clv_impact"]
        net_profit = metrics["net_value"]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Cost Bar (Red)
        p1 = ax.bar(categories[0], cost, color="#e74c3c", alpha=0.8, width=0.5)
        
        # Return Bar (Green)
        p2 = ax.bar(categories[1], gross_return, bottom=cost, color="#2ecc71", alpha=0.8, width=0.5)
        
        # Net Profit Bar (Blue)
        p3 = ax.bar(categories[2], net_profit, color="#3498db", alpha=0.8, width=0.5)
        
        # Add values
        for i, val in enumerate([metrics["total_cost"], gross_return, net_profit]):
            label = f"${val:,.0f}"
            if i == 0: # Cost
                y_pos = cost / 2
                prefix = "-"
            elif i == 1: # Return
                y_pos = cost + (gross_return / 2)
                prefix = "+"
            else: # Net
                y_pos = net_profit / 2
                prefix = "="
            
            ax.text(i, y_pos, f"{prefix}{label}", ha="center", va="center", 
                    color="white", fontweight="bold", fontsize=12)

        # Connector lines
        ax.plot([0, 1], [cost, cost], color="gray", linestyle="--", linewidth=1)
        ax.plot([1, 2], [net_profit, net_profit], color="gray", linestyle="--", linewidth=1)
        
        # Zero line
        ax.axhline(0, color='black', linewidth=1)
        
        # Formatting
        ax.set_title("ROI Bridge Analysis", fontsize=14, fontweight="bold")
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
        ax.grid(axis="y", alpha=0.3)
        
        output_path = PLOTS_DIR / "roi_waterfall.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        _verify_plot_saved(output_path, "ROI Waterfall")

    except Exception as e:
        logger.error(f"Waterfall plot failed: {e}", exc_info=True)

def plot_profitability_curve(sweep_results: pd.DataFrame):
    """Plot net value and ROI efficiency across budget levels."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Filter ILP results
    ilp_data = sweep_results[sweep_results["method"] == "ilp"]
    
    if ilp_data.empty:
        logger.warning("No ILP data for profitability curve")
        return
    
    # Plot 1: Net Value vs Budget
    ax1.plot(ilp_data["budget"], ilp_data["net_value"], 
             marker="o", linewidth=2, markersize=6, color="#2ecc71")
    ax1.set_xlabel("Budget ($)")
    ax1.set_ylabel("Net Value ($)")
    ax1.set_title("Profitability Curve: Net Value vs Budget")
    ax1.grid(alpha=0.3)
    
    # Annotate max point
    max_idx = ilp_data["net_value"].idxmax()
    if not pd.isna(max_idx):
        max_budget = ilp_data.loc[max_idx, "budget"]
        max_net = ilp_data.loc[max_idx, "net_value"]
        ax1.annotate(f'Peak: ${max_net:,.0f}', 
                    xy=(max_budget, max_net), 
                    xytext=(max_budget * 0.7, max_net * 1.1),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                    fontsize=10, fontweight='bold')
    
    # Plot 2: ROI Ratio vs Budget
    ax2.plot(ilp_data["budget"], ilp_data["roi_ratio"], 
             marker="s", linewidth=2, markersize=6, color="#3498db")
    ax2.axhline(y=1.0, color='orange', linestyle='--', alpha=0.7, label='Breakeven ROI')
    ax2.set_xlabel("Budget ($)")
    ax2.set_ylabel("ROI Ratio")
    ax2.set_title("ROI Efficiency vs Budget")
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = PLOTS_DIR / "profitability_curve.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    _verify_plot_saved(output_path, "Profitability curve")

def plot_clv_uplift_scatter(df: pd.DataFrame, selected_df: pd.DataFrame):
    """Scatter plot highlighting selected customers in CLV-Uplift space."""
    plt.figure(figsize=(12, 8))
    
    # Plot all customers
    plt.scatter(df[COLS["clv"]], df[COLS["tau_hat"]], 
               alpha=0.4, s=30, color="gray", label="All Customers")
    
    # Highlight selected
    plt.scatter(selected_df[COLS["clv"]], selected_df[COLS["tau_hat"]], 
               alpha=0.8, s=60, color="red", label="Selected for Treatment")
    
    plt.xlabel("Customer Lifetime Value ($)")
    plt.ylabel("Predicted Uplift (percentage points)")
    plt.title("Treatment Selection: CLV vs Uplift")
    plt.legend()
    plt.grid(alpha=0.3)
    
    output_path = PLOTS_DIR / "clv_uplift_scatter.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    _verify_plot_saved(output_path, "CLV Uplift Scatter")

def plot_budget_allocation_pie(selected_df: pd.DataFrame):
    """Pie chart of budget allocation by segment."""
    if COLS["segment"] not in selected_df.columns:
        logger.warning("No segment column for budget allocation pie")
        return
    
    segment_cost = selected_df.groupby(COLS["segment"], observed=False)[COLS["cost"]].sum()
    
    plt.figure(figsize=(12, 8))
    wedges, texts, autotexts = plt.pie(
        segment_cost.values, 
        labels=segment_cost.index, 
        autopct='%1.1f%%',
        startangle=90,
        colors=sns.color_palette("pastel"),
        explode=[0.05] * len(segment_cost),
        pctdistance=0.85,
        labeldistance=1.1,
        textprops={'fontsize': 11, 'fontweight': 'bold'}
    )
    
    # Add value labels outside pie
    for i, (wedge, cost) in enumerate(zip(wedges, segment_cost.values)):
        ang = (wedge.theta2 - wedge.theta1) / 2. + wedge.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        plt.text(x * 1.3, y * 1.3, f"${cost:,.0f}", 
                ha="center", va="center", fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    
    plt.title("Budget Allocation by Customer Segment", fontsize=16, fontweight='bold', pad=20)
    plt.axis('equal')
    
    output_path = PLOTS_DIR / "budget_allocation_pie.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    _verify_plot_saved(output_path, "Budget allocation pie")