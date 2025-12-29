"""Phase 4 Reports: Comprehensive reporting and analysis generation."""

import json
import logging
from pathlib import Path
from typing import Dict, Any
import pandas as pd
from datetime import datetime

from .config import REPORTS_DIR, COLS, BUSINESS_CONFIG, DATA_DIR, PLOTS_DIR

logger = logging.getLogger("phase4.reports")

def generate_optimization_report(
    split: str,
    model_hint: str,
    budget: float,
    selected_greedy: pd.DataFrame,
    metrics_greedy: Dict[str, Any],
    selected_ilp: pd.DataFrame,
    metrics_ilp: Dict[str, Any],
    sweep_results: pd.DataFrame = None,
    scenario_results: pd.DataFrame = None,
    comparison_results: pd.DataFrame = None,
    compatibility_checks: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Generate comprehensive JSON report with business metrics and model diagnostics."""
    
    customers_evaluated = len(selected_greedy) + len(selected_ilp)
    
    report = {
        "metadata": {
            "phase": "4",
            "split": split,
            "model": model_hint,
            "budget": float(budget),
            "timestamp": datetime.now().isoformat(),
            "business_config": BUSINESS_CONFIG.dict(),
        },
        "data_summary": {
            "customers_evaluated": int(customers_evaluated),
            "customers_selected_greedy": int(len(selected_greedy)),
            "customers_selected_ilp": int(len(selected_ilp)),
            "clv_range": {
                "min": float(selected_greedy[COLS["clv"]].min()) if not selected_greedy.empty else 0,
                "max": float(selected_greedy[COLS["clv"]].max()) if not selected_greedy.empty else 0,
            },
            "incremental_value_range": {
                "min": float(selected_greedy[COLS["incremental_value"]].min()) if not selected_greedy.empty else 0,
                "max": float(selected_greedy[COLS["incremental_value"]].max()) if not selected_greedy.empty else 0,
            },
        },
        "optimization_results": {
            "greedy": {
                "metrics": metrics_greedy,
                "selection_file": f"selected_{split}_greedy.csv",
            },
            "ilp": {
                "metrics": metrics_ilp,
                "selection_file": f"selected_{split}_ilp.csv",
            },
            "comparison": {
                "net_value_diff": float(metrics_ilp["net_value"] - metrics_greedy["net_value"]),
                "customer_diff": int(metrics_ilp["total_customers"] - metrics_greedy["total_customers"]),
                "roi_diff": float(metrics_ilp["roi_ratio"] - metrics_greedy["roi_ratio"]),
                "ilp_is_better": metrics_ilp["net_value"] > metrics_greedy["net_value"],
            },
        },
        "business_impact": {
            "roi_ratio": float(metrics_ilp["roi_ratio"]),
            "total_clv_impact": float(metrics_ilp["incremental_clv_impact"]),
            "net_profit": float(metrics_ilp["net_value"]),
            "cost_efficiency": float(metrics_ilp["value_per_dollar"]),
            "segment_distribution": metrics_ilp.get("segment_distribution", {}),
            "budget_utilization": float(metrics_ilp["total_cost"] / (budget + 1e-8) * 100),
        },
        "compatibility_checks": compatibility_checks or {"errors": [], "warnings": []},
        "files_generated": {
            "greedy_selection": f"selected_{split}_greedy.csv",
            "ilp_selection": f"selected_{split}_ilp.csv",
            "budget_sweep": f"budget_sweep_{split}.csv" if sweep_results is not None else None,
            "scenario_analysis": f"scenario_analysis_{split}.csv" if scenario_results is not None else None,
            "method_comparison": f"method_comparison_{split}.csv" if comparison_results is not None else None,
        }
    }
    
    return report

def save_report(report: Dict[str, Any], split: str):
    """Save report as JSON and generate markdown summary."""
    
    # Validate report structure before proceeding
    if "optimization_results" not in report:
        logger.error(f"Report structure invalid. Keys: {list(report.keys())}")
        raise ValueError(f"Report missing 'optimization_results' key")
    
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    json_path = REPORTS_DIR / f"report_{split}.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"✓ JSON report saved: {json_path}")
    
    markdown_path = REPORTS_DIR / f"summary_{split}.md"
    _generate_markdown_summary(report, markdown_path, split)
    
    return json_path, markdown_path

def _generate_markdown_summary(report: Dict[str, Any], output_path: Path, split: str):
    """Generate human-readable markdown summary."""
    
    # Double-check report structure
    if "optimization_results" not in report or "ilp" not in report["optimization_results"]:
        logger.error(f"Invalid report structure for markdown generation")
        logger.error(f"Keys: {list(report.keys())}")
        if "optimization_results" in report:
            logger.error(f"optimization_results keys: {list(report['optimization_results'].keys())}")
        raise ValueError("Cannot generate markdown: missing optimization results")
    
    md = f"""# Phase 4 Optimization Report: {report['metadata']['split'].title()} Split
    
**Generated:** {report['metadata']['timestamp']}
**Model:** {report['metadata']['model']}
**Budget:** ${report['metadata']['budget']:,.2f}

## Executive Summary

- **Total Customers Evaluated:** {report['data_summary']['customers_evaluated']:,}
- **Optimal Selection (ILP):** {report['optimization_results']['ilp']['metrics']['total_customers']:,} customers
- **Net Profit:** ${report['business_impact']['net_profit']:,.2f}
- **ROI Ratio:** {report['business_impact']['roi_ratio']:.2f}x
- **Budget Utilization:** {report['business_impact']['budget_utilization']:.1f}%

## Method Comparison

| Metric | Greedy | ILP | Difference |
|--------|--------|-----|------------|
| Customers Selected | {report['optimization_results']['greedy']['metrics']['total_customers']} | {report['optimization_results']['ilp']['metrics']['total_customers']} | {report['optimization_results']['comparison']['customer_diff']:+d} |
| Net Value | ${report['optimization_results']['greedy']['metrics']['net_value']:,.2f} | ${report['optimization_results']['ilp']['metrics']['net_value']:,.2f} | ${report['optimization_results']['comparison']['net_value_diff']:+.2f} |
| ROI Ratio | {report['optimization_results']['greedy']['metrics']['roi_ratio']:.2f}x | {report['optimization_results']['ilp']['metrics']['roi_ratio']:.2f}x | {report['optimization_results']['comparison']['roi_diff']:+.2f}x |

**ILP is {'✅ BETTER' if report['optimization_results']['comparison']['ilp_is_better'] else '❌ WORSE'} than Greedy**

## Business Impact

- **Total CLV Impact:** ${report['business_impact']['total_clv_impact']:,.2f}
- **Treatment Cost:** ${report['optimization_results']['ilp']['metrics']['total_cost']:,.2f}
- **Value per Dollar:** ${report['business_impact']['cost_efficiency']:.2f}
- **Segment Distribution:** {report['business_impact']['segment_distribution']}

## Key Insights

1. **Optimization Efficiency:** The ILP method achieved a net value of ${report['optimization_results']['ilp']['metrics']['net_value']:,.2f} 
   with an ROI of {report['optimization_results']['ilp']['metrics']['roi_ratio']:.2f}x.

2. **Budget Utilization:** {report['business_impact']['budget_utilization']:.1f}% of the allocated budget was used to treat
   {report['optimization_results']['ilp']['metrics']['total_customers']} customers.

3. **Segment Strategy:** The optimal allocation distributed customers across segments as follows:
{chr(10).join([f"   - {seg}: {count} customers" for seg, count in report['business_impact']['segment_distribution'].items()])}

## Files Generated

- Selections: `{DATA_DIR / report['files_generated']['ilp_selection']}`
- Metrics: `{REPORTS_DIR / f'report_{split}.json'}`
- Visualizations: `{PLOTS_DIR}`

---
*Report generated by Phase 4 Optimization Engine*
"""
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)
    
    logger.info(f"✓ Markdown summary saved: {output_path}")

def generate_segment_deep_dive(selected_df: pd.DataFrame, split: str):
    """Generate detailed segment analysis report."""
    
    if COLS["segment"] not in selected_df.columns:
        logger.warning("No segment column for deep dive analysis")
        return None
    
    segment_report = {}
    for segment in selected_df[COLS["segment"]].unique():
        segment_data = selected_df[selected_df[COLS["segment"]] == segment]
        
        segment_report[segment] = {
            "count": len(segment_data),
            "avg_clv": float(segment_data[COLS["clv"]].mean()),
            "avg_uplift": float(segment_data[COLS["tau_hat"]].mean()),
            "total_cost": float(segment_data[COLS["cost"]].sum()),
            "total_incremental_value": float(segment_data[COLS["incremental_value"]].sum()),
            "roi_ratio": float(
                segment_data[COLS["incremental_value"]].sum() / 
                (segment_data[COLS["cost"]].sum() + 1e-8)
            ),
            "bang_per_buck_avg": float(segment_data[COLS["bang_per_buck"]].mean()),
        }
    
    segment_path = REPORTS_DIR / f"segment_deep_dive_{split}.json"
    with open(segment_path, "w") as f:
        json.dump(segment_report, f, indent=2, default=str)
    
    logger.info(f"✓ Segment deep-dive saved: {segment_path}")
    return segment_report

def export_selection_for_crm(selected_df: pd.DataFrame, split: str, model: str, budget: float):
    """Export selection in CRM-ready format."""
    
    if selected_df.empty:
        logger.warning("No selected customers to export")
        return None
    
    crm_export = selected_df[[
        COLS["customer_id"],
        COLS["clv"],
        COLS["tau_hat"],
        COLS["incremental_value"],
        COLS["cost"],
        COLS["segment"]
    ]].copy()
    
    crm_export["treatment_recommendation"] = "target"
    crm_export["priority_score"] = (
        crm_export[COLS["incremental_value"]] / crm_export[COLS["cost"]]
    ).rank(ascending=False, method="dense")
    
    crm_export = crm_export.sort_values("priority_score")
    
    crm_path = DATA_DIR / f"crm_export_{split}_{model}_{int(budget)}.csv"
    crm_export.to_csv(crm_path, index=False)
    
    logger.info(f"✓ CRM export saved: {crm_path} ({len(crm_export)} customers)")
    return crm_path