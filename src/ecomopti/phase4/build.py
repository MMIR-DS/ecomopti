"""Phase 4 Build: Main orchestration with CLI entry point and compatibility checks."""
# For small datasets (<5K rows), ILP runtime is <1s
# Greedy and ILP often produce identical results on test/val splits
# Decision: Use Greedy for real-time, ILP for batch reporting

import logging
import pandas as pd
from pathlib import Path
import time
import argparse
import warnings
from .config import (
    BUSINESS_CONFIG, 
    BUDGET_SWEEP_RANGE, 
    COLS, 
    PLOTS_DIR, 
    DATA_DIR, 
    REPORTS_DIR,
    verify_phase_compatibility
)
from .loader import load_data_for_optimization, DataLoadingError
from .optimizer import greedy_optimizer, ilp_optimizer
from .evaluator import run_budget_sweep, run_scenario_analysis, run_method_comparison
from . import plots
from . import reports

warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*found in sys.modules.*")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("phase4.build")

def run_optimization_workflow(
    split: str = "val",
    model_hint: str = "dr_learner",
    budget: float = None,
    run_sweep: bool = True,
    run_scenarios: bool = True,
    run_comparison: bool = True
):
    """Execute complete Phase 4 workflow."""
    budget = budget or BUSINESS_CONFIG.target_budget
    
    logger.info("=" * 60)
    logger.info("PHASE 4: BUDGET-CONSTRAINED OPTIMIZATION")
    logger.info("=" * 60)
    logger.info(f"Budget: ${budget:,.2f} | Model: {model_hint} | Split: {split}")
    
    # Verify compatibility before proceeding
    logger.info("Verifying phase compatibility...")
    compat = verify_phase_compatibility(split, model_hint)
    if compat["errors"]:
        raise DataLoadingError(f"Phase compatibility check failed: {compat['errors']}")
    for w in compat["warnings"]:
        logger.warning(w)
    
    # Step 1: Load data
    logger.info("Loading data...")
    df = load_data_for_optimization(split, model_hint)

    # ✅ Budget sanity check: prevents grossly oversized budgets
    # If budget > 15% of total possible value, warn user
    total_possible_value = df[COLS["incremental_value"]].sum()
    if budget > total_possible_value / BUSINESS_CONFIG.max_budget_utilization:
        logger.warning(
            f"Budget ${budget:,.0f} exceeds {BUSINESS_CONFIG.max_budget_utilization:.0%} "
            f"of total possible value (${total_possible_value:,.0f}). "
            f"Recommend budget < ${total_possible_value * 0.2:,.0f}"
        )

    # Step 2: Optimize (both methods)
    
    logger.info("Running optimizations...")
    start_time = time.time()
    selected_greedy, metrics_greedy = greedy_optimizer(df, budget)
    greedy_time = time.time() - start_time
    
    start_time = time.time()
    selected_ilp, metrics_ilp = ilp_optimizer(df, budget)
    ilp_time = time.time() - start_time
    
    logger.info(f"Greedy: {metrics_greedy['total_customers']} customers, ${metrics_greedy['net_value']:,.2f} in {greedy_time:.3f}s")
    logger.info(f"ILP: {metrics_ilp['total_customers']} customers, ${metrics_ilp['net_value']:,.2f} in {ilp_time:.3f}s")
    
    # Step 3: Save selections
    if len(selected_greedy) > 0:
        selected_greedy.to_csv(DATA_DIR / f"selected_{split}_greedy.csv", index=False)
    if len(selected_ilp) > 0:
        selected_ilp.to_csv(DATA_DIR / f"selected_{split}_ilp.csv", index=False)
    logger.info("✓ Saved selections to data/phase4/")
    
    # Step 4: Budget sweep
    sweep_results = None
    if run_sweep:
        logger.info("Running budget sweep...")
        sweep_results = run_budget_sweep(df, budgets=BUDGET_SWEEP_RANGE, methods=["greedy", "ilp"])
        sweep_results.to_csv(DATA_DIR / f"budget_sweep_{split}.csv", index=False)
        plots.plot_budget_efficiency_curve(sweep_results)
        logger.info("✓ Budget sweep complete")
    
    # Step 5: Scenario analysis
    scenario_results = None
    if run_scenarios:
        logger.info("Running scenario analysis...")
        scenario_results = run_scenario_analysis(df, budget)
        scenario_results.to_csv(DATA_DIR / f"scenario_analysis_{split}.csv", index=False)
        plots.plot_scenario_comparison(scenario_results)
        logger.info("✓ Scenario analysis complete")
    
    # Step 6: Method comparison
    comparison_results = None
    if run_comparison:
        logger.info("Comparing optimization methods...")
        comparison_results = run_method_comparison(df, budget)
        comparison_results.to_csv(DATA_DIR / f"method_comparison_{split}.csv", index=False)
        logger.info("✓ Method comparison complete")
    
    # Step 7: Advanced visualizations (using ILP as primary)
    if len(selected_ilp) > 0:
        logger.info("Generating advanced plots...")
        plots.plot_segment_distribution(selected_ilp)
        
        # Use ILP sweep data for profitability curve
        if run_sweep:
            ilp_sweep = run_budget_sweep(df, budgets=BUDGET_SWEEP_RANGE, methods=["ilp"])
            plots.plot_profitability_curve(ilp_sweep)
        
        plots.plot_clv_uplift_scatter(df, selected_ilp)
        plots.plot_budget_allocation_pie(selected_ilp)
        plots.plot_roi_waterfall(metrics_ilp)
        plots.plot_cumulative_gain(df, selected_ilp)
        plots.plot_uplift_calibration(df)
        
        logger.info("✓ Advanced plots generated")
    
    # Step 8: Generate comprehensive reports
    logger.info("Generating reports...")
    report = reports.generate_optimization_report(
        split=split,
        model_hint=model_hint,
        budget=budget,
        selected_greedy=selected_greedy,
        metrics_greedy=metrics_greedy,
        selected_ilp=selected_ilp,
        metrics_ilp=metrics_ilp,
        sweep_results=sweep_results,
        scenario_results=scenario_results,
        comparison_results=comparison_results,
        compatibility_checks=compat
    )
    if not isinstance(report, dict) or "optimization_results" not in report:
        logger.error(f"Invalid report generated: {type(report)}")
        raise RuntimeError("Report generation failed")
    
    # Save main report
    json_path, markdown_path = reports.save_report(report, split)
    
    # Generate additional reports
    reports.generate_segment_deep_dive(selected_ilp, split)
    reports.export_selection_for_crm(selected_ilp, split, model_hint, budget)
    logger.info("✓ Comprehensive reports saved")
    logger.info("=" * 60)
    logger.info("✅ PHASE 4 COMPLETE")
    logger.info("=" * 60)
    
    return report

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Phase 4 Optimization")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--model", type=str, default="dr_learner", help="dr_learner, s_learner, ensemble")
    parser.add_argument("--budget", type=float, default=None, help="Marketing budget (uses config if not specified)")
    parser.add_argument("--sweep", action="store_true", help="Run budget sweep analysis")
    parser.add_argument("--scenarios", action="store_true", help="Run scenario analysis")
    parser.add_argument("--comparison", action="store_true", help="Compare Greedy vs ILP")
    
    args = parser.parse_args()
    logger.info(f"CLI Args: {vars(args)}")
    
    try:
        run_optimization_workflow(
            split=args.split,
            model_hint=args.model,
            budget=args.budget,
            run_sweep=args.sweep,
            run_scenarios=args.scenarios,
            run_comparison=args.comparison
        )
    except DataLoadingError as e:
        logger.error(f"Fatal loading error: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        exit(1)

if __name__ == "__main__":
    main()