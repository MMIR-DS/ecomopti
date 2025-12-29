```markdown
# Phase 4: Budget-Constrained Treatment Optimization

**Prerequisites**: Python 3.9+, install dependencies via `pip install -r src/ecomopti/phase4/requirements.txt`

**Run Full Pipeline (Recommended):**  
```bash
python -m ecomopti.phase4.build --split train --model dr_learner --budget 2500
```

**Common Options:**
```bash
# With budget sweep and scenario analysis
python -m ecomopti.phase4.build --split val --model ensemble --sweep --scenarios

# Method comparison (Greedy vs ILP)
python -m ecomopti.phase4.build --split test --model dr_learner --comparison

# Quick real-time run (Greedy only)
python -m ecomopti.phase4.build --split test --model ensemble --budget 1000
```

**Prerequisite:** Phases 1–3 must be completed (CLV from Phase 2, uplift predictions from Phase 3).

**Produced Artifacts:**
- `data/phase4/crm_export_{split}_{model}_{budget}.csv` → CRM-ready target list with priority scores
- `data/phase4/selected_{split}_greedy.csv`, `selected_{split}_ilp.csv` → raw selected customers
- `reports/phase4/report_{split}.json` → comprehensive JSON report
- `reports/phase4/summary_{split}.md` → human-readable Markdown summary
- `reports/phase4/segment_deep_dive_{split}.json` → detailed segment analysis
- `plots/phase4/*.png` → cumulative gain, ROI waterfall, CLV-uplift scatter, budget pie, calibration, segment distribution

**Configuration:** `src/ecomopti/phase4/config.py`

**Tests:** `pytest -q`

---

## Overview

Phase 4 solves the **budget-constrained customer selection problem** to maximize net business value:

**Objective:**  
Maximize Σ (Uplift × CLV) − Treatment Cost  
Subject to total cost ≤ budget

Two solvers are available:
- **Greedy** – near-instant (<1ms), ideal for real-time or large-scale deployment
- **ILP (Integer Linear Programming)** – globally optimal, used for batch reporting (PuLP + CBC)

---

## Example Results (Run: 2025-12-29)

### Train Split – DR-Learner – Budget $2,500
| Method | Customers | Net Value   | Runtime  |
|--------|-----------|-------------|----------|
| Greedy | 465       | $3,395.53   | 0.004s   |
| ILP    | **485**   | **$3,441.89** | 0.367s   |

- **Data Loaded:** 4,930 customers with positive uplift
- **CLV Range:** $92.19 - $689.75
- **Incremental Value Range:** $0.46 - $61.91
- **ILP Advantage:** +$46.36 net value (+20 customers) vs Greedy

### Test Split – Ensemble – Budget $500
| Method | Customers | Net Value   | Runtime  |
|--------|-----------|-------------|----------|
| Greedy | 80        | $641.12     | ~0s      |
| ILP    | 80        | $641.12     | 0.14s    |

→ Greedy and ILP converge on small-to-medium budgets; ILP finds slightly better solutions on larger budgets.

---

## Key Design Decisions

### Business Logic
- **Incremental Value** = Uplift × Predicted 6-month CLV (from Phase 2)
- **Variable Treatment Cost** by customer segment (new/mid/long)
- **Profitability Filter** – only target customers where incremental value > cost
- **Fairness Constraints** – minimum customers per segment (optional in ILP)
- **Risk Adjustment** – conservative uplift using uncertainty estimates (fallback 20% SE)

### Optimization Strategy
- Greedy ranks by **value per dollar** → selects until budget exhausted
- ILP solves exact knapsack with optional fairness and time limits
- Automatic fallback to Greedy if ILP fails or times out

### Reproducibility & Robustness
- Strict Phase 2/3 compatibility checks
- Safe loading with fallbacks and clear error messages
- Comprehensive logging and file verification
- Headless-compatible plotting (Agg backend)

### Directory Structure Created
```
data/phase4/                    ← CRM exports & raw selections
reports/phase4/                 ← JSON reports, Markdown summaries, segment deep-dives
plots/phase4/                   ← ROI waterfall, cumulative gain, CLV-uplift scatter, budget pie, calibration
```

Run Phase 4 to generate actionable customer target lists, financial projections, and stakeholder-ready reports. The CRM export is ready for direct upload to marketing platforms.
```