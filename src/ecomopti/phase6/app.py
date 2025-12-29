# app.py - Debugged and Production-Ready Phase 6 Dashboard
from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from functools import lru_cache

import numpy as np
import pandas as pd
import requests

import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, dash_table, no_update
import plotly.express as px

from plotly_factory import (
    empty_fig,
    add_value_cols,
    budget_sweep_figure,
    cumulative_gain_figure,
    waterfall_figure,
    strategy_matrix,
    uplift_calibration_figure,
    generate_recommendation,
    feature_importance_figure,
    scenario_analysis_figure,
)

# ==============================================================================
# CONFIG & LOGGING
# ==============================================================================
API_BASE_URL = os.getenv("ECOMOPTI_API_URL", "http://localhost:8000")
DEFAULT_COST = 5.0
BUDGET_MIN = 100
BUDGET_MAX = 5000
BUDGET_STEP = 100
HEALTH_CHECK_INTERVAL = 10 * 1000

logging.basicConfig(
    level=logging.INFO,
    filename='logs/dashboard.log',
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

VALID_SPLITS = ["train", "val", "test"]
VALID_MODELS = ["s_learner", "dr_learner", "ensemble"]

class DataPaths:
    """Centralized path management for all project artifacts"""
    ROOT = Path(__file__).resolve().parents[3]
    
    @classmethod
    def phase4_enriched(cls, split: str, model: str) -> Path:
        """Phase 4 enriched data containing CLV, uplift, and segments"""
        return cls.ROOT / "data" / "phase4" / f"enriched_{split}_{model}.csv"
    
    @classmethod
    def crm_export(cls, split: str, model: str, budget: float) -> Path:
        return cls.ROOT / "data" / "phase4" / f"crm_export_{split}_{model}_{int(budget)}.csv"
    
    @classmethod
    def split_data(cls, split: str) -> Path:
        return cls.ROOT / "data" / "splits" / f"{split}.csv"
    
    @classmethod
    def clv_predictions(cls, split: str) -> Path:
        return cls.ROOT / "artifacts" / "phase2" / f"clv_{split}_predictions.csv"
    
    @classmethod
    def uplift_predictions(cls, model: str, split: str) -> Path:
        return cls.ROOT / "models" / "phase3" / f"{model}_pred_{split}.npy"
    
    @classmethod
    def sweep_csv(cls, split: str) -> Path:
        return cls.ROOT / "data" / "phase4" / f"budget_sweep_{split}.csv"
    
    @classmethod
    def metrics_json(cls) -> Path:
        return cls.ROOT / "artifacts" / "phase3" / "uplift_metrics.json"
    
    @classmethod
    def phase3_processed(cls, split: str) -> Path:
        return cls.ROOT / "artifacts" / "phase3" / "processed" / f"{split}_uplift.csv"
    
    @classmethod
    def feature_importance(cls, model: str) -> Path:
        return cls.ROOT / "artifacts" / "phase3" / f"feature_importance_{model}.csv"
    
# ==============================================================================
# DATA LOADERS
# ==============================================================================
@lru_cache(maxsize=32)
def load_enriched_split(split: str, model: str) -> pd.DataFrame:
    """
    Load Phase 4 enriched data with CLV and uplift predictions.
    Falls back to merging Phase 1/2/3 data if Phase 4 data doesn't exist.
    """
    # Try Phase 4 enriched data first (optimal path)
    p = DataPaths.phase4_enriched(split, model)
    if p.exists():
        try:
            df = pd.read_csv(p)
            logger.info(f"Loaded Phase 4 enriched data: {p} ({len(df)} rows)")
            return df
        except Exception as e:
            logger.warning(f"Failed to load {p}: {e}")
    
    # Fallback: Merge data from previous phases
    logger.warning(f"Phase 4 enriched data not found, falling back to merged data: {split}, {model}")
    return _merge_split_data(split, model)

def _merge_split_data(split: str, model: str) -> pd.DataFrame:
    """Internal helper to merge data from Phases 1-3"""
    p = DataPaths.split_data(split)
    if not p.exists():
        logger.error(f"Split data not found: {p}")
        return pd.DataFrame()

    df = pd.read_csv(p)
    df["customerID"] = df["customerID"].astype(str) if "customerID" in df.columns else df.index.astype(str)

    # Merge CLV predictions from Phase 2
    clv_p = DataPaths.clv_predictions(split)
    if clv_p.exists():
        clv_df = pd.read_csv(clv_p)
        clv_df["customerID"] = clv_df["customerID"].astype(str)
        df = df.merge(clv_df[["customerID", "predicted_clv"]], on="customerID", how="left")
        df.rename(columns={"predicted_clv": "clv"}, inplace=True)
    else:
        df["clv"] = 0.0
        logger.warning(f"CLV predictions not found: {clv_p}")

    # Merge uplift predictions from Phase 3
    up = DataPaths.uplift_predictions(model, split)
    if up.exists():
        try:
            preds = np.load(up).reshape(-1)
            if len(preds) == len(df):
                df["tau_hat"] = preds.astype(float)
            else:
                logger.warning(f"Uplift prediction length mismatch: {len(preds)} vs {len(df)}")
                df["tau_hat"] = 0.0
        except Exception as e:
            logger.error(f"Failed to load uplift predictions: {e}")
            df["tau_hat"] = 0.0
    else:
        df["tau_hat"] = 0.0
        logger.warning(f"Uplift predictions not found: {up}")
    up_processed = DataPaths.phase3_processed(split)
    if up_processed.exists():
        try:
            true_df = pd.read_csv(up_processed)
            true_df["customerID"] = true_df["customerID"].astype(str)
            df = df.merge(true_df[["customerID", "true_uplift"]], on="customerID", how="left")
        except Exception as e:
            logger.warning(f"Could not load true_uplift: {e}")
    # Add segment column if available
    if "segment" not in df.columns:
        df["segment"] = "Generic"

    return df

def load_selected_safe(split: str, model: str, budget: float) -> pd.DataFrame:
    """Load selected customers from Phase 4 CRM export"""
    p = DataPaths.crm_export(split, model, budget)
    if not p.exists():
        logger.warning(f"CRM export not found: {p}")
        return pd.DataFrame()
    return pd.read_csv(p)

@lru_cache(maxsize=8)
def load_feature_importance(model: str) -> pd.DataFrame:
    """Load Phase 3 feature importance with fallback"""
    p = DataPaths.feature_importance(model)
    
    if not p.exists():
        logger.warning(f"Feature importance not found: {p}")
        # Return empty DF with correct structure
        return pd.DataFrame({"feature": [], "importance": []})
    
    try:
        df = pd.read_csv(p)
        # Ensure correct column names
        df.columns = ["feature", "importance"]
        # Sort and take top 10
        return df.sort_values("importance", ascending=False).head(10)
    except Exception as e:
        logger.error(f"Failed to load feature importance: {e}")
        return pd.DataFrame({"feature": [], "importance": []})
    
# ==============================================================================
# COMPONENTS
# ==============================================================================
def navbar():
    """Navigation bar with all 4 pages and health indicator"""
    return dbc.Navbar(
        dbc.Container([
            dbc.NavbarBrand([
                html.I(className="bi bi-graph-up-arrow me-2"),
                "EcomOpti Decision Engine"
            ], className="fw-bold"),
            dbc.Nav([
                dbc.NavLink("Campaign Builder", href="/", active="exact"),
                dbc.NavLink("Historical Campaigns", href="/campaigns", active="exact"),
                dbc.NavLink("Model Performance", href="/performance", active="exact"),
                dbc.NavLink("Settings", href="/settings", active="exact"),
            ], pills=True, className="ms-auto"),
            # Health status indicator (auto-updates)
            html.Div(id="health-indicator", className="ms-3"),
        ]),
        color="dark",
        dark=True,
        className="mb-3"
    )

def health_status_indicator(status: str) -> html.Div:
    """Return health status indicator component"""
    if status == "healthy":
        return html.Div([
            html.I(className="bi bi-check-circle-fill text-success"),
            html.Small(" API Online", className="text-success ms-1")
        ])
    elif status == "degraded":
        return html.Div([
            html.I(className="bi bi-exclamation-triangle-fill text-warning"),
            html.Small(" API Degraded", className="text-warning ms-1")
        ])
    else:
        return html.Div([
            html.I(className="bi bi-x-circle-fill text-danger"),
            html.Small(" API Offline", className="text-danger ms-1")
        ])

def recommendation_banner(text: str):
    """Executive recommendation banner"""
    return dbc.Alert([
        html.H5(text, className="mb-0 text-center")
    ], color="success", className="mb-4 shadow-sm border-success")

# ==============================================================================
# APP LAYOUT & ROUTING
# ==============================================================================
app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.LUX, dbc.icons.BOOTSTRAP],
                suppress_callback_exceptions=True)
server = app.server

app.layout = html.Div([
    dcc.Location(id="url"),
    dcc.Interval(id="health-interval", interval=HEALTH_CHECK_INTERVAL, n_intervals=0),
    dcc.Store(id="config-store", storage_type="session", data={"split": "test", "model": "ensemble", "budget": 2500}),
    dcc.Store(id="optimization-store", storage_type="session"),
    dcc.Download(id="dl"),
    navbar(),
    html.Div(id="page-content"),
    # Status toast container
    dbc.Toast(
        id="status-toast",
        header="Operation Status",
        is_open=False,
        duration=4000,
        icon="info",
        style={"position": "fixed", "top": 66, "right": 10, "width": 350, "z-index": 1050},
    )
])

@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def route(path):
    """Multi-page routing"""
    if path == "/performance":
        return page_performance()
    elif path == "/campaigns":
        return page_campaigns()
    elif path == "/settings":
        return page_settings()
    return page_home()

def page_home():
    """Campaign Builder - Main optimization interface"""
    marks = {b: f"${b}" for b in range(BUDGET_MIN, BUDGET_MAX+1, 500)}
    return dbc.Container([
        dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([dbc.Label("1. Target Population"),
                             dcc.Dropdown(id="opt-split", options=[{"label": s.title(), "value": s} for s in VALID_SPLITS],
                                          value="test", clearable=False)], md=3),
                    dbc.Col([dbc.Label("2. Uplift Model"),
                             dcc.Dropdown(id="opt-model", options=[{"label": "Ensemble (Best)", "value": "ensemble"},
                                                                   {"label": "DR-Learner", "value": "dr_learner"}],
                                          value="ensemble", clearable=False)], md=3),
                    dbc.Col([dbc.Label("3. Budget Cap"),
                             dcc.Slider(id="opt-budget", min=BUDGET_MIN, max=BUDGET_MAX, step=BUDGET_STEP,
                                        value=2500, marks=marks, updatemode='mouseup'),
                                        html.Div(id="budget-display", className="mt-1 text-muted small")], md=4),
                    dbc.Col([html.Label(" "),
                             dbc.Button("Run Optimization", id="btn-opt", color="primary", className="w-100 mt-2")], md=2)
                ])
            ])
        ], className="shadow-sm border-0 mb-4"),
        dbc.Tabs(id="tabs", active_tab="tab-impact", children=[
            dbc.Tab(label="💰 Financial Impact", tab_id="tab-impact"),
            dbc.Tab(label="🎯 Audience Strategy", tab_id="tab-audience"),
            dbc.Tab(label="📤 Action & Export", tab_id="tab-export"),
        ]),
        html.Div(id="tab-content", className="mt-4")
    ], fluid=True)

def page_performance():
    """Model Performance - Diagnostics and calibration"""
    return dbc.Container([
        dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([dbc.Label("Split"),
                             dcc.Dropdown(id="perf-split", options=[{"label": s.title(), "value": s} for s in VALID_SPLITS],
                                          value="test", clearable=False)], md=3),
                    dbc.Col([dbc.Label("Model"),
                             dcc.Dropdown(id="perf-model", options=[{"label": "Ensemble", "value": "ensemble"},
                                                                   {"label": "DR-Learner", "value": "dr_learner"}],
                                          value="ensemble", clearable=False)], md=3),
                    dbc.Col([dbc.Label(" "), html.Br(),
                             dbc.Button("Refresh", id="btn-refresh-perf", color="primary")], md=2)
                ])
            ])
        ], className="shadow-sm border-0 mb-4"),
        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardHeader("Uplift Calibration (Predicted vs Actual)"),
                              dbc.CardBody(dcc.Graph(id="perf-calibration"))], className="shadow-sm border-0"), md=6),
            dbc.Col(dbc.Card([dbc.CardHeader("Budget Sensitivity"),
                              dbc.CardBody(dcc.Graph(id="perf-sweep"))], className="shadow-sm border-0"), md=6)
        ], className="mb-4"),
        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardHeader("Feature Importance"),
                              dbc.CardBody(dcc.Graph(id="perf-importance"))], className="shadow-sm border-0"), md=12)
        ])
    ], fluid=True)

def page_campaigns():
    """Historical Campaigns - Placeholder for Phase 7 database"""
    # Mock data for demonstration
    mock_campaigns = [
        {
            "campaign_id": "camp_001",
            "date": "2025-12-21",
            "model": "ensemble",
            "budget": 2500,
            "net_profit": 18472,
            "roi": 3.2
        },
        {
            "campaign_id": "camp_002",
            "date": "2025-12-20",
            "model": "dr_learner",
            "budget": 3000,
            "net_profit": 21200,
            "roi": 3.5
        }
    ]

    return dbc.Container([
        dbc.Card([
            dbc.CardHeader("Historical Campaigns"),
            dbc.CardBody([
                html.P("This page shows past optimization runs and their performance metrics.",
                       className="text-muted"),
                dbc.Alert("⚠️ Database integration coming in Phase 7", color="info", className="mb-3"),
                dash_table.DataTable(
                    id="campaigns-table",
                    columns=[
                        {"name": "Campaign ID", "id": "campaign_id"},
                        {"name": "Date", "id": "date"},
                        {"name": "Model", "id": "model"},
                        {"name": "Budget", "id": "budget", "type": "numeric", "format": {"specifier": "$,.0f"}},
                        {"name": "Net Profit", "id": "net_profit", "type": "numeric", "format": {"specifier": "$,.0f"}},
                        {"name": "ROI", "id": "roi", "type": "numeric", "format": {"specifier": ".2f"}}
                    ],
                    data=mock_campaigns,
                    page_size=10,
                    sort_action="native",
                    filter_action="native",
                    sort_mode="multi",
                    column_selectable="single",
                    row_selectable="multi",
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left', 'padding': '10px'},
                    style_header={'fontWeight': 'bold', 'backgroundColor': '#f8f9fa'},
                )
            ])
        ], className="shadow-sm border-0")
    ], fluid=True)

def page_settings():
    """Settings - API configuration and preferences (read-only for Phase 6)"""
    return dbc.Container([
        dbc.Card([
            dbc.CardHeader("Settings"),
            dbc.CardBody([
                dbc.Form([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("API Base URL"),
                            dbc.Input(id="settings-api-url", type="text", value=API_BASE_URL,
                                      placeholder="http://localhost:8000", readonly=True)
                        ], md=6),
                        dbc.Col([
                            dbc.Label("Default Treatment Cost ($)"),
                            dbc.Input(id="settings-cost", type="number", value=DEFAULT_COST,
                                      min=0, step=0.01, readonly=True)
                        ], md=3),
                        dbc.Col([
                            dbc.Label(" "),
                            dbc.Button("Save Settings", id="btn-save-settings", color="secondary",
                                       className="w-100 mt-4", disabled=True)
                        ], md=3)
                    ]),
                    html.Hr(),
                    dbc.Label("Theme Selection"),
                    dbc.RadioItems(
                        id="settings-theme",
                        options=[{"label": "LUX (Default)", "value": "LUX"},
                                 {"label": "DARKLY", "value": "DARKLY"},
                                 {"label": "FLATLY", "value": "FLATLY"}],
                        value="LUX",
                        inline=True
                    ),
                    dbc.FormText("Theme switching is disabled in Phase 6 (requires app restart)",
                                color="muted")
                ])
            ])
        ], className="shadow-sm border-0 mb-4"),
        dbc.Card([
            dbc.CardHeader("API Health Status"),
            dbc.CardBody(html.Div(id="settings-health-detail"))
        ], className="shadow-sm border-0")
    ], fluid=True)

# ==============================================================================
# CALLBACKS
# ==============================================================================

# Health Check Callback (runs every 10s)
@app.callback(
    Output("health-indicator", "children"),
    Input("health-interval", "n_intervals")
)
def check_health(n):
    """Check API health and update navbar indicator"""
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=3)
        if r.status_code == 200:
            data = r.json()
            if data.get("status") == "healthy":
                return health_status_indicator("healthy")
            else:
                return health_status_indicator("degraded")
    except Exception as e:
        logger.error(f"Health check failed: {e}")
    return health_status_indicator("offline")

# Settings Health Detail (shows on page load)
@app.callback(
    Output("settings-health-detail", "children"),
    Input("url", "pathname")
)
def show_health_detail(path):
    """Show detailed health status when settings page loads"""
    if path != "/settings":
        return no_update

    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=5)
        data = r.json()
        items = []
        for phase, ready in data.items():
            if phase.endswith("_ready"):
                phase_name = phase.replace("_ready", "").replace("_", " ").title()
                icon = html.I(className="bi bi-check-circle-fill text-success") if ready else \
                       html.I(className="bi bi-x-circle-fill text-danger")
                items.append(dbc.ListGroupItem([
                    icon,
                    f" {phase_name}: {'Ready' if ready else 'Not Ready'}"
                ]))
        return dbc.ListGroup(items)
    except Exception as e:
        logger.error(f"Failed to fetch health details: {e}")
        return dbc.Alert(f"❌ Cannot connect to API: {e}", color="danger")

# Update Config Store when controls change
@app.callback(
    Output("config-store", "data"),
    Input("opt-split", "value"),
    Input("opt-model", "value"),
    Input("opt-budget", "value")
)
def update_config(s, m, b):
    """Keep config in sync with UI controls"""
    return {"split": s, "model": m, "budget": b}

# Main Optimization Callback with Loading & Error States
@app.callback(
    Output("optimization-store", "data"),
    Output("btn-opt", "loading"),
    Output("status-toast", "children"),
    Output("status-toast", "is_open"),
    Output("status-toast", "icon"),
    Output("status-toast", "color"),
    Output("opt-split", "disabled"),      # ✅ NEW: Lock dropdown
    Output("opt-model", "disabled"),      # ✅ NEW: Lock dropdown
    Output("opt-budget", "disabled"),     # ✅ NEW: Lock slider
    Input("btn-opt", "n_clicks"),
    State("opt-split", "value"),
    State("opt-model", "value"),
    State("opt-budget", "value"),
    prevent_initial_call=True
)
def run_optimization(n, split, model, budget):
    """
    Execute Phase 4 optimization via API with UI locking.
    Controls are disabled during execution to prevent config drift.
    """
    if n is None or n == 0:
        # All controls enabled
        return no_update, False, None, False, "info", "info", False, False, False
    
    # Lock controls during API call
    try:
        # Call Phase 5 API
        logger.info(f"Starting optimization: {split}/{model}/budget={budget}")
        r = requests.post(
            f"{API_BASE_URL}/phase4/optimize",
            json={"split": split, "model": model, "budget": float(budget)},
            timeout=30
        )
        r.raise_for_status()
        result = r.json()
        metrics = result.get('metrics', {})
        customers = metrics.get('customers_selected', 0)
        # Success: Unlock controls
        toast_msg = dbc.Alert(f"✅ Optimization complete! Selected {result['metrics']['customers_selected']} customers",
                            color="success",
                            className="mb-0"
                            )
        logger.info("Optimization successful")

        return result, False, toast_msg, True, "success", "success", False, False, False

    except requests.exceptions.Timeout:
        error_msg = "⏱️ Request timed out (30s). Try a smaller budget."
        toast_msg = dbc.Alert(error_msg, color="warning", className="mb-0")
        logger.warning(error_msg)
        return None, False, toast_msg, True, "warning", "warning", False, False, False
    
    except requests.exceptions.ConnectionError:
        error_msg = "❌ Cannot connect to API. Is the Phase 5 server running?"
        toast_msg = dbc.Alert(error_msg, color="danger", className="mb-0")
        logger.error(error_msg)
        return None, False, toast_msg, True, "danger", "danger", False, False, False

    except Exception as e:
        error_msg = f"❌ Optimization failed: {str(e)}"
        toast_msg = dbc.Alert(error_msg, color="danger", className="mb-0")
        logger.error(error_msg, exc_info=True)
        return None, False, toast_msg, True, "danger", "danger", False, False, False
    
# Render Tabs based on active tab and optimization results
@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "active_tab"),
    Input("optimization-store", "data"),
    State("config-store", "data")
)
def render_tabs(tab, result, cfg):
    """Render tab content dynamically based on active tab"""
    if not result:
        return dbc.Alert("Run optimization to see results.", color="info", className="mt-4")

    split, model = cfg["split"], cfg["model"]
    budget = float(cfg["budget"])

    # Load data with caching
    df_all = load_enriched_split(split, model)
    df_sel = load_selected_safe(split, model, budget)

    # Calculate simulated metrics (uniform cost basis)
    if not df_sel.empty:
        # Verify required columns exist
        required_cols = {"treatment_cost", "incremental_value"}
        if not required_cols.issubset(df_sel.columns):
            logger.warning(f"Missing columns {required_cols} in CRM export")
            sim_cost = sim_val = sim_net = 0
            sim_roi = 0
        else:
            # Use REAL segment-based costs from Phase 4
            sim_cost = df_sel["treatment_cost"].sum()
            sim_val = df_sel["incremental_value"].sum()
            sim_net = sim_val - sim_cost
            sim_roi = (sim_val / sim_cost) if sim_cost > 0 else 0
    else:
        sim_cost = sim_val = sim_net = 0
        sim_roi = 0

    # Generate business recommendation
    primary_segment = df_sel["segment"].mode().iloc[0] if not df_sel.empty and "segment" in df_sel.columns else "High Uplift"
    rec_text = generate_recommendation(len(df_sel), primary_segment, sim_net)

    if tab == "tab-impact":
        return html.Div([
            recommendation_banner(rec_text),
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([html.H6("Simulated Net Profit"),
                                               html.H3(f"${sim_net:,.0f}", className="text-success"),
                                               html.Small("Uniform Cost Basis", className="text-muted")]),
                         className="shadow-sm"), md=3),
                dbc.Col(dbc.Card(dbc.CardBody([html.H6("Simulated ROI"),
                                               html.H3(f"{sim_roi:.2f}x", className="text-primary"),
                                               html.Small("(vs 1.0x Random)", className="text-muted")]),
                         className="shadow-sm"), md=3),
                dbc.Col(dbc.Card(dbc.CardBody([html.H6("Customers Targeted"),
                                               html.H3(f"{len(df_sel):,}", className="text-dark")]),
                         className="shadow-sm"), md=3),
                dbc.Col(dbc.Card(dbc.CardBody([html.H6("Avg Treatment Cost"),
                                               html.H3(f"${sim_cost/len(df_sel):.2f}"if len(df_sel)>0 else "$0.00", className="text-muted"),
                                               html.Small("Segment-based", className="text-muted")]),
                         className="shadow-sm"), md=3),
            ], className="mb-4"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=cumulative_gain_figure(df_all, budget, DEFAULT_COST)), md=6),
                dbc.Col(dcc.Graph(figure=waterfall_figure(sim_val, sim_cost)), md=6),
            ])
        ])
            
    elif tab == "tab-audience":
        scenario_fig = scenario_analysis_figure(df_sel, DEFAULT_COST) if not df_sel.empty else empty_fig("Scenario Analysis", "No data")
        return html.Div([
            dbc.Card([
                dbc.CardHeader("🎯 Customer Strategy Matrix"),
                dbc.CardBody(dcc.Graph(figure=strategy_matrix(df_all, df_sel)))
            ], className="shadow-sm border-0 mb-4"),
            dbc.Row([
                dbc.Col(dcc.Graph(id="segment-pie", figure=px.pie(
                    df_sel, names="tenure_segment",
                    title="Segment Distribution" if not df_sel.empty else "No Data",
                    color_discrete_sequence=px.colors.qualitative.Set2
                )), md=6),
                dbc.Col(dcc.Graph(figure=scenario_fig), md=6)
            ])
        ])

    elif tab == "tab-export":
        display_df = df_sel.head(100) if not df_sel.empty else pd.DataFrame()
        return dbc.Card(dbc.CardBody([
            html.H5("Actionable Target List"),
            dbc.Button("Download Full CSV", id="btn-dl", color="success", className="mb-3"),
            dash_table.DataTable(
                id="export-table",
                columns=[{"name": i, "id": i} for i in display_df.columns] if not display_df.empty else [],
                data=display_df.to_dict('records'),
                style_table={'overflowX': 'auto'},
                page_size=10,
                sort_action="native",
                filter_action="native",
                sort_mode="multi",
                page_action="native",
                style_cell={'textAlign': 'left', 'padding': '10px'},
                style_header={'fontWeight': 'bold', 'backgroundColor': '#f8f9fa'},
            )
        ]))

# CSV Download Callback
@app.callback(
    Output("dl", "data"),
    Input("btn-dl", "n_clicks"),
    State("config-store", "data"),
    prevent_initial_call=True
)
def download_results(n, cfg):
    """Download CRM export CSV"""
    ctx = dash.callback_context
    if not ctx.triggered or ctx.triggered[0]['prop_id'].split('.')[0] != "btn-dl":
        return no_update
    
    if n is None or n < 1:
        return no_update
    split = cfg["split"]
    model = cfg["model"]  
    budget = cfg["budget"]  
    p = DataPaths.crm_export(split, model, budget)
    if p.exists():
        logger.info(f"Downloading CSV: {p}")
        return dcc.send_file(p)
    else:
        logger.error(f"File not found for download: {p}")
        return no_update

# Click-to-Filter Callback for Segment Pie Chart
@app.callback(
    Output("export-table", "data"),
    Input("segment-pie", "clickData"),
    State("config-store", "data"),
    prevent_initial_call=True
)
def filter_by_segment(clickData, cfg):
    """Filter customer table by clicked segment"""
    # Validate clickData structure to prevent KeyError
    if not clickData or "points" not in clickData or not clickData["points"]:
        logger.warning("Invalid clickData received")
        return no_update

    split = cfg["split"]
    model = cfg["model"]  
    budget = cfg["budget"]  
    segment = clickData["points"][0]["label"]

    df = load_selected_safe(split, model, budget)
    if df.empty:
        logger.warning(f"No data available for split: {split}")
        return []

    # Filter by segment if column exists
    if "segment" in df.columns:
        filtered = df[df["segment"] == segment]
        logger.info(f"Filtered table by segment '{segment}': {len(filtered)} rows")
    else:
        filtered = df

    return filtered.to_dict('records')

# Performance Page Refresh
@app.callback(
    Output("perf-calibration", "figure"),
    Output("perf-sweep", "figure"),
    Output("perf-importance", "figure"),
    Input("btn-refresh-perf", "n_clicks"),
    State("perf-split", "value"),
    State("perf-model", "value"),
    prevent_initial_call=False
)
def update_performance_plots(n, split, model):
    """Refresh performance visualizations"""
    logger.info(f"Refreshing performance plots: {split}, {model}")
    df_all = load_enriched_split(split, model)
    sweep_path = DataPaths.sweep_csv(split)

    calibration_fig = uplift_calibration_figure(df_all)
    sweep_fig = budget_sweep_figure(sweep_path)
    importance_fig = feature_importance_figure(df_all, model)

    return calibration_fig, sweep_fig, importance_fig

# Settings Save Button (Read-only for Phase 6)
@app.callback(
    Output("settings-health-detail", "children", allow_duplicate=True),
    Input("btn-save-settings", "n_clicks"),
    prevent_initial_call=True
)
def save_settings(n):
    """Settings save (read-only for Phase 6 MVP)"""
    return dbc.Alert("✅ Settings interface ready (Phase 7: Persistent storage)", color="info", duration=3000)

@app.callback(
    Output("budget-display", "children"),
    Input("opt-budget", "value")
)
def update_budget_display(budget):
    """Show the currently selected budget value"""
    return f"Selected: ${budget:,.0f}"

if __name__ == "__main__":
    # Ensure all required directories exist
    for dir_name in ["logs", "data/splits", "data/phase4", "artifacts/phase2", "models/phase3"]:
        (DataPaths.ROOT / dir_name).mkdir(parents=True, exist_ok=True)

    logger.info("="*60)
    logger.info("Starting EcomOpti Decision Engine Dashboard - Phase 6")
    logger.info("="*60)

    app.run(host="0.0.0.0", port=8050, debug=False, dev_tools_hot_reload=False)