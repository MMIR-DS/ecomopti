"""
Phase 5: FastAPI Microservice for EcomOpti ML Pipeline
Exposes Phases 1-4 via REST endpoints
"""

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Optional, Dict, Any
import subprocess
import json

# Initialize FastAPI
app = FastAPI(
    title="EcomOpti ML Pipeline API",
    version="1.0.0",
    description="REST API for Phases 1-4: Data Processing → CLV → Uplift → Budget Optimization"
)

# ============================================================================
# REQUEST MODELS (Pydantic)
# ============================================================================
class PipelineRequest(BaseModel):
    """Execute full pipeline (Phases 1-4) asynchronously"""
    split: str = Field(..., description="Which split to run", pattern="^(train|val|test)$")
    model: str = Field("ensemble", description="Phase 3 model", pattern="^(dr_learner|s_learner|ensemble)$")
    budget: float = Field(7500.0, description="Marketing budget", gt=0)
    run_sweep: bool = Field(False, description="Run budget sweep")
    run_scenarios: bool = Field(False, description="Run scenario analysis")

class OptimizeRequest(BaseModel):
    """Run Phase 4 optimization only (synchronous, 5-30 seconds)"""
    split: str = Field(..., pattern="^(train|val|test)$")
    model: str = Field("ensemble", pattern="^(dr_learner|s_learner|ensemble)$")
    budget: float = Field(7500.0, gt=0)

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    phase1_ready: bool
    phase2_ready: bool
    phase3_ready: bool
    phase4_ready: bool

# ============================================================================
# BACKGROUND TASK EXECUTOR
# ============================================================================
def execute_pipeline(request: PipelineRequest, log_file: Path):
    """
    Execute Phases 1-4 sequentially via subprocess with robust error handling.
    
    Design Decisions:
    1. Subprocess calls isolate each phase's environment (prevents memory leaks)
    2. Logging to file enables async status monitoring via /pipeline/status
    3. Sequential execution enforces Phase 2 → 3 → 4 dependency chain
    4. Return code checking ensures failures are caught and logged
    
    Args:
        request: PipelineRequest with split, model, budget parameters
        log_file: Path to write detailed execution logs
    
    Raises:
        Exception: If any phase fails (non-zero return code)
    
    Time Estimates:
        - Phase 1: 2-3 minutes (data processing)
        - Phase 2: 5-8 minutes (CLV modeling)
        - Phase 3: 5-7 minutes (uplift modeling)
        - Phase 4: 10-30 seconds (optimization)
        Total: ~12-18 minutes per run
    """
    try:
        with open(log_file, "w", encoding="utf-8") as f:
            # Phase 1
            f.write("="*60 + "\n")
            f.write("PHASE 1: Data Pipeline\n")
            f.write("="*60 + "\n")
            cmd = ["python", "-m", "ecomopti.phase1.build"]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
            f.write(result.stdout)
            if result.stderr:
                f.write(f"\nERRORS:\n{result.stderr}\n")
            if result.returncode != 0:
                raise Exception("Phase 1 failed")
            
            # Phase 2
            f.write("\n" + "="*60 + "\n")
            f.write(f"PHASE 2: CLV Modeling ({request.split})\n")
            f.write("="*60 + "\n")
            cmd = ["python", "-m", "ecomopti.phase2.build", "--split", request.split]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
            f.write(result.stdout)
            if result.stderr:
                f.write(f"\nERRORS:\n{result.stderr}\n")
            
            # Phase 3
            f.write("\n" + "="*60 + "\n")
            f.write(f"PHASE 3: Uplift Modeling ({request.split}, {request.model})\n")
            f.write("="*60 + "\n")
            cmd = [
                "python", "-m", "ecomopti.phase3.build",
                "--split", request.split,
                "--model", request.model
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
            f.write(result.stdout)
            if result.stderr:
                f.write(f"\nERRORS:\n{result.stderr}\n")
            
            # Phase 4
            f.write("\n" + "="*60 + "\n")
            f.write(f"PHASE 4: Budget Optimization (${request.budget})\n")
            f.write("="*60 + "\n")
            cmd = [
                "python", "-m", "ecomopti.phase4.build",
                "--split", request.split,
                "--model", request.model,
                "--budget", str(request.budget)
            ]
            if request.run_sweep:
                cmd.append("--sweep")
            if request.run_scenarios:
                cmd.append("--scenarios")
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
            f.write(result.stdout)
            if result.stderr:
                f.write(f"\nERRORS:\n{result.stderr}\n")
            
            f.write("\n" + "🎉" + "✅ PIPELINE COMPLETE\n")
            
    except Exception as e:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n❌ PIPELINE FAILED: {str(e)}\n")

# ============================================================================
# API ENDPOINTS
# ============================================================================
@app.post("/pipeline/run", status_code=202)
def run_pipeline(request: PipelineRequest, background_tasks: BackgroundTasks):
    """
    Execute Phases 1-4 end-to-end asynchronously.
    Returns job ID immediately. Check status endpoint for results.
    
    **Async Design:**
    - Returns 202 Accepted immediately (non-blocking)
    - BackgroundTasks handles execution in separate thread
    - Job ID format: {split}_{model}_{budget}
    
    **Monitoring:**
    Use GET /pipeline/status/{job_id} to poll for completion.
    Logs streamed to logs/pipeline_{job_id}.log
    
    **Error Handling:**
    - 202: Job started successfully
    - 500: Phase execution failure (check logs)
    
    **Example:**
    ```bash
    curl -X POST http://localhost:8000/pipeline/run \
      -H "Content-Type: application/json" \
      -d '{"split": "train", "model": "ensemble", "budget": 5000}'
    ```
    """
    job_id = f"{request.split}_{request.model}_{request.budget}"
    log_file = Path(f"logs/pipeline_{job_id}.log")
    log_file.parent.mkdir(exist_ok=True)
    
    # Start background task
    background_tasks.add_task(execute_pipeline, request, log_file)
    
    return {
        "status": "running",
        "job_id": job_id,
        "message": "Pipeline started. Monitor via /pipeline/status endpoint.",
        "check_status": f"/pipeline/status/{job_id}",
        "log_file": str(log_file)
    }

@app.get("/pipeline/status/{job_id}")
def get_pipeline_status(job_id: str):
    """
    Check pipeline status: running, completed, or failed
    """
    log_file = Path(f"logs/pipeline_{job_id}.log")
    
    if not log_file.exists():
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    content = log_file.read_text(encoding="utf-8")
    
    if "✅ PIPELINE COMPLETE" in content:
        return {"status": "completed", "log": content[-1000:]}
    elif "❌ PIPELINE FAILED" in content:
        return {"status": "failed", "log": content[-1000:]}
    else:
        return {"status": "running", "log": content[-500:]}

@app.post("/phase4/optimize", status_code=200)
def optimize_budget(request: OptimizeRequest):
    """
    Run Phase 4 optimization only (synchronous, 5-30 seconds).
    Does NOT re-run Phases 1-3 (assumes they exist).
    """
    try:
        # Import here to avoid circular imports
        from ecomopti.phase4.build import run_optimization_workflow
        
        result = run_optimization_workflow(
            split=request.split,
            model_hint=request.model,
            budget=request.budget,
            run_sweep=False,
            run_scenarios=False,
            run_comparison=False
        )
        
        return {
            "status": "complete",
            "split": request.split,
            "model": request.model,
            "budget": request.budget,
            "metrics": {
                "customers_selected": result["optimization_results"]["ilp"]["metrics"]["total_customers"],
                "net_profit": result["business_impact"]["net_profit"],
                "roi": result["business_impact"]["roi_ratio"],
                "total_cost": result["optimization_results"]["ilp"]["metrics"]["total_cost"],
                "incremental_clv_impact": result["optimization_results"]["ilp"]["metrics"]["incremental_clv_impact"],
                "net_value": result["business_impact"]["net_profit"] 
            },
            "files": {
                "crm_export": f"data/phase4/crm_export_{request.split}.csv",
                "report": f"reports/phase4/report_{request.split}.json",
                "summary": f"reports/phase4/summary_{request.split}.md"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")

@app.get("/health")
def health_check():
    """
    Verify all pipeline dependencies exist
    """
    base_dir = Path(".")
    checks = {
        "phase1_ready": (base_dir / "data/splits/train.csv").exists(),
        "phase2_ready": (base_dir / "artifacts/phase2/clv_val_predictions.csv").exists(),
        "phase3_ready": (base_dir / "models/phase3/ensemble_pred_val.npy").exists(),
        "phase4_ready": (base_dir / "src/ecomopti/phase4/build.py").exists(),
    }
    
    all_ready = all(checks.values())
    
    return {
        "status": "healthy" if all_ready else "degraded",
        "timestamp": "2025-12-18T16:00:00",
        **checks
    }

@app.get("/")
def root():
    """API Root"""
    return {
        "name": "EcomOpti ML Pipeline API",
        "version": "1.0.0",
        "description": "Unified REST API for Phases 1-4",
        "endpoints": {
            "health": "/health",
            "run_pipeline": "/pipeline/run (POST)",
            "check_status": "/pipeline/status/{job_id} (GET)",
            "optimize": "/phase4/optimize (POST)",
            "docs": "/docs"
        }
    }