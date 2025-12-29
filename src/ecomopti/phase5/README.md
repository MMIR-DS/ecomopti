```markdown
# Phase 5: ML Pipeline API

FastAPI microservice exposing Phases 1-4 ML pipeline via REST API.

## Quick Start

### Prerequisites
- Python 3.9+
- All Phase 1-4 requirements installed
- Phases 1-3 completed (artifacts exist)

### Local Development
```bash
# From project root
python run_api.py
# Open http://localhost:8000/docs for interactive Swagger UI
```

### Production Deployment
```bash
# Use gunicorn with multiple workers for production
gunicorn src.ecomopti.phase5.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000

# Or uvicorn directly
uvicorn src.ecomopti.phase5.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Endpoints

### 1. Run Full Pipeline (Async)
**POST** `/pipeline/run`

Execute Phases 1-4 end-to-end asynchronously. Returns job ID for status monitoring.

**Request:**
```json
{
  "split": "train",
  "model": "ensemble",
  "budget": 5000.0,
  "run_sweep": false,
  "run_scenarios": false
}
```

**Response (202 Accepted):**
```json
{
  "status": "running",
  "job_id": "train_ensemble_5000.0",
  "message": "Pipeline started. Monitor via /pipeline/status endpoint.",
  "check_status": "/pipeline/status/train_ensemble_5000.0",
  "log_file": "logs/pipeline_train_ensemble_5000.0.log"
}
```

**Typical Runtime:** 10-15 minutes for full pipeline

---

### 2. Check Pipeline Status
**GET** `/pipeline/status/{job_id}`

Monitor async pipeline execution.

**Response:**
```json
{
  "status": "completed",  // or "running" or "failed"
  "log": "Last 1000 characters of log..."
}
```

---

### 3. Run Optimization Only (Sync)
**POST** `/phase4/optimize`

Run Phase 4 budget optimization synchronously (5-30 seconds). Does NOT re-run Phases 1-3.

**Request:**
```json
{
  "split": "test",
  "model": "dr_learner",
  "budget": 2500.0
}
```

**Response (200 OK):**
```json
{
  "status": "complete",
  "split": "test",
  "model": "dr_learner",
  "budget": 2500.0,
  "metrics": {
    "customers_selected": 485,
    "net_profit": 3441.89,
    "roi": 1.38,
    "total_cost": 2500.0,
    "incremental_clv_impact": 5941.89,
    "net_value": 3441.89
  },
  "files": {
    "crm_export": "data/phase4/crm_export_test.csv",
    "report": "reports/phase4/report_test.json",
    "summary": "reports/phase4/summary_test.md"
  }
}
```

---

### 4. Health Check
**GET** `/health`

Verify all pipeline dependencies exist.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-12-29T21:30:48",
  "phase1_ready": true,
  "phase2_ready": true,
  "phase3_ready": true,
  "phase4_ready": true
}
```

---

### 5. API Root
**GET** `/`

Returns API metadata and endpoint catalog.

## Error Handling

All endpoints return standardized error responses:

```json
{
  "detail": "Error message here"
}
```

**Common HTTP Status Codes:**
- `202` – Async pipeline started
- `200` – Synchronous operation successful
- `500` – Internal server error (pipeline failure)
- `404` – Job ID not found
- `422` – Validation error (invalid request format)

## Configuration

**Environment Variables:**
```bash
# Optional: Override default host/port
export API_HOST=0.0.0.0
export API_PORT=8000

# Optional: Increase log level for debugging
export LOG_LEVEL=DEBUG
```

**Request Limits:**
- Max async jobs: Unlimited (monitor disk space for logs)
- Sync optimization timeout: 30 seconds (default)
- Request body size: 1MB (FastAPI default)

## Development Notes

### Adding New Endpoints
1. Create Pydantic model for request/response
2. Add endpoint function with proper decorators
3. Update `root()` endpoint catalog

### Testing
```bash
# Run tests
pytest tests/test_phase5_api.py

# Manual testing with httpie
http POST localhost:8000/phase4/optimize split=test model=ensemble budget=1000

# Load testing
locust -f tests/load_test.py --host http://localhost:8000
```

### Logging
- Logs: `logs/pipeline_{job_id}.log`
- API access logs: console (uvicorn)
- Error logs: `logs/api_errors.log` (if configured)

## Integration with Phase 6 Dashboard

The Phase 6 Dash dashboard calls this API via:
```python
# In app.py
requests.post("http://localhost:8000/phase4/optimize", json={...})
```

Ensure the API is running before launching the dashboard:
```bash
# Terminal 1
python run_api.py

# Terminal 2
python run_dashboard.py
```

---

## Troubleshooting

**Issue:** `ModuleNotFoundError: No module named 'ecomopti'`
- **Fix:** Run from project root, ensure `PYTHONPATH` includes `src/`

**Issue:** Phase 4 optimization fails with "Artifacts not found"
- **Fix:** Run Phase 3 first to generate uplift predictions

**Issue:** Health check shows `phase4_ready: false`
- **Fix:** Verify `./src/ecomopti/phase4/build.py` exists

**Issue:** Async pipeline stuck in "running" state
- **Fix:** Check `logs/pipeline_{job_id}.log` for errors, ensure Phase 2/3 scripts are executable
```