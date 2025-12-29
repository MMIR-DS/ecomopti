```markdown
# Phase 6: Campaign Dashboard & Decision Interface

## Installation

```bash
# Install Phase 6 runtime dependencies (from project root)
pip install -r src/ecomopti/phase6/requirements.txt

# For development (testing, code quality)
pip install -r src/ecomopti/phase6/requirements-dev.txt
```

## Prerequisites

**API Dependency: Phase 5 MUST be running**
```bash
# Terminal 1: Start Phase 5 API
python run_api.py
# Verify: curl http://localhost:8000/health → should return "healthy"
```

**Artifacts Required:**
- Phase 4 optimization results in `data/phase4/`
- CLV predictions in `artifacts/phase2/`
- Uplift predictions in `models/phase3/`

## Quick Start

```bash
# Terminal 2: Launch dashboard
python run_dashboard.py

# Open browser
# Main interface: http://localhost:8050
# Model Performance: http://localhost:8050/performance
# Settings: http://localhost:8050/settings
```

**Default credentials:** None (local development)

---

## Configuration

Edit `src/ecomopti/phase6/app.py` constants:

```python
API_BASE_URL = "http://localhost:8000"  # Change for production
DEFAULT_COST = 5.0                      # Uniform cost for simulations
BUDGET_MIN = 100                        # UI slider minimum
BUDGET_MAX = 5000                       # UI slider maximum
BUDGET_STEP = 100                       # Slider increment
```

**Environment Variables:**
```bash
export ECOMOPTI_API_URL="http://api.production.com"  # Override default API URL
```

---

## Architecture

### Business Logic: Two Cost Models

| Component | Cost Model | Purpose |
|-----------|------------|---------|
| **Phase 4 (Optimization Engine)** | Variable ($5-15 by segment) | True ROI calculation & CRM export |
| **Phase 6 (Dashboard Simulator)** | Uniform (user-controlled) | What-if scenario analysis |

**Why?** Phase 4 finds the optimal budget allocation. Phase 6 lets executives play with assumptions.

### UI Flow
1. **Campaign Builder** → Configure parameters → Run optimization
2. **Financial Impact** → ROI waterfall, cumulative gain curves
3. **Audience Strategy** → Segment distribution, sleeping dogs detection
4. **Action & Export** → Download CRM-ready CSV

### Data Flow
```python
User Input → config-store → API Call → optimization-store → Tab Rendering
     ↓              ↓              ↓              ↓              ↓
  Controls    Session State   Phase 5      Cached Result   Plotly Graphs
```

---

## API Endpoints Used

| Dashboard Feature | API Endpoint | Method |
|-------------------|--------------|--------|
| Health indicator | `/health` | GET |
| Run optimization | `/phase4/optimize` | POST |
| Model performance | `/phase4/optimize` (internal) | POST |

---

## Troubleshooting

### **Dashboard won't start**
```bash
# Check if port 8050 is in use
lsof -i :8050
kill -9 <PID>

# Verify Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### **"API Offline" in navbar**
- **Cause:** Phase 5 not running or wrong `API_BASE_URL`
- **Fix:** Start Phase 5 API and verify `curl http://localhost:8000/health`

### **Optimization fails with "No data"**
- **Cause:** Missing Phase 4 artifacts
- **Fix:** Run Phase 4 first: `python -m ecomopti.phase4.build --split test --model ensemble`

### **CRM export empty**
- **Cause:** No profitable customers at given budget
- **Fix:** Increase budget or check Phase 4 logs in `logs/pipeline_*.log`

### **Plots don't load**
- **Cause:** Large dataset (>10k rows) causing memory issues
- **Fix:** Dashboard auto-samples to 5k points. For full data, use Phase 4 CSV exports directly.

---

## Production Deployment

### **Gunicorn (Recommended)**
```bash
gunicorn src.ecomopti.phase6.app:server -w 4 -b 0.0.0.0:8050
```

### **Docker**
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8050
CMD ["gunicorn", "src.ecomopti.phase6.app:server", "-w", "4", "-b", "0.0.0.0:8050"]
```

### **Environment Variables**
```bash
export ECOMOPTI_API_URL="http://api.internal.com"
export DASH_DEBUG_MODE="false"
```

---

## Security Notes

- **No authentication**: Phase 6 MVP assumes internal network use
- **No CORS protection**: Add `flask_cors` if accessing API cross-domain
- **File system access**: Dashboard reads local files; restrict permissions in production

---

## Performance

- **Health polling**: Every 30 seconds (configurable)
- **Data caching**: LRU cache stores up to 32 datasets
- **Auto-sampling**: Large datasets (>5k rows) automatically sampled for scatter plots
- **Tab lazy-loading**: Only active tab renders, reducing initial load time

---

## Roadmap (Phase 7)

- **Database persistence** for historical campaigns
- **User authentication** with role-based access
- **Advanced what-if scenarios** (cost curves, seasonality)
- **A/B test integration** for campaign measurement

---

## Support

**Logs:** Check `logs/dashboard.log` for runtime errors
**API Logs:** Check `logs/pipeline_*.log` for Phase 5 issues

**Total runtime:** ~30 seconds for full optimization on 5k customers