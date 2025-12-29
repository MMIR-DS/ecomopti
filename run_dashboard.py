# run_dashboard.py - Robust Launcher
# File 3 of 5
#!/usr/bin/env python3

import os
import sys
from pathlib import Path
import subprocess
import time
import requests
import logging

# Setup logging for launcher
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

def check_api_health():
    """Verify Phase 5 API is running before starting dashboard"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "healthy":
                logger.info("✅ Phase 5 API is healthy and ready")
                return True
            else:
                logger.warning("⚠️ Phase 5 API is running but degraded")
                return True
    except requests.exceptions.ConnectionError:
        logger.error("❌ Phase 5 API is not running on http://localhost:8000")
        print("\nPlease start the API first with:")
        print("  python run_api.py")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error checking API health: {e}")
        return False

def create_directories():
    """Ensure all required directories exist"""
    dirs = [
        "logs",
        "data/splits",
        "data/phase4",
        "artifacts/phase2",
        "models/phase3",
        "plots/phase6",
        "reports/phase4"  # Added for completeness
    ]

    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Ensured directory exists: {dir_path}")

def print_startup_info():
    """Display startup banner with navigation URLs"""
    banner = """
╔════════════════════════════════════════════════════════════════╗
║        EcomOpti Decision Engine - Phase 6 Dashboard            ║
║                   Debugged & Production Ready                  ║
╚════════════════════════════════════════════════════════════════╝

📊 Dashboard Pages:
   • Campaign Builder:     http://localhost:8050
   • Historical Campaigns: http://localhost:8050/campaigns
   • Model Performance:    http://localhost:8050/performance
   • Settings:             http://localhost:8050/settings

🔌 API Integration:
   • Phase 5 API:          http://localhost:8000
   • Health Check:         http://localhost:8000/health
   • Swagger UI:           http://localhost:8000/docs

⚙️  Features:
   • Multi-page Power BI-style navigation
   • Real-time optimization with loading states
   • Click-to-filter on segment visualizations
   • CSV export with pagination & sorting
   • API health monitoring
   • Scenario analysis controls
   • LRU caching for 10x performance

📝 Logs:
   • Dashboard logs:       logs/dashboard.log
   • API logs:             logs/pipeline_*.log

Press Ctrl+C to stop the dashboard.
    """
    print(banner)

def main():
    """Main entry point with pre-flight checks"""
    logger.info("="*60)
    logger.info("Starting EcomOpti Phase 6 Dashboard Launcher")
    logger.info("="*60)

    print("\n🚀 Starting EcomOpti Phase 6 Dashboard...\n")

    # Step 1: Create directories
    logger.info("Step 1: Creating directory structure...")
    create_directories()

    # Step 2: Check API health
    logger.info("Step 2: Checking Phase 5 API health...")
    print("\n🔍 Checking Phase 5 API health...")
    if not check_api_health():
        response = input("\n⚠️  Warning: API not healthy. Continue anyway? [y/N]: ")
        if response.lower() != 'y':
            logger.info("User cancelled startup due to API issues")
            print("Exiting.")
            sys.exit(1)
        else:
            logger.warning("Proceeding despite API issues")

    # Step 3: Print startup info
    print_startup_info()

    # Step 4: Launch Dash app
    try:
        # Set environment variable for API URL
        os.environ.setdefault("ECOMOPTI_API_URL", "http://localhost:8000")

        # Launch the dashboard
        cmd = [sys.executable, "src/ecomopti/phase6/app.py"]

        logger.info("Step 3: Starting Dash server...")
        print("\n🔄 Starting Dash server...")
        process = subprocess.Popen(cmd)

        # Wait for graceful shutdown
        try:
            process.wait()
        except KeyboardInterrupt:
            logger.info("Received Ctrl+C, shutting down...")
            print("\n\n⏹️  Shutting down dashboard...")
            process.terminate()
            process.wait(timeout=5)
            logger.info("Dashboard stopped gracefully")
            print("✅ Dashboard stopped gracefully")

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt during startup")
        print("\n\n⏹️  Shutting down dashboard...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Failed to start dashboard: {e}", exc_info=True)
        print(f"❌ Failed to start dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()