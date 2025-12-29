#!/usr/bin/env python3
"""
Quick start script for Phase 5 API
Run locally without Docker for development
"""
import subprocess
import sys
from pathlib import Path

def main():
    # Ensure directories exist
    for d in ["data", "artifacts", "models", "plots", "reports", "logs"]:
        Path(d).mkdir(exist_ok=True)
    
    print("Starting EcomOpti Phase 5 API...")
    print("Open http://localhost:8000/docs for Swagger UI")
    print("Press Ctrl+C to stop\n")
    
    cmd = [
        "uvicorn",
        "src.ecomopti.phase5.main:app",
        "--reload",  # Auto-reload on code changes
        "--host", "0.0.0.0",
        "--port", "8000",
        "--log-level", "info"
    ]
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nAPI stopped.")
        sys.exit(0)

if __name__ == "__main__":
    main()