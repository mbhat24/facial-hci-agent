"""Entrypoint: python scripts/run_dashboard.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard.server import run

if __name__ == "__main__":
    run()
