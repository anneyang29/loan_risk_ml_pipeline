"""
C3 Challenger: Decision Tuning  (legacy convenience wrapper)
=============================================================
Core logic has been moved to utils/challenger_manager.py.
Run via:
    python main.py --run-c3
or directly:
    python experiments/c3_decision_tuning/run_decision_tuning.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.challenger_manager import run_c3_decision_tuning_challenger

if __name__ == "__main__":
    result = run_c3_decision_tuning_challenger(project_root=PROJECT_ROOT)
    if result:
        print(f"C3 complete.  Output    : {result.get('output_dir', 'N/A')}")
        print(f"              Best model: {result.get('best_candidate', 'N/A')}")
        print(f"              Upgrade   : {result.get('upgrade_candidate', 'N/A')}")
