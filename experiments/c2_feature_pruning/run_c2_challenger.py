"""
C2 Challenger: Feature Pruning  (legacy convenience wrapper)
=============================================================
Core logic has been moved to utils/challenger_manager.py.
Run via:
    python main.py --run-c2
or directly:
    python experiments/c2_feature_pruning/run_c2_challenger.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.challenger_manager import run_c2_feature_pruning_challenger

if __name__ == "__main__":
    result = run_c2_feature_pruning_challenger(project_root=PROJECT_ROOT)
    if result:
        print(f"C2 complete.  Output : {result.get('output_dir', 'N/A')}")
        print(f"              Upgrade: {result.get('upgrade_candidate', 'N/A')}")
