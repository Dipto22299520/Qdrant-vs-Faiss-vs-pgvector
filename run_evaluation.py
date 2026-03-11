#!/usr/bin/env python3
"""
Run Evaluation - Test all APIs with Bangla test set
===================================================
Runs the comprehensive evaluation test against all three APIs
and generates accuracy/timing reports.
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("\n" + "=" * 80)
    print("🧪 RUNNING EVALUATION TEST")
    print("=" * 80)
    
    script_path = Path(__file__).parent / "evaluate_bangla_test_set.py"
    
    if not script_path.exists():
        print(f"\n❌ Error: {script_path} not found!")
        return 1
    
    print("\n📊 Running evaluate_bangla_test_set.py...")
    print("   This will test all 3 APIs with 442 Bangla queries\n")
    
    # Run evaluation
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(script_path.parent)
    )
    
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())
