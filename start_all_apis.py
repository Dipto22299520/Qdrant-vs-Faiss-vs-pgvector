#!/usr/bin/env python3
"""
Start All APIs - Launch all three search APIs simultaneously
============================================================
Starts:
  - Port 8081: Qdrant API
  - Port 8082: FAISS + pgvector API
  - Port 8083: pgvector-only API
"""

import subprocess
import sys
import time
from pathlib import Path

def main():
    print("\n" + "=" * 80)
    print("🚀 STARTING ALL SEARCH APIS")
    print("=" * 80)
    
    # Get vector_comparison directory
    vec_comp_dir = Path(__file__).parent / "vector_comparison"
    
    # API scripts
    apis = [
        {"name": "Qdrant", "port": 8081, "script": "search_api_qdrant.py"},
        {"name": "FAISS+pgvector", "port": 8082, "script": "search_api_pgvector.py"},
        {"name": "pgvector-only", "port": 8083, "script": "search_api_pgvector_only.py"},
    ]
    
    processes = []
    
    for api in apis:
        script_path = vec_comp_dir / api["script"]
        print(f"\n📡 Starting {api['name']} API on port {api['port']}...")
        
        # Start process
        process = subprocess.Popen(
            [sys.executable, str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(vec_comp_dir)
        )
        processes.append({"process": process, **api})
        print(f"   ✅ Process started (PID: {process.pid})")
    
    print("\n" + "=" * 80)
    print("✅ ALL APIS STARTED")
    print("=" * 80)
    print("\n⏳ Waiting 30 seconds for models to load...")
    time.sleep(30)
    
    print("\n" + "=" * 80)
    print("🎯 APIS READY")
    print("=" * 80)
    for api in apis:
        print(f"  • {api['name']}: http://localhost:{api['port']}/docs")
    
    print("\n💡 Tips:")
    print("  • Test with: python run_evaluation.py")
    print("  • UI at: http://localhost:9000/hybrid_search_ui.html")
    print("  • Press Ctrl+C to stop all APIs")
    
    print("\n" + "=" * 80)
    
    # Keep running until user stops
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping all APIs...")
        for p in processes:
            p["process"].terminate()
            print(f"   ✅ Stopped {p['name']} API")
        print("\n✅ All APIs stopped.\n")

if __name__ == "__main__":
    main()
