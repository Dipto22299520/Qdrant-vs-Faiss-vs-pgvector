#!/usr/bin/env python3
"""
Rebuild All Databases - Recreate PostgreSQL, Qdrant, and FAISS
==============================================================
Rebuilds all three vector databases with identical data:
  - PostgreSQL (8,460 chunks with HNSW index)
  - Qdrant (8,460 vectors)
  - FAISS (8,460 vectors with metadata)

Strategy: 20 chunks per book (5 pages each) from first 100 pages
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("\n" + "=" * 80)
    print("🔨 REBUILD ALL DATABASES")
    print("=" * 80)
    
    script_path = Path(__file__).parent / "vector_comparison" / "fast_rebuild.py"
    
    if not script_path.exists():
        print(f"\n❌ Error: {script_path} not found!")
        return 1
    
    print("\n⚠️  This will DELETE all existing data and rebuild:")
    print("   • PostgreSQL book_chunks table")
    print("   • Qdrant books_bgem3 collection")
    print("   • FAISS index and metadata")
    print()
    print("   Expected: 423 books × 20 chunks = 8,460 total chunks")
    print("   Estimated time: 60 minutes on RTX 5080")
    print()
    
    response = input("Type 'REBUILD' to confirm: ")
    if response.strip().upper() != "REBUILD":
        print("\n❌ Cancelled.")
        return 0
    
    print("\n✅ Starting rebuild...\n")
    
    # Auto-confirm by piping "REBUILD" to stdin
    result = subprocess.run(
        [sys.executable, str(script_path)],
        input="REBUILD\n",
        text=True,
        cwd=str(script_path.parent)
    )
    
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())
