#!/usr/bin/env python3
"""
Configuration Loader for Book Search System
Loads configuration from .env file
"""

import os
from pathlib import Path
from typing import Optional

# Try to load python-dotenv if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv not installed. Using environment variables only.")
    print("   Install with: pip install python-dotenv")


class Config:
    """Centralized configuration for the entire system"""
    
    # =============================================================================
    # EMBEDDING MODEL
    # =============================================================================
    EMBED_MODEL: str = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
    EMBED_DIM: int = int(os.getenv("EMBED_DIM", "1024"))
    EMBED_BATCH_SIZE: int = int(os.getenv("EMBED_BATCH_SIZE", "64"))
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    DEVICE: str = os.getenv("DEVICE", "cuda")
    
    # =============================================================================
    # POSTGRESQL
    # =============================================================================
    PG_HOST: str = os.getenv("PG_HOST", "localhost")
    PG_PORT: int = int(os.getenv("PG_PORT", "5432"))
    PG_DB: str = os.getenv("PG_DB", "booksearch")
    PG_USER: str = os.getenv("PG_USER", "admin")
    PG_PASS: str = os.getenv("PG_PASS", "admin123")
    PG_COMMIT_EVERY: int = int(os.getenv("PG_COMMIT_EVERY", "5000"))
    
    @classmethod
    def get_pg_config(cls) -> dict:
        """Get PostgreSQL connection config as dict"""
        return {
            'host': cls.PG_HOST,
            'port': cls.PG_PORT,
            'database': cls.PG_DB,
            'user': cls.PG_USER,
            'password': cls.PG_PASS
        }
    
    # =============================================================================
    # QDRANT
    # =============================================================================
    QDRANT_PATH: str = os.getenv("QDRANT_PATH", "./qdrant_bgem3")
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "books_bgem3")
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_BATCH_SIZE: int = int(os.getenv("QDRANT_BATCH_SIZE", "100"))
    
    # =============================================================================
    # FAISS
    # =============================================================================
    FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "./faiss_bgem3")
    FAISS_HNSW_M: int = int(os.getenv("FAISS_HNSW_M", "32"))
    FAISS_EF_CONSTRUCTION: int = int(os.getenv("FAISS_EF_CONSTRUCTION", "40"))
    
    # =============================================================================
    # API SERVERS
    # =============================================================================
    API_PORT_QDRANT: int = int(os.getenv("API_PORT_QDRANT", "8081"))
    API_PORT_PGVECTOR_FAISS: int = int(os.getenv("API_PORT_PGVECTOR_FAISS", "8082"))
    API_PORT_PGVECTOR_ONLY: int = int(os.getenv("API_PORT_PGVECTOR_ONLY", "8083"))
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_RELOAD: bool = os.getenv("API_RELOAD", "False").lower() == "true"
    API_WORKERS: int = int(os.getenv("API_WORKERS", "1"))
    
    # =============================================================================
    # BOOK PROCESSING
    # =============================================================================
    BOOKS_DIR: str = os.getenv("BOOKS_DIR", r"C:\Users\Administrator\Downloads\wget installation (1)\wget installation\gutenberg_auto\books_utf8")
    CHARS_PER_PAGE: int = int(os.getenv("CHARS_PER_PAGE", "3000"))
    MAX_PAGES: int = int(os.getenv("MAX_PAGES", "100"))
    NUM_CHUNKS_PER_BOOK: int = int(os.getenv("NUM_CHUNKS_PER_BOOK", "4"))
    MAX_CHUNK_TOKENS: int = int(os.getenv("MAX_CHUNK_TOKENS", "512"))
    MIN_CHUNK_TOKENS: int = int(os.getenv("MIN_CHUNK_TOKENS", "50"))
    
    # =============================================================================
    # HYBRID SEARCH
    # =============================================================================
    DEFAULT_SEMANTIC_WEIGHT: float = float(os.getenv("DEFAULT_SEMANTIC_WEIGHT", "0.7"))
    DEFAULT_KEYWORD_WEIGHT: float = float(os.getenv("DEFAULT_KEYWORD_WEIGHT", "0.3"))
    BM25_K1: float = float(os.getenv("BM25_K1", "1.5"))
    BM25_B: float = float(os.getenv("BM25_B", "0.75"))
    DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "5"))
    DEFAULT_RETRIEVE_K: int = int(os.getenv("DEFAULT_RETRIEVE_K", "10"))
    
    # =============================================================================
    # GOOGLE GEMINI (Optional)
    # =============================================================================
    OPENROUTER_API_KEY: Optional[str] = os.getenv("OPENROUTER_API_KEY")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "google/gemini-embedding-001")
    GEMINI_EMBED_DIM: int = int(os.getenv("GEMINI_EMBED_DIM", "3072"))
    
    # =============================================================================
    # CHECKPOINTS
    # =============================================================================
    CHECKPOINT_DIR: str = os.getenv("CHECKPOINT_DIR", "./pipeline_checkpoints")
    REBUILD_CHECKPOINT_DIR: str = os.getenv("REBUILD_CHECKPOINT_DIR", "./rebuild_checkpoints")
    EMBEDDINGS_CACHE: str = os.getenv("EMBEDDINGS_CACHE", "embeddings_cleaned.npz")
    
    # =============================================================================
    # TESTING
    # =============================================================================
    TEST_SET_FILE: str = os.getenv("TEST_SET_FILE", "bangla_test_set_500.json")
    TEST_RESULTS_FILE: str = os.getenv("TEST_RESULTS_FILE", "bangla_test_results.json")
    TEST_REPORT_FILE: str = os.getenv("TEST_REPORT_FILE", "bangla_test_report.txt")
    
    # =============================================================================
    # LOGGING
    # =============================================================================
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "book_search.log")
    
    # =============================================================================
    # PERFORMANCE
    # =============================================================================
    NUM_WORKERS: int = int(os.getenv("NUM_WORKERS", "4"))
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("=" * 80)
        print("CONFIGURATION")
        print("=" * 80)
        print(f"\n📊 Embedding:")
        print(f"   Model: {cls.EMBED_MODEL}")
        print(f"   Dimension: {cls.EMBED_DIM}")
        print(f"   Batch Size: {cls.EMBED_BATCH_SIZE}")
        print(f"   Device: {cls.DEVICE}")
        
        print(f"\n🗄️  PostgreSQL:")
        print(f"   Host: {cls.PG_HOST}:{cls.PG_PORT}")
        print(f"   Database: {cls.PG_DB}")
        print(f"   User: {cls.PG_USER}")
        
        print(f"\n🔷 Qdrant:")
        print(f"   Path: {cls.QDRANT_PATH}")
        print(f"   Collection: {cls.QDRANT_COLLECTION}")
        
        print(f"\n🔶 FAISS:")
        print(f"   Index Path: {cls.FAISS_INDEX_PATH}")
        
        print(f"\n🌐 API Ports:")
        print(f"   Qdrant: {cls.API_PORT_QDRANT}")
        print(f"   FAISS+pgvector: {cls.API_PORT_PGVECTOR_FAISS}")
        print(f"   pgvector Only: {cls.API_PORT_PGVECTOR_ONLY}")
        
        print(f"\n📚 Books:")
        print(f"   Directory: {cls.BOOKS_DIR}")
        print(f"   Max Pages: {cls.MAX_PAGES}")
        print(f"   Chunks per Book: {cls.NUM_CHUNKS_PER_BOOK}")
        
        print(f"\n🔍 Hybrid Search:")
        print(f"   Semantic Weight: {cls.DEFAULT_SEMANTIC_WEIGHT}")
        print(f"   Keyword Weight: {cls.DEFAULT_KEYWORD_WEIGHT}")
        print(f"   BM25 k1: {cls.BM25_K1}, b: {cls.BM25_B}")
        
        print("\n" + "=" * 80)
    
    @classmethod
    def validate(cls):
        """Validate critical configuration"""
        errors = []
        
        # Check if books directory exists
        if not os.path.isdir(cls.BOOKS_DIR):
            errors.append(f"Books directory not found: {cls.BOOKS_DIR}")
        
        # Check weights sum to 1
        weight_sum = cls.DEFAULT_SEMANTIC_WEIGHT + cls.DEFAULT_KEYWORD_WEIGHT
        if abs(weight_sum - 1.0) > 0.01:
            errors.append(f"Semantic + Keyword weights must sum to 1.0 (got {weight_sum})")
        
        if errors:
            print("\n❌ Configuration Errors:")
            for error in errors:
                print(f"   - {error}")
            return False
        
        print("✅ Configuration validated")
        return True


# Convenience function
def get_config() -> Config:
    """Get configuration instance"""
    return Config


if __name__ == "__main__":
    # Test configuration
    config = get_config()
    config.print_config()
    config.validate()
