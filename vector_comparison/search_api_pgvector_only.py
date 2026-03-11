#!/usr/bin/env python3
"""
Simplified: FastAPI + pgvector only
====================================
Architecture: query -> BGE-M3 embedding -> pgvector retrieval -> BGE reranker -> top results

No FAISS needed - pgvector handles all similarity search.

Requires: Docker PostgreSQL (pgvector-db) running on port 5432.

Run:  python search_api_pgvector_only.py
Docs: http://localhost:8083/docs
"""

import time
from typing import List, Optional
from contextlib import asynccontextmanager

import torch
import psycopg2
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from reranker import BGEReranker
from hybrid_search import HybridSearcher

# ---- CONFIG ----
EMBED_MODEL      = "BAAI/bge-m3"
EMBED_DIM        = 1024
RERANKER_MODEL   = "BAAI/bge-reranker-v2-m3"

PG_HOST = "localhost"
PG_PORT = 5432
PG_DB   = "booksearch"
PG_USER = "admin"
PG_PASS = "admin123"

RETRIEVE_K  = 20    # Retrieve top 20 candidates from pgvector
FINAL_TOP_K = 3     # Return top 3 after reranking

# ---- Schemas ----
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000)
    top_k: int = Field(default=FINAL_TOP_K, ge=1, le=20)
    retrieve_k: int = Field(default=RETRIEVE_K, ge=1, le=100)
    book_filter: Optional[str] = None

class HybridSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000)
    top_k: int = Field(default=FINAL_TOP_K, ge=1, le=20)
    semantic_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    keyword_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    book_filter: Optional[str] = None

class ResultItem(BaseModel):
    rank: int
    text: str
    book_title: str
    book_id: str
    chapter_num: int
    source_file: str
    pgvector_score: float
    rerank_score: float

class HybridResultItem(BaseModel):
    rank: int
    text: str
    book_title: str
    book_id: str
    chapter_num: int
    source_file: str
    semantic_score: float
    keyword_score: float
    hybrid_score: float

class SearchResponse(BaseModel):
    query: str
    results: List[ResultItem]
    total_candidates: int
    pgvector_ms: float
    rerank_ms: float
    total_ms: float
    backend: str

class HybridSearchResponse(BaseModel):
    query: str
    results: List[HybridResultItem]
    total_chunks: int
    semantic_weight: float
    keyword_weight: float
    search_ms: float
    total_ms: float
    backend: str

# ---- Global state ----
_model = None
_reranker = None
_pg_conn = None
_hybrid_searcher = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    global _model, _reranker, _pg_conn, _hybrid_searcher
    
    print("🚀 Loading BGE-M3 embedding model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _model = SentenceTransformer(EMBED_MODEL, device=device)
    print(f"✅ BGE-M3 loaded on {device}")
    
    print("🚀 Loading BGE-reranker-v2-m3...")
    _reranker = BGEReranker(RERANKER_MODEL, device=device)
    print(f"✅ Reranker loaded on {device}")
    
    print("🚀 Connecting to PostgreSQL...")
    _pg_conn = psycopg2.connect(
        host=PG_HOST, port=PG_PORT, dbname=PG_DB, 
        user=PG_USER, password=PG_PASS
    )
    print("✅ PostgreSQL connected")
    
    # Check table exists
    cur = _pg_conn.cursor()
    cur.execute("SELECT COUNT(*) FROM book_chunks")
    count = cur.fetchone()[0]
    cur.close()
    print(f"✅ Found {count:,} chunks in pgvector")
    
    # Initialize hybrid searcher
    print("🚀 Initializing hybrid search (BM25 + BGE-M3)...")
    _hybrid_searcher = HybridSearcher(
        model=_model,
        conn=_pg_conn,
        semantic_weight=0.7,
        keyword_weight=0.3,
        device=device
    )
    print("✅ Hybrid search initialized")
    
    yield
    
    # Cleanup
    if _pg_conn:
        _pg_conn.close()
    print("👋 Shutdown complete")

app = FastAPI(
    title="Semantic Book Search (pgvector only)",
    description="BGE-M3 embeddings + pgvector + BGE reranker",
    version="2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Search Logic ----
def _search_pgvector(query_vec_str: str, k: int, book_filter: Optional[str] = None):
    """
    Search pgvector using cosine distance (<=> operator).
    
    Args:
        query_vec_str: Embedding as PostgreSQL array string "[0.1, 0.2, ...]"
        k: Number of results to retrieve
        book_filter: Optional book_id or book_title substring to filter by
    
    Returns:
        (results, elapsed_ms)
    """
    cur = _pg_conn.cursor()
    t0 = time.time()
    
    if book_filter:
        # Filter by book_id or book_title
        cur.execute("""
            SELECT text, book_id, book_title, chapter_num, source_file,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM book_chunks
            WHERE book_id ILIKE %s OR book_title ILIKE %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (query_vec_str, f"%{book_filter}%", f"%{book_filter}%", query_vec_str, k))
    else:
        # No filter - search all books
        cur.execute("""
            SELECT text, book_id, book_title, chapter_num, source_file,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM book_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (query_vec_str, query_vec_str, k))

    rows = cur.fetchall()
    cur.close()
    elapsed_ms = (time.time() - t0) * 1000

    results = []
    for text, book_id, book_title, chapter_num, source_file, similarity in rows:
        results.append({
            "text": text,
            "book_id": book_id or "",
            "book_title": book_title or "",
            "chapter_num": chapter_num or 0,
            "source_file": source_file or "",
            "pgvector_score": float(similarity),
        })
    return results, elapsed_ms


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    """
    Semantic search endpoint.
    
    Flow:
    1. Embed query with BGE-M3
    2. Retrieve top candidates from pgvector (cosine similarity)
    3. Rerank candidates with BGE-reranker-v2-m3
    4. Return top_k results
    """
    total_t0 = time.time()

    # 1. Embed query
    query_np = _model.encode(req.query, normalize_embeddings=True)
    query_list = query_np.tolist()
    query_vec_str = "[" + ",".join(str(x) for x in query_list) + "]"

    # 2. Retrieve from pgvector
    candidates, pgvector_ms = _search_pgvector(
        query_vec_str, 
        req.retrieve_k, 
        req.book_filter
    )

    # 3. Rerank with BGE-reranker
    t0 = time.time()
    if candidates:
        pairs = [[req.query, c["text"]] for c in candidates]
        scores = _reranker.compute_score(pairs)
        
        # Handle single result case
        if isinstance(scores, (int, float)):
            scores = [scores]
        
        # Attach rerank scores
        for c, score in zip(candidates, scores):
            c["rerank_score"] = float(score)
        
        # Sort by rerank score (higher is better)
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    rerank_ms = (time.time() - t0) * 1000

    # 4. Build response
    items = []
    for rank, c in enumerate(candidates[:req.top_k], 1):
        items.append(ResultItem(
            rank=rank,
            text=c["text"],
            book_title=c["book_title"],
            book_id=c["book_id"],
            chapter_num=c["chapter_num"],
            source_file=c["source_file"],
            pgvector_score=c["pgvector_score"],
            rerank_score=c["rerank_score"],
        ))

    total_ms = (time.time() - total_t0) * 1000

    return SearchResponse(
        query=req.query,
        results=items,
        total_candidates=len(candidates),
        pgvector_ms=round(pgvector_ms, 2),
        rerank_ms=round(rerank_ms, 2),
        total_ms=round(total_ms, 2),
        backend="pgvector-only",
    )


@app.post("/hybrid_search", response_model=HybridSearchResponse)
async def hybrid_search(req: HybridSearchRequest):
    """
    Hybrid search endpoint combining semantic (BGE-M3) and keyword (BM25) scoring.
    
    Flow:
    1. Retrieve all chunks from database (or filtered by book)
    2. Compute semantic scores using BGE-M3 embeddings
    3. Compute keyword scores using BM25 algorithm
    4. Combine scores: hybrid = semantic_weight × semantic + keyword_weight × keyword
    5. Return top_k results sorted by hybrid score
    
    Note: Weights should sum to 1.0 (default: 0.7 semantic, 0.3 keyword)
    """
    total_t0 = time.time()
    
    # Update hybrid searcher weights if provided
    _hybrid_searcher.semantic_weight = req.semantic_weight
    _hybrid_searcher.keyword_weight = req.keyword_weight
    
    # Perform hybrid search
    results, search_ms = _hybrid_searcher.search(
        query=req.query,
        top_k=req.top_k,
        book_filter=req.book_filter
    )
    
    # Build response
    items = []
    for rank, res in enumerate(results, 1):
        items.append(HybridResultItem(
            rank=rank,
            text=res["text"],
            book_title=res["book_title"],
            book_id=res["book_id"],
            chapter_num=res["chapter_num"],
            source_file=res["source_file"],
            semantic_score=res["semantic_score"],
            keyword_score=res["keyword_score"],
            hybrid_score=res["hybrid_score"],
        ))
    
    total_ms = (time.time() - total_t0) * 1000
    
    return HybridSearchResponse(
        query=req.query,
        results=items,
        total_chunks=len(results),
        semantic_weight=req.semantic_weight,
        keyword_weight=req.keyword_weight,
        search_ms=round(search_ms, 2),
        total_ms=round(total_ms, 2),
        backend="pgvector-hybrid",
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    cur = _pg_conn.cursor()
    cur.execute("SELECT COUNT(*) FROM book_chunks")
    count = cur.fetchone()[0]
    cur.close()
    
    return {
        "status": "ok",
        "backend": "pgvector-only",
        "total_chunks": count,
        "embed_model": EMBED_MODEL,
        "reranker_model": RERANKER_MODEL,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Semantic Book Search API (pgvector only)",
        "endpoints": {
            "search": "/search - Pure semantic search (BGE-M3 + reranker)",
            "hybrid_search": "/hybrid_search - Hybrid search (semantic + BM25 keyword)",
            "health": "/health - Health check",
            "docs": "/docs - API documentation"
        },
        "version": "2.1",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "search_api_pgvector_only:app", 
        host="0.0.0.0", 
        port=8083, 
        reload=False
    )
