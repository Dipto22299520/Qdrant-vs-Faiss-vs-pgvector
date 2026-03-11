#!/usr/bin/env python3
"""
Option B: FastAPI + pgvector + FAISS
=====================================
Dual-backend: query -> BGE-M3 -> pgvector(10) + FAISS(10) -> merge -> Reranker -> best 3

Requires: Docker PostgreSQL (pgvector-db) running on port 5432.

Run:  python search_api_pgvector.py
Docs: http://localhost:8082/docs
"""

import os, time, pickle, asyncio
from typing import List, Optional
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import faiss
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
FAISS_INDEX_PATH = "./faiss_bgem3"

PG_HOST = "localhost"
PG_PORT = 5432
PG_DB   = "booksearch"
PG_USER = "admin"
PG_PASS = "admin123"

RETRIEVE_K  = 10    # per backend
FINAL_TOP_K = 3

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
    pgvector_score: Optional[float] = None
    faiss_score: Optional[float] = None
    rerank_score: float
    source: str   # "pgvector", "faiss", or "both"

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
    faiss_ms: float
    rerank_ms: float
    total_ms: float
    backend: str = "pgvector+faiss"

class HybridSearchResponse(BaseModel):
    query: str
    results: List[HybridResultItem]
    total_chunks: int
    semantic_weight: float
    keyword_weight: float
    search_ms: float
    total_ms: float
    backend: str

# ---- Globals ----
_model = None
_reranker = None
_pg_conn = None
_faiss_index = None
_faiss_texts = None
_faiss_meta = None
_pool = None
_hybrid_searcher = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _reranker, _pg_conn
    global _faiss_index, _faiss_texts, _faiss_meta, _pool, _hybrid_searcher
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("  Option B: FastAPI + pgvector + FAISS")
    print("=" * 60)

    t0 = time.time()
    _model = SentenceTransformer(EMBED_MODEL, device=device)
    print(f"  Embedding model loaded in {time.time()-t0:.1f}s")

    t0 = time.time()
    _reranker = BGEReranker(RERANKER_MODEL, device=device, use_fp16=(device=="cuda"))
    print(f"  Reranker loaded in {time.time()-t0:.1f}s")

    # pgvector
    _pg_conn = psycopg2.connect(host=PG_HOST, port=PG_PORT, dbname=PG_DB, user=PG_USER, password=PG_PASS)
    cur = _pg_conn.cursor()
    cur.execute("SELECT COUNT(*) FROM book_chunks")
    pg_count = cur.fetchone()[0]
    cur.close()
    print(f"  pgvector: {pg_count:,} rows in book_chunks")

    # FAISS
    t0 = time.time()
    _faiss_index = faiss.read_index(f"{FAISS_INDEX_PATH}.faiss")
    with open(f"{FAISS_INDEX_PATH}.meta", "rb") as f:
        meta = pickle.load(f)
    _faiss_texts = meta["texts"]
    _faiss_meta = meta["metadata"]
    print(f"  FAISS: {_faiss_index.ntotal:,} vectors loaded in {time.time()-t0:.1f}s")

    # Initialize hybrid searcher
    _hybrid_searcher = HybridSearcher(
        model=_model,
        conn=_pg_conn,
        semantic_weight=0.7,
        keyword_weight=0.3,
        device=device
    )
    print(f"  Hybrid search initialized")

    _model.encode(["warmup"], normalize_embeddings=True)
    _reranker.compute_score([["w", "w"]])
    _pool = ThreadPoolExecutor(max_workers=2)

    print(f"\n  READY on http://localhost:8082/docs\n")
    yield
    _pool.shutdown(wait=False)
    if _pg_conn:
        _pg_conn.close()

app = FastAPI(title="Book Search - pgvector+FAISS", version="1.0", lifespan=lifespan,
              description="**Option B:** query -> BGE-M3 -> pgvector(10) + FAISS(10) -> merge -> Reranker -> best 3")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _search_pgvector(query_vec_str, k, book_filter):
    """Search PostgreSQL pgvector using cosine similarity."""
    t0 = time.time()
    cur = _pg_conn.cursor()

    if book_filter:
        cur.execute("""
            SELECT text, book_id, book_title, chapter_num, source_file,
                   1 - (embedding <=> %s::vector) AS score
            FROM book_chunks
            WHERE book_title = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (query_vec_str, book_filter, query_vec_str, k))
    else:
        cur.execute("""
            SELECT text, book_id, book_title, chapter_num, source_file,
                   1 - (embedding <=> %s::vector) AS score
            FROM book_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (query_vec_str, query_vec_str, k))

    rows = cur.fetchall()
    cur.close()
    elapsed_ms = (time.time() - t0) * 1000

    results = []
    for text, book_id, book_title, chapter_num, source_file, score in rows:
        results.append({
            "text": text, "book_id": book_id or "", "book_title": book_title or "",
            "chapter_num": chapter_num or 0, "source_file": source_file or "",
            "score": float(score),
        })
    return results, elapsed_ms


def _search_faiss(query_vec_np, k):
    """Search FAISS index."""
    q = query_vec_np.copy().reshape(1, -1).astype("float32")
    faiss.normalize_L2(q)
    t0 = time.time()
    scores, indices = _faiss_index.search(q, k)
    elapsed_ms = (time.time() - t0) * 1000
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(_faiss_texts):
            continue
        m = _faiss_meta[idx]
        results.append({
            "text": _faiss_texts[idx], "book_id": m.get("book_id", ""),
            "book_title": m.get("book_title", ""), "chapter_num": m.get("chapter_num", 0),
            "source_file": m.get("source_file", ""), "score": float(score),
        })
    return results, elapsed_ms


def _merge(pg_results, faiss_results):
    """Merge and deduplicate results from both backends."""
    seen = {}
    for r in pg_results:
        key = hash(r["text"])
        if key not in seen:
            seen[key] = {**r, "pgvector_score": r["score"], "faiss_score": None, "source": "pgvector"}
        else:
            seen[key]["pgvector_score"] = r["score"]
            seen[key]["source"] = "both"
    for r in faiss_results:
        key = hash(r["text"])
        if key not in seen:
            seen[key] = {**r, "pgvector_score": None, "faiss_score": r["score"], "source": "faiss"}
        else:
            seen[key]["faiss_score"] = r["score"]
            if seen[key]["source"] == "pgvector":
                seen[key]["source"] = "both"
    return list(seen.values())


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    total_t0 = time.time()

    # 1. Embed
    query_np = _model.encode(req.query, normalize_embeddings=True)
    query_list = query_np.tolist()
    query_vec_str = "[" + ",".join(str(x) for x in query_list) + "]"

    # 2. Parallel search
    loop = asyncio.get_running_loop()
    pg_future = loop.run_in_executor(_pool, _search_pgvector, query_vec_str, req.retrieve_k, req.book_filter)
    faiss_future = loop.run_in_executor(_pool, _search_faiss, query_np, req.retrieve_k)
    pg_results, pg_ms = await pg_future
    faiss_results, faiss_ms = await faiss_future

    # 3. Merge
    candidates = _merge(pg_results, faiss_results)

    # 4. Rerank
    t0 = time.time()
    if candidates:
        pairs = [[req.query, c["text"]] for c in candidates]
        scores = _reranker.compute_score(pairs)
        if isinstance(scores, (int, float)):
            scores = [scores]
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    rerank_ms = (time.time() - t0) * 1000

    # 5. Return top_k
    items = []
    for rank, c in enumerate(candidates[:req.top_k], 1):
        items.append(ResultItem(
            rank=rank, text=c["text"], book_title=c.get("book_title", ""),
            book_id=c.get("book_id", ""), chapter_num=c.get("chapter_num", 0),
            source_file=c.get("source_file", ""),
            pgvector_score=c.get("pgvector_score"), faiss_score=c.get("faiss_score"),
            rerank_score=c.get("rerank_score", 0.0), source=c.get("source", ""),
        ))

    return SearchResponse(
        query=req.query, results=items, total_candidates=len(candidates),
        pgvector_ms=round(pg_ms, 2), faiss_ms=round(faiss_ms, 2),
        rerank_ms=round(rerank_ms, 2), total_ms=round((time.time()-total_t0)*1000, 2),
        backend="pgvector+faiss",
    )

@app.post("/hybrid_search", response_model=HybridSearchResponse)
async def hybrid_search(req: HybridSearchRequest):
    """
    Hybrid search endpoint combining semantic (BGE-M3) and keyword (BM25) scoring.
    Uses pgvector backend only for hybrid search.
    """
    total_t0 = time.time()
    
    # Update hybrid searcher weights
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
        backend="pgvector+faiss-hybrid",
    )

@app.get("/health")
async def health():
    cur = _pg_conn.cursor()
    cur.execute("SELECT COUNT(*) FROM book_chunks")
    pg_count = cur.fetchone()[0]
    cur.close()
    return {"status": "ok", "backend": "pgvector+faiss", "pgvector_rows": pg_count,
            "faiss_vectors": _faiss_index.ntotal, "model": EMBED_MODEL, "reranker": RERANKER_MODEL}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("search_api_pgvector:app", host="0.0.0.0", port=8082, reload=False)
