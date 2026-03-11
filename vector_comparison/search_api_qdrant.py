#!/usr/bin/env python3
"""
Option A: FastAPI + Qdrant Vector DB
=====================================
Simple architecture: query -> BGE-M3 -> Qdrant -> Reranker -> best 3

No Docker needed. No PostgreSQL. Just Python.

Run:  python search_api_qdrant.py
Docs: http://localhost:8081/docs
"""

import os, time, asyncio
from typing import List, Optional
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from reranker import BGEReranker
from hybrid_search_qdrant import HybridSearcherQdrant

# ---- CONFIG ----
EMBED_MODEL       = "BAAI/bge-m3"
EMBED_DIM         = 1024
QDRANT_PATH       = "./qdrant_bgem3"
QDRANT_COLLECTION = "books_bgem3"
RERANKER_MODEL    = "BAAI/bge-reranker-v2-m3"
RETRIEVE_K        = 20
FINAL_TOP_K       = 3

# ---- Schemas ----
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000)
    top_k: int = Field(default=FINAL_TOP_K, ge=1, le=20)
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
    similarity_score: float
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
    retrieval_ms: float
    rerank_ms: float
    total_ms: float
    backend: str = "qdrant"

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
_qdrant = None
_hybrid_searcher = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _reranker, _qdrant, _hybrid_searcher
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("  Option A: FastAPI + Qdrant")
    print("=" * 60)

    t0 = time.time()
    _model = SentenceTransformer(EMBED_MODEL, device=device)
    print(f"  Embedding model loaded in {time.time()-t0:.1f}s")

    t0 = time.time()
    _reranker = BGEReranker(RERANKER_MODEL, device=device, use_fp16=(device=="cuda"))
    print(f"  Reranker loaded in {time.time()-t0:.1f}s")

    _qdrant = QdrantClient(path=QDRANT_PATH)
    info = _qdrant.get_collection(QDRANT_COLLECTION)
    print(f"  Qdrant: {info.points_count:,} vectors")

    # Initialize hybrid searcher
    _hybrid_searcher = HybridSearcherQdrant(
        model=_model,
        qdrant_client=_qdrant,
        collection_name=QDRANT_COLLECTION,
        semantic_weight=0.7,
        keyword_weight=0.3,
        device=device
    )
    print(f"  Hybrid search initialized")

    _model.encode(["warmup"], normalize_embeddings=True)
    _reranker.compute_score([["w", "w"]])

    print(f"\n  READY on http://localhost:8081/docs\n")
    yield
    if _qdrant:
        _qdrant.close()

app = FastAPI(title="Book Search - Qdrant", version="1.0", lifespan=lifespan,
              description="**Option A:** query -> BGE-M3 -> Qdrant (20) -> BGE Reranker -> best 3")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    total_t0 = time.time()

    # 1. Embed
    query_vec = _model.encode(req.query, normalize_embeddings=True).tolist()

    # 2. Qdrant search
    search_filter = None
    if req.book_filter:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        search_filter = Filter(must=[FieldCondition(key="book_title", match=MatchValue(value=req.book_filter))])

    t0 = time.time()
    resp = _qdrant.query_points(
        collection_name=QDRANT_COLLECTION, query=query_vec,
        limit=RETRIEVE_K, query_filter=search_filter,
    )
    retrieval_ms = (time.time() - t0) * 1000

    candidates = []
    for pt in resp.points:
        p = pt.payload
        candidates.append({
            "text": p["text"], "book_title": p.get("book_title", ""),
            "book_id": p.get("book_id", ""), "chapter_num": p.get("chapter_num", 0),
            "source_file": p.get("source_file", ""), "similarity_score": pt.score,
        })

    # 3. Rerank
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

    # 4. Return top_k
    items = []
    for rank, c in enumerate(candidates[:req.top_k], 1):
        items.append(ResultItem(rank=rank, **{k: c[k] for k in ResultItem.model_fields if k != "rank"}))

    return SearchResponse(
        query=req.query, results=items, total_candidates=len(candidates),
        retrieval_ms=round(retrieval_ms, 2), rerank_ms=round(rerank_ms, 2),
        total_ms=round((time.time()-total_t0)*1000, 2), backend="qdrant",
    )

@app.post("/hybrid_search", response_model=HybridSearchResponse)
async def hybrid_search(req: HybridSearchRequest):
    """
    Hybrid search endpoint combining semantic (BGE-M3) and keyword (BM25) scoring.
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
        backend="qdrant-hybrid",
    )

@app.get("/health")
async def health():
    info = _qdrant.get_collection(QDRANT_COLLECTION)
    return {"status": "ok", "backend": "qdrant", "vectors": info.points_count,
            "model": EMBED_MODEL, "reranker": RERANKER_MODEL}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("search_api_qdrant:app", host="0.0.0.0", port=8081, reload=False)
