#!/usr/bin/env python3
"""
Hybrid Search for Qdrant: BGE-M3 (Semantic) + BM25 (Keyword)
=============================================================
Adapted for Qdrant vector database.
"""

import time
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from typing import List, Dict, Tuple, Optional
import re
from collections import Counter
import math

class HybridSearcherQdrant:
    """
    Combines semantic search (BGE-M3 via Qdrant) with keyword search (BM25)
    """
    
    def __init__(
        self, 
        model: SentenceTransformer,
        qdrant_client: QdrantClient,
        collection_name: str,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        device: str = "cuda"
    ):
        """
        Initialize hybrid searcher for Qdrant.
        
        Args:
            model: Pre-loaded SentenceTransformer model
            qdrant_client: Qdrant client instance
            collection_name: Name of Qdrant collection
            semantic_weight: Weight for semantic scores (0-1)
            keyword_weight: Weight for keyword scores (0-1)
            device: Device for model (cuda/cpu)
        """
        self.model = model
        self.qdrant = qdrant_client
        self.collection_name = collection_name
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.device = device
        
        # BM25 parameters
        self.k1 = 1.5  # Term frequency saturation
        self.b = 0.75  # Length normalization
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        tokens = re.findall(r'\w+', text.lower())
        return tokens
    
    def _compute_bm25_scores(self, query: str, documents: List[Dict]) -> List[float]:
        """
        Compute BM25 scores for query against documents.
        """
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        # Tokenize all documents
        doc_tokens = [self._tokenize(doc['text']) for doc in documents]
        doc_lengths = [len(tokens) for tokens in doc_tokens]
        avg_doc_length = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 1
        
        # Compute document frequencies
        doc_freqs = Counter()
        for tokens in doc_tokens:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                doc_freqs[token] += 1
        
        # Compute IDF for query terms
        num_docs = len(documents)
        idf = {}
        for token in query_tokens:
            df = doc_freqs.get(token, 0)
            idf[token] = math.log((num_docs - df + 0.5) / (df + 0.5) + 1)
        
        # Compute BM25 score for each document
        scores = []
        for tokens, doc_len in zip(doc_tokens, doc_lengths):
            score = 0.0
            term_freqs = Counter(tokens)
            
            for query_token in query_tokens:
                if query_token not in term_freqs:
                    continue
                
                tf = term_freqs[query_token]
                idf_val = idf.get(query_token, 0)
                
                # BM25 formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / avg_doc_length)
                score += idf_val * (numerator / denominator)
            
            scores.append(score)
        
        # Normalize scores to 0-1 range
        if scores:
            max_score = max(scores)
            if max_score > 0:
                scores = [s / max_score for s in scores]
        
        return scores
    
    def search(
        self, 
        query: str, 
        top_k: int = 5,
        book_filter: Optional[str] = None
    ) -> Tuple[List[Dict], float]:
        """
        Hybrid search combining semantic + keyword matching.
        
        Args:
            query: Search query
            top_k: Number of final results to return
            book_filter: Optional book filter
        
        Returns:
            (results, elapsed_ms)
        """
        t0 = time.time()
        
        # Step 1: Get semantic embedding
        query_embedding = self.model.encode(query, normalize_embeddings=True)
        
        # Step 2: Retrieve candidates from Qdrant
        search_params = {
            "collection_name": self.collection_name,
            "query_vector": query_embedding.tolist(),
            "limit": 100  # Get more candidates for BM25
        }
        
        if book_filter:
            search_params["query_filter"] = {
                "should": [
                    {"key": "book_id", "match": {"text": book_filter}},
                    {"key": "source_file", "match": {"text": book_filter}}
                ]
            }
        
        search_results = self.qdrant.search(**search_params)
        
        # Convert to dict format
        candidates = []
        for hit in search_results:
            candidates.append({
                'text': hit.payload.get('text', ''),
                'book_id': hit.payload.get('book_id', ''),
                'book_title': hit.payload.get('book_title', ''),
                'chapter_num': hit.payload.get('chapter_num', 0),
                'source_file': hit.payload.get('source_file', ''),
                'semantic_score': float(hit.score)
            })
        
        if not candidates:
            return [], (time.time() - t0) * 1000
        
        # Step 3: Compute BM25 (keyword) scores
        keyword_scores = self._compute_bm25_scores(query, candidates)
        
        # Add keyword scores to candidates
        for candidate, kw_score in zip(candidates, keyword_scores):
            candidate['keyword_score'] = kw_score
        
        # Step 4: Compute hybrid scores
        for candidate in candidates:
            candidate['hybrid_score'] = (
                self.semantic_weight * candidate['semantic_score'] +
                self.keyword_weight * candidate['keyword_score']
            )
        
        # Step 5: Sort by hybrid score and return top_k
        candidates.sort(key=lambda x: x['hybrid_score'], reverse=True)
        results = candidates[:top_k]
        
        elapsed_ms = (time.time() - t0) * 1000
        
        return results, elapsed_ms
