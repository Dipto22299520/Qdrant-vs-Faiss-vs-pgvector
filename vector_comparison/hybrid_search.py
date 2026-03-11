#!/usr/bin/env python3
"""
Hybrid Search: BGE-M3 (Semantic) + BM25 (Keyword)
==================================================
Combines semantic similarity with keyword matching to improve accuracy.

Why Hybrid Search?
- Pure semantic search can return semantically similar but contextually wrong results
- Pure keyword search misses synonyms and related concepts
- Hybrid gets best of both worlds

Example:
    Query: "dull thump couple fallen"
    
    Pure Semantic (BGE-M3):
        → Might return "heap of bones collapsed" (similar concepts)
    
    Pure Keyword (BM25):
        → Might miss "loud crash couple tumbled" (synonyms)
    
    Hybrid:
        → Returns exact passage with "dull thump couple fallen" ✅
"""

import time
import psycopg2
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional
import re
from collections import Counter
import math

class HybridSearcher:
    """
    Combines semantic search (BGE-M3) with keyword search (BM25)
    """
    
    def __init__(
        self, 
        model: SentenceTransformer,
        conn: psycopg2.extensions.connection,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        device: str = "cuda"
    ):
        """
        Initialize hybrid searcher.
        
        Args:
            model: Pre-loaded SentenceTransformer model
            conn: PostgreSQL connection
            semantic_weight: Weight for semantic scores (0-1)
            keyword_weight: Weight for keyword scores (0-1)
            device: Device for model (cuda/cpu)
        """
        self.model = model
        self.conn = conn
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.device = device
        
        # BM25 parameters
        self.k1 = 1.5  # Term frequency saturation
        self.b = 0.75  # Length normalization
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        # Lowercase and split on non-alphanumeric
        tokens = re.findall(r'\w+', text.lower())
        return tokens
    
    def _compute_bm25_scores(self, query: str, documents: List[Dict]) -> List[float]:
        """
        Compute BM25 scores for query against documents.
        
        BM25 formula:
            score(D,Q) = Σ IDF(qi) × (f(qi,D) × (k1+1)) / (f(qi,D) + k1×(1-b+b×|D|/avgdl))
        
        Where:
            - IDF: Inverse document frequency (rare terms score higher)
            - f(qi,D): Frequency of term qi in document D
            - |D|: Length of document D
            - avgdl: Average document length
            - k1, b: Tuning parameters
        """
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        # Tokenize all documents
        doc_tokens = [self._tokenize(doc['text']) for doc in documents]
        doc_lengths = [len(tokens) for tokens in doc_tokens]
        avg_doc_length = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 1
        
        # Compute document frequencies (how many docs contain each term)
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
            # IDF = log((N - df + 0.5) / (df + 0.5) + 1)
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
            book_filter: Optional book_id or book_title filter
        
        Returns:
            (results, elapsed_ms)
        """
        t0 = time.time()
        
        # Step 1: Get semantic embedding
        query_embedding = self.model.encode(query, normalize_embeddings=True)
        query_list = query_embedding.tolist()
        query_vec_str = "[" + ",".join(str(x) for x in query_list) + "]"
        
        # Step 2: Retrieve all candidates (or filtered)
        cur = self.conn.cursor()
        
        # Increase ef_search for better recall (default is 40)
        cur.execute("SET LOCAL hnsw.ef_search = 200;")
        
        if book_filter:
            # Filter by book_id or book_title
            cur.execute("""
                SELECT text, book_id, book_title, chapter_num, source_file,
                       1 - (embedding <=> %s::vector) AS semantic_score
                FROM book_chunks
                WHERE book_id ILIKE %s OR book_title ILIKE %s
                ORDER BY embedding <=> %s::vector
                LIMIT 100
            """, (query_vec_str, f"%{book_filter}%", f"%{book_filter}%", query_vec_str))
        else:
            # Get top 100 semantic candidates for BM25 scoring
            cur.execute("""
                SELECT text, book_id, book_title, chapter_num, source_file,
                       1 - (embedding <=> %s::vector) AS semantic_score
                FROM book_chunks
                ORDER BY embedding <=> %s::vector
                LIMIT 100
            """, (query_vec_str, query_vec_str))
        
        rows = cur.fetchall()
        cur.close()
        
        # Convert to dict format
        candidates = [
            {
                'text': row[0],
                'book_id': row[1] or "",
                'book_title': row[2] or "",
                'chapter_num': row[3] or 0,
                'source_file': row[4] or "",
                'semantic_score': float(row[5])
            }
            for row in rows
        ]
        
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
    
    def print_results(self, results: List[Dict]):
        """Pretty print search results"""
        print("=" * 80)
        print("SEARCH RESULTS")
        print("=" * 80)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['book_title']} ({result['source_file']})")
            print(f"   Hybrid Score:   {result['hybrid_score']:.4f}")
            print(f"   ├─ Semantic:    {result['semantic_score']:.4f} (×{self.semantic_weight})")
            print(f"   └─ Keyword:     {result['keyword_score']:.4f} (×{self.keyword_weight})")
            print(f"   Text: {result['text'][:150]}...")
        
        print("\n" + "=" * 80)


def main():
    """Demo of hybrid search"""
    
    # PostgreSQL config
    pg_config = {
        'host': 'localhost',
        'port': 5432,
        'database': 'booksearch',
        'user': 'admin',
        'password': 'admin123'
    }
    
    # Initialize searcher
    searcher = HybridSearcher(
        pg_config=pg_config,
        semantic_weight=0.7,  # 70% semantic
        keyword_weight=0.3    # 30% keyword
    )
    
    # Test query (from pg110.txt - Tess of the d'Urbervilles)
    query = """Suddenly there was a dull thump on the ground: a couple had fallen, 
    and lay in a mixed heap. The next couple, unable to check its progress, 
    came toppling over the obstacle."""
    
    print("\n" + "=" * 80)
    print("HYBRID SEARCH DEMO")
    print("=" * 80)
    print(f"\nQuery: {query[:100]}...")
    
    # Search
    results = searcher.search(query, top_k=5, retrieve_k=50)
    
    # Display
    searcher.print_results(results)
    
    # Compare with pure semantic
    print("\n\n📊 COMPARISON: What would pure semantic return?")
    print("(This is what you were seeing before)")
    results_semantic = sorted(results, key=lambda x: x['semantic_score'], reverse=True)[:3]
    for i, r in enumerate(results_semantic, 1):
        print(f"{i}. {r['book_title']} - Semantic: {r['semantic_score']:.4f}")


if __name__ == "__main__":
    main()
