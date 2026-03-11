#!/usr/bin/env python3
"""
Fast Rebuild Pipeline - Simple 20-Chunk Per Book Strategy
=========================================================
For each book:
1. Read first 100 pages (~300,000 chars)
2. Split into 20 equal chunks of 5 pages each
3. Batch embed all 20 chunks at once
4. Upload to all backends
5. Print "book_name: done"

This is MUCH faster than complex chunking!
"""

import os
import sys
import time
import uuid
import numpy as np
import torch
import psycopg2
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import faiss
import pickle

# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════

EMBED_MODEL = "BAAI/bge-m3"
EMBED_DIM = 1024

# Books
BOOKS_DIR = r"C:\Users\Administrator\Downloads\wget installation (1)\wget installation\gutenberg_auto\books_utf8"

# PostgreSQL
PG_HOST = "localhost"
PG_PORT = 5432
PG_DB = "booksearch"
PG_USER = "admin"
PG_PASS = "admin123"

# Qdrant
QDRANT_PATH = "./qdrant_bgem3"
QDRANT_COLLECTION = "books_bgem3"

# FAISS
FAISS_INDEX_PATH = "./faiss_bgem3"

# Simple chunking
CHARS_PER_PAGE = 3000
MAX_PAGES = 100
NUM_CHUNKS_PER_BOOK = 20  # 20 chunks of 5 pages each

# ══════════════════════════════════════════════════════════════════

def confirm_rebuild():
    """Ask user to confirm"""
    print("\n" + "=" * 80)
    print("⚠️  WARNING: THIS WILL DELETE ALL EXISTING DATA!")
    print("=" * 80)
    print()
    print("Fast 20-chunk strategy:")
    print(f"  - Each book: {MAX_PAGES} pages → {NUM_CHUNKS_PER_BOOK} chunks of {MAX_PAGES//NUM_CHUNKS_PER_BOOK} pages")
    print(f"  - Source: {BOOKS_DIR}")
    print(f"  - Expected: ~423 books × 20 chunks = ~8,460 total chunks")
    print()
    print("Estimated time: 30-40 minutes on RTX 5080")
    print()
    
    response = input("Type 'REBUILD' to confirm: ")
    if response.strip().upper() != "REBUILD":
        print("\n❌ Cancelled.")
        sys.exit(0)
    print("\n✅ Starting...\n")


def load_model():
    """Load BGE-M3 model"""
    print("=" * 80)
    print("LOADING BGE-M3 MODEL")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device.upper()}")
    
    start = time.time()
    model = SentenceTransformer(EMBED_MODEL, device=device)
    print(f"✅ Model loaded in {time.time() - start:.1f}s\n")
    
    return model, device


def simple_chunk_book(filepath, num_chunks=4, max_pages=100):
    """
    Simple chunking:
    1. Read first 100 pages
    2. Split into 4 equal chunks
    """
    try:
        # Read first 100 pages
        max_chars = max_pages * CHARS_PER_PAGE
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(max_chars)
        
        if len(content) < 100:  # Skip tiny files
            return []
        
        # Get book name
        filename = os.path.basename(filepath)
        book_title = filename.replace('_djvu.txt', '').replace('.txt', '').replace('_', ' ')
        book_id = filename.replace('.txt', '')
        
        # Split into equal chunks
        chunk_size = len(content) // num_chunks
        chunks = []
        
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < num_chunks - 1 else len(content)
            
            chunk_text = content[start_idx:end_idx].strip()
            
            if chunk_text:
                chunks.append({
                    'text': chunk_text,
                    'book_id': book_id,
                    'book_title': book_title,
                    'chapter_num': i + 1,  # Chunk number
                    'source_file': filename
                })
        
        return chunks
    
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return []


def init_backends():
    """Initialize all backends (drop and recreate)"""
    print("=" * 80)
    print("INITIALIZING BACKENDS")
    print("=" * 80)
    
    # PostgreSQL
    print("\n📊 PostgreSQL...")
    conn = psycopg2.connect(host=PG_HOST, port=PG_PORT, database=PG_DB, user=PG_USER, password=PG_PASS)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS book_chunks CASCADE")
    cur.execute("""
        CREATE TABLE book_chunks (
            id SERIAL PRIMARY KEY,
            text TEXT NOT NULL,
            embedding vector(1024),
            book_id TEXT,
            book_title TEXT,
            chapter_num INTEGER,
            source_file TEXT
        )
    """)
    conn.commit()
    cur.close()
    print("✅ PostgreSQL ready")
    
    # Qdrant
    print("\n🔷 Qdrant...")
    qdrant = QdrantClient(path=QDRANT_PATH)
    try:
        qdrant.delete_collection(QDRANT_COLLECTION)
    except:
        pass
    qdrant.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE)
    )
    print("✅ Qdrant ready")
    
    # FAISS
    print("\n🔶 FAISS...")
    if os.path.exists(f"{FAISS_INDEX_PATH}.faiss"):
        os.remove(f"{FAISS_INDEX_PATH}.faiss")
        os.remove(f"{FAISS_INDEX_PATH}.meta")
    faiss_index = faiss.IndexHNSWFlat(EMBED_DIM, 32)
    faiss_index.hnsw.efConstruction = 40
    print("✅ FAISS ready\n")
    
    return conn, qdrant, faiss_index


def process_and_upload_book(filepath, model, conn, qdrant, faiss_index, all_texts, all_metadata):
    """
    Process one book:
    1. Chunk into 4 pieces
    2. Batch embed
    3. Upload to all backends
    4. Print done
    """
    filename = os.path.basename(filepath)
    book_name = filename.replace('_djvu.txt', '').replace('.txt', '')
    
    # Step 1: Chunk
    chunks = simple_chunk_book(filepath, NUM_CHUNKS_PER_BOOK, MAX_PAGES)
    if not chunks:
        return
    
    # Step 2: Batch embed all chunks at once
    texts = [c['text'] for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    
    # Step 3: Upload to PostgreSQL
    cur = conn.cursor()
    rows = [(c['text'], emb.tolist(), c['book_id'], c['book_title'], c['chapter_num'], c['source_file'])
            for c, emb in zip(chunks, embeddings)]
    execute_values(cur, """
        INSERT INTO book_chunks (text, embedding, book_id, book_title, chapter_num, source_file)
        VALUES %s
    """, rows, page_size=100)
    conn.commit()
    cur.close()
    
    # Step 4: Upload to Qdrant
    points = []
    for c, emb in zip(chunks, embeddings):
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=emb.tolist(),
            payload={
                'text': c['text'],
                'book_id': c['book_id'],
                'book_title': c['book_title'],
                'chapter_num': c['chapter_num'],
                'source_file': c['source_file']
            }
        ))
    qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points)
    
    # Step 5: Add to FAISS (will build index at end)
    faiss_index.add(embeddings.astype('float32'))
    all_texts.extend(texts)
    for c in chunks:
        all_metadata.append({
            'book_id': c['book_id'],
            'book_title': c['book_title'],
            'chapter_num': c['chapter_num'],
            'source_file': c['source_file']
        })
    
    # Done!
    print(f"   ✅ {book_name}: done ({len(chunks)} chunks)")


def finalize_backends(conn, qdrant, faiss_index, all_texts, all_metadata):
    """Create indexes and save FAISS"""
    print("\n" + "=" * 80)
    print("FINALIZING")
    print("=" * 80)
    
    # PostgreSQL index
    print("\n📊 Creating PostgreSQL HNSW index...")
    cur = conn.cursor()
    start = time.time()
    cur.execute("""
        CREATE INDEX IF NOT EXISTS book_chunks_embedding_idx 
        ON book_chunks 
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)
    conn.commit()
    print(f"✅ Index created in {time.time() - start:.1f}s")
    
    cur.execute("SELECT COUNT(*) FROM book_chunks")
    count = cur.fetchone()[0]
    print(f"✅ PostgreSQL: {count} chunks")
    cur.close()
    conn.close()
    
    # Qdrant verify
    info = qdrant.get_collection(QDRANT_COLLECTION)
    print(f"✅ Qdrant: {info.points_count} vectors")
    
    # FAISS save
    print(f"\n🔶 Saving FAISS index...")
    faiss.write_index(faiss_index, f"{FAISS_INDEX_PATH}.faiss")
    
    meta = {
        'texts': all_texts,
        'metadata': all_metadata  # API expects this format
    }
    with open(f"{FAISS_INDEX_PATH}.meta", 'wb') as f:
        pickle.dump(meta, f)
    
    print(f"✅ FAISS: {faiss_index.ntotal} vectors")


def main():
    print("\n" + "=" * 80)
    print("FAST REBUILD - 20 CHUNKS PER BOOK")
    print("=" * 80)
    
    # Confirm
    confirm_rebuild()
    
    # Load model
    model, device = load_model()
    
    # Init backends
    conn, qdrant, faiss_index = init_backends()
    
    # Get all books
    files = [f for f in os.listdir(BOOKS_DIR) if f.endswith('.txt')]
    files.sort()
    print(f"📚 Found {len(files)} books\n")
    
    print("=" * 80)
    print("PROCESSING BOOKS")
    print("=" * 80)
    print()
    
    # Track for FAISS
    all_texts = []
    all_metadata = []
    
    # Process each book
    start_time = time.time()
    for i, filename in enumerate(files, 1):
        filepath = os.path.join(BOOKS_DIR, filename)
        process_and_upload_book(filepath, model, conn, qdrant, faiss_index, all_texts, all_metadata)
        
        # Progress every 50 books
        if i % 50 == 0:
            elapsed = time.time() - start_time
            books_per_min = i / (elapsed / 60)
            print(f"\n   📊 Progress: {i}/{len(files)} books | {books_per_min:.1f} books/min\n")
    
    elapsed = time.time() - start_time
    print(f"\n✅ All books processed in {elapsed/60:.1f} minutes")
    print(f"   Speed: {len(files) / (elapsed/60):.1f} books/min")
    
    # Finalize
    finalize_backends(conn, qdrant, faiss_index, all_texts, all_metadata)
    
    # Done
    print("\n" + "=" * 80)
    print("✅ REBUILD COMPLETE!")
    print("=" * 80)
    print(f"\nBooks: {len(files)}")
    print(f"Total chunks: {len(all_texts)} (~{len(all_texts)/len(files):.1f} per book)")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print("\nAll three backends now have IDENTICAL data!")
    print("\nRestart your APIs and re-run the Bangla test!")


if __name__ == "__main__":
    main()
