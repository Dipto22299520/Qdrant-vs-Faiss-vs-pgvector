import os
import json
import random
from pathlib import Path

# Configuration
BOOKS_DIR = r"C:\Users\Administrator\Downloads\wget installation (1)\wget installation\gutenberg_auto\books_utf8"
OUTPUT_FILE = r"C:\Users\Administrator\Downloads\wget installation (1)\wget installation\bangla_test_set_500.json"
NUM_QUERIES = 500
MIN_SENTENCES = 2
MAX_SENTENCES = 3
MAX_PAGES = 100
CHARS_PER_PAGE = 3000  # Approximate characters per page

def is_bangla_book(filename):
    """Check if file is a Bangla book"""
    return filename.endswith('_djvu.txt')

def extract_sentences(text, delimiter='|'):
    """Split text by delimiter and clean up"""
    sentences = text.split(delimiter)
    # Clean and filter sentences
    cleaned = []
    for sent in sentences:
        sent = sent.strip()
        # Keep sentences with at least 20 characters and some Bangla characters
        if len(sent) >= 20 and any('\u0980' <= c <= '\u09FF' for c in sent):
            cleaned.append(sent)
    return cleaned

def read_first_n_pages(filepath, max_pages=100, chars_per_page=3000):
    """Read first N pages of a book"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read(max_pages * chars_per_page)
        return content
    except Exception as e:
        print(f"   ⚠️ Error reading {filepath}: {e}")
        return None

def generate_query(sentences, min_sent=2, max_sent=3):
    """Generate a query from random consecutive sentences"""
    if len(sentences) < min_sent:
        return None
    
    # Random number of sentences
    num_sentences = random.randint(min_sent, max_sent)
    
    # Make sure we don't exceed available sentences
    num_sentences = min(num_sentences, len(sentences))
    
    # Random starting position
    max_start = len(sentences) - num_sentences
    if max_start < 0:
        return None
    
    start_idx = random.randint(0, max_start)
    
    # Combine sentences
    query_sentences = sentences[start_idx:start_idx + num_sentences]
    query_text = ' '.join(query_sentences)
    
    return query_text

def main():
    print("🔍 Bangla Test Set Generator")
    print("=" * 80)
    
    # Find all Bangla books
    bangla_books = []
    for filename in os.listdir(BOOKS_DIR):
        if is_bangla_book(filename):
            filepath = os.path.join(BOOKS_DIR, filename)
            if os.path.isfile(filepath):
                bangla_books.append((filename, filepath))
    
    print(f"📚 Found {len(bangla_books)} Bangla books")
    
    if not bangla_books:
        print("❌ No Bangla books found!")
        return
    
    # Extract sentences from all books
    print(f"\n📖 Processing books (first {MAX_PAGES} pages each)...")
    book_sentences = {}
    
    for filename, filepath in bangla_books:
        print(f"   Processing: {filename}")
        content = read_first_n_pages(filepath, MAX_PAGES, CHARS_PER_PAGE)
        if content:
            sentences = extract_sentences(content, '|')
            if sentences:
                book_sentences[filename] = sentences
                print(f"      ✅ Extracted {len(sentences)} sentences")
            else:
                print(f"      ⚠️ No valid sentences found")
    
    if not book_sentences:
        print("\n❌ No sentences extracted from any book!")
        return
    
    print(f"\n✅ Successfully processed {len(book_sentences)} books")
    total_sentences = sum(len(sents) for sents in book_sentences.values())
    print(f"📝 Total sentences available: {total_sentences}")
    
    # Generate test queries
    print(f"\n🎲 Generating {NUM_QUERIES} test queries...")
    test_set = []
    books_list = list(book_sentences.keys())
    
    generated = 0
    attempts = 0
    max_attempts = NUM_QUERIES * 10  # Prevent infinite loop
    
    while generated < NUM_QUERIES and attempts < max_attempts:
        attempts += 1
        
        # Randomly select a book
        book = random.choice(books_list)
        sentences = book_sentences[book]
        
        # Generate query
        query_text = generate_query(sentences, MIN_SENTENCES, MAX_SENTENCES)
        
        if query_text and len(query_text) >= 50:  # Minimum query length
            # Remove .txt extension for cleaner book name
            book_name = book.replace('_djvu.txt', '').replace('.txt', '')
            
            test_set.append({
                "text": query_text,
                "book": book_name,
                "source_file": book
            })
            
            generated += 1
            
            if generated % 50 == 0:
                print(f"   Generated {generated}/{NUM_QUERIES} queries...")
    
    print(f"\n✅ Successfully generated {len(test_set)} queries")
    
    # Save to JSON
    print(f"\n💾 Saving to: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(test_set, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Test set saved successfully!")
    
    # Statistics
    print("\n" + "=" * 80)
    print("📊 STATISTICS")
    print("=" * 80)
    print(f"Total queries: {len(test_set)}")
    
    # Count queries per book
    book_counts = {}
    for item in test_set:
        book = item['book']
        book_counts[book] = book_counts.get(book, 0) + 1
    
    print(f"\nQueries per book:")
    for book, count in sorted(book_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   {book}: {count} queries")
    
    # Average query length
    avg_length = sum(len(item['text']) for item in test_set) / len(test_set)
    print(f"\nAverage query length: {avg_length:.1f} characters")
    
    print("\n🎉 Done!")

if __name__ == "__main__":
    main()
