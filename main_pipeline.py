from src.loaders import load_documents
from src.splitters import split_text

def load_and_split_documents(data_path):
    """Unified Preprocessing Pipeline - Task 11"""
    print("=" * 60)
    print("STEP 1: Loading Documents (Part 1)")
    print("=" * 60)
    docs = load_documents(data_path)
    print(f"\nTotal documents loaded: {len(docs)}")
    
    print("\n" + "=" * 60)
    print("STEP 2: Splitting Documents (Part 2)")
    print("=" * 60)
    char_chunks, rec_chunks = split_text(docs)
    print(f"\nCharacter-based chunks: {len(char_chunks)}")
    print(f"Recursive-based chunks: {len(rec_chunks)}")
    
    print("\n" + "=" * 60)
    print("STEP 3: Sample Chunks")
    print("=" * 60)
    if char_chunks:
        print(f"\nCharacter Text Splitter - Chunk 1:\n{char_chunks[0].page_content[:200]}...")
    if rec_chunks:
        print(f"\nRecursive Character Text Splitter - Chunk 1:\n{rec_chunks[0].page_content[:200]}...")
    
    return char_chunks, rec_chunks

if __name__ == "__main__":
    load_and_split_documents("data")
