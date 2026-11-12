"""
Main Pipeline Module - Part 3 of Assignment 21

Unified preprocessing pipeline that combines document loading and text splitting
into a single workflow. This is the integration point for Tasks 11-12.

Author: Patan Khalilulla Khan
"""

from src.loaders import load_documents
from src.splitters import split_text


def load_and_split_documents(data_path: str):
    """
    Unified Preprocessing Pipeline - Task 11
    
    Combines all document loaders and text splitters into one workflow.
    This is the main integration function for the assignment.
    
    Pipeline Steps:
    1. Load documents from multiple sources (Text, CSV, PDF, Directory)
    2. Split documents using Character Text Splitter
    3. Split documents using Recursive Character Text Splitter
    4. Return both chunk sets for comparison
    
    Args:
        data_path: Path to the data directory containing source files
        
    Returns:
        Tuple of (char_chunks, rec_chunks)
        
    Example:
        >>> char_chunks, rec_chunks = load_and_split_documents("data")
        >>> print(f"Total chunks: {len(char_chunks) + len(rec_chunks)}")
    """
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
