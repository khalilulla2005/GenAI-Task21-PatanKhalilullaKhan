"""
Text Splitters Module - Part 2 of Assignment 21

This module implements text splitting functionality:
- CharacterTextSplitter (length-based splitting)
- RecursiveCharacterTextSplitter (structure-based splitting)

These splitters break large documents into smaller chunks suitable for
LLM processing, with overlap to preserve context across boundaries.

Author: Patan Khalilulla Khan
"""

from src.loaders import Document
from typing import List


def split_text(documents: List[Document]):
    """
    Split documents using both Character-based and Recursive Character-based splitting.
    
    Implements Tasks 7-9:
    - Task 7: Length-Based Text Splitter (CharacterTextSplitter)
    - Task 8: Character Text Splitter behavior demonstration
    - Task 9: Text Structure-Based Splitter (RecursiveCharacterTextSplitter)
    
    Args:
        documents: List of Document objects to split
        
    Returns:
        Tuple of (char_chunks, rec_chunks) where:
        - char_chunks: List of chunks from CharacterTextSplitter
        - rec_chunks: List of chunks from RecursiveCharacterTextSplitter
        
    Example:
        >>> char_chunks, rec_chunks = split_text(documents)
        >>> print(f"Character: {len(char_chunks)}, Recursive: {len(rec_chunks)}")
    """
    
    # Task 7 & 8: Character Text Splitter (Length-based)
    def character_split(documents, chunk_size=100, chunk_overlap=20):
        """
        Split text into fixed-size chunks with overlap.
        
        Splits text strictly by character count. May break mid-sentence
        but is fast and predictable. Best for simple unstructured text.
        
        Args:
            documents: List of Document objects
            chunk_size: Maximum characters per chunk
            chunk_overlap: Characters to overlap between consecutive chunks
            
        Returns:
            List of Document chunks
        """
        chunks = []
        for doc in documents:
            text = doc.page_content
            start = 0
            while start < len(text):
                end = start + chunk_size
                chunk_text = text[start:end]
                chunks.append(Document(
                    page_content=chunk_text,
                    metadata={**doc.metadata, "splitter": "CharacterTextSplitter"}
                ))
                start = end - chunk_overlap
        return chunks
    
    # Task 9: Recursive Character Text Splitter (Structure-based)
    def recursive_split(documents, chunk_size=100, chunk_overlap=20):
        """
        Split text using hierarchical separators to preserve structure.
        
        Tries to split at paragraph boundaries first, then falls back to
        sentences, words, and finally characters. Produces chunks that
        better respect natural document structure.
        
        Separator hierarchy:
        1. "\n\n" - Paragraph breaks (preferred)
        2. "\n" - Line breaks
        3. ". " - Sentence boundaries
        4. " " - Word boundaries
        5. "" - Character boundaries (fallback)
        
        Args:
            documents: List of Document objects
            chunk_size: Maximum characters per chunk
            chunk_overlap: Characters to overlap between consecutive chunks
            
        Returns:
            List of Document chunks
        """
        chunks = []
        for doc in documents:
            text = doc.page_content
            parts = _split_recursive(text, chunk_size)
            for part in parts:
                chunks.append(Document(
                    page_content=part,
                    metadata={**doc.metadata, "splitter": "RecursiveCharacterTextSplitter"}
                ))
        return chunks
    
    def _split_recursive(text, chunk_size):
        """
        Recursively split text using hierarchical separators.
        
        Args:
            text: Text string to split
            chunk_size: Maximum characters per chunk
            
        Returns:
            List of text chunks
        """
        separators = ["\n\n", "\n", ". ", " ", ""]
        
        def _split_with_separator(text, separator, remaining_seps, chunk_size):
            if separator == "":
                # Base case: split by characters
                return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            
            splits = text.split(separator)
            chunks = []
            current = ""
            
            for split in splits:
                separator_len = len(separator) if current else 0
                if len(current) + len(split) + separator_len <= chunk_size:
                    if current:
                        current += separator + split
                    else:
                        current = split
                else:
                    if current:
                        chunks.append(current)
                    current = split
            
            if current:
                chunks.append(current)
            
            # If any chunk is still too large, recurse with remaining separators
            result = []
            for chunk in chunks:
                if len(chunk) > chunk_size and remaining_seps:
                    result.extend(_split_with_separator(
                        chunk, remaining_seps[0], remaining_seps[1:], chunk_size
                    ))
                else:
                    result.append(chunk)
            
            return result
        
        return _split_with_separator(text, separators[0], separators[1:], chunk_size)
    
    char_chunks = character_split(documents)
    rec_chunks = recursive_split(documents)
    
    return char_chunks, rec_chunks
