from src.loaders import Document
from typing import List

def split_text(documents: List[Document]):
    """Split documents using Character-based and Recursive Character-based splitting."""
    
    # Task 7 & 8: Character Text Splitter (Length-based)
    def character_split(documents, chunk_size=100, chunk_overlap=20):
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
        # Separators in order of preference (paragraphs, sentences, words, characters)
        separators = ["\n\n", "\n", ". ", " ", ""]
        chunks = []
        for doc in documents:
            text = doc.page_content
            # First split by paragraphs
            parts = _split_recursive(text, separators, chunk_size)
            for part in parts:
                chunks.append(Document(
                    page_content=part,
                    metadata={**doc.metadata, "splitter": "RecursiveCharacterTextSplitter"}
                ))
        return chunks
    
    def _split_recursive(text, separators, chunk_size):
        if not separators:
            # Base case: split by characters
            return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        if separator == "":
            return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        splits = text.split(separator)
        chunks = []
        current = ""
        
        for split in splits:
            if len(current) + len(split) + len(separator) <= chunk_size:
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
            if len(chunk) > chunk_size and remaining_separators:
                result.extend(_split_recursive(chunk, remaining_separators, chunk_size))
            else:
                result.append(chunk)
        
        return result
    
    char_chunks = character_split(documents)
    rec_chunks = recursive_split(documents)
    
    return char_chunks, rec_chunks
