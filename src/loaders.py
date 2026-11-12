import os
import glob
import pandas as pd
from typing import List

class Document:
    """Simple Document class to mimic LangChain Document"""
    def __init__(self, page_content: str, metadata: dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}

def _read_file_with_encoding(filepath: str) -> str:
    """Try multiple encodings to read a file."""
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    for enc in encodings:
        try:
            with open(filepath, "r", encoding=enc) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    # Last resort: read with errors='ignore'
    with open(filepath, "r", encoding='utf-8', errors='ignore') as f:
        return f.read()

def load_documents(data_path: str) -> List[Document]:
    """Load documents from various sources using Python standard library."""
    docs = []
    
    # Task 1: Text Loader - load .txt file
    try:
        content = _read_file_with_encoding(f"{data_path}/sample.txt")
        docs.append(Document(page_content=content, metadata={"source": "sample.txt", "loader": "TextLoader"}))
        print("Text Loader: Loaded sample.txt successfully")
    except Exception as e:
        print(f"Text Loader Error: {e}")
    
    # Task 2: CSV Loader - load .csv file
    try:
        for enc in ['utf-8', 'latin-1', 'cp1252', 'utf-16']:
            try:
                df = pd.read_csv(f"{data_path}/sample.csv", encoding=enc)
                content = df.to_string(index=False)
                docs.append(Document(page_content=content, metadata={"source": "sample.csv", "loader": "CSVLoader"}))
                print("CSV Loader: Loaded sample.csv successfully")
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
    except Exception as e:
        print(f"CSV Loader Error: {e}")
    
    # Task 3: PDF Loader - load .pdf file
    try:
        import PyPDF2
        with open(f"{data_path}/sample.pdf", "rb") as f:
            reader = PyPDF2.PdfReader(f)
            content = ""
            for page in reader.pages:
                content += page.extract_text() + "\n"
            docs.append(Document(page_content=content, metadata={"source": "sample.pdf", "loader": "PyPDFLoader"}))
            print("PDF Loader: Loaded sample.pdf successfully")
    except Exception as e:
        print(f"PDF Loader Error: {e}")
    
    # Task 4: Directory Loader - load all .txt files inside data folder
    try:
        txt_files = glob.glob(os.path.join(data_path, "**", "*.txt"), recursive=True)
        for txt_file in txt_files:
            content = _read_file_with_encoding(txt_file)
            docs.append(Document(page_content=content, metadata={"source": os.path.basename(txt_file), "loader": "DirectoryLoader"}))
        print(f"Directory Loader: Loaded {len(txt_files)} .txt file(s) successfully")
    except Exception as e:
        print(f"Directory Loader Error: {e}")
        
    return docs
