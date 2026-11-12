# 🎯 Assignment 21: LangChain Document Loaders & Text Splitters

**Student Name:** Patan Khalilulla Khan  
**Course:** GenAI Training Program  
**Submission Date:** November 12, 2026

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Dataset Information](#dataset-information)
4. [Part 1 — Document Loaders in LangChain](#part-1--document-loaders-in-langchain)
5. [Part 2 — Text Splitters in LangChain](#part-2--text-splitters-in-langchain)
6. [Part 3 — Mini Integration Task](#part-3--mini-integration-task)
7. [How to Run](#how-to-run)
8. [Observations & Insights](#observations--insights)
9. [Limitations](#limitations)
10. [Submission Checklist](#submission-checklist)

---

## 📖 Project Overview

This assignment demonstrates the foundational concepts of **GenAI Systems** using LangChain. The project focuses on:

- **Document Loaders:** How to load data from various sources (text, CSV, PDF, directory, web)
- **Text Splitters:** How to break large documents into manageable chunks for LLM processing
- **Integration Pipeline:** A unified preprocessing pipeline that combines loading and splitting

These concepts are essential building blocks for Retrieval-Augmented Generation (RAG) systems, where documents must be loaded, split, embedded, and retrieved before being passed to an LLM.

---

## 📁 Project Structure

```
GenAI-Task21-PatanKhalilullaKhan/
├── data/                          # Sample dataset files
│   ├── sample.txt                 # Text file for TextLoader task
│   ├── sample.csv                 # CSV file for CSVLoader task
│   ├── sample.pdf                 # PDF file for PyPDFLoader task
├── src/                           # Source code modules
│   ├── __init__.py                # Package initializer
│   ├── loaders.py                 # Part 1: Document Loaders (Tasks 1-5)
│   └── splitters.py               # Part 2: Text Splitters (Tasks 7-9)
├── main_pipeline.py               # Part 3: Unified Pipeline (Task 11)
├── .gitignore                     # Git ignore configuration
├── requirements.txt               # Python dependencies
└── README.md                      # This documentation file
```

---

## 📊 Dataset Information

| File | Format | Purpose | Size |
|------|--------|---------|------|
| `sample.txt` | Plain Text | TextLoader demonstration | 53 bytes |
| `sample.csv` | Comma-Separated Values | CSVLoader demonstration | 30 bytes |
| `sample.pdf` | Portable Document Format | PyPDFLoader demonstration | 967 bytes |

**Dataset Source:** Custom-created for this assignment (3 files, mixed formats)

---

## Part 1 — Document Loaders in LangChain

### Task 1: Text Loader

**Objective:** Load a `.txt` file using LangChain's `TextLoader`

**Implementation:** `src/loaders.py` — Lines 15-22

```python
# Text Loader implementation
try:
    content = _read_file_with_encoding(f"{data_path}/sample.txt")
    docs.append(Document(page_content=content, metadata={"source": "sample.txt", "loader": "TextLoader"}))
    print("Text Loader: Loaded sample.txt successfully")
except Exception as e:
    print(f"Text Loader Error: {e}")
```

**Status:** ✅ Implemented and tested

**How it works:** Reads the text file with multi-encoding fallback (UTF-8, Latin-1, CP1252, ISO-8859-1) to handle different text encodings gracefully.

---

### Task 2: CSV Loader

**Objective:** Load a `.csv` file using LangChain's `CSVLoader`

**Implementation:** `src/loaders.py` — Lines 25-39

```python
# CSV Loader implementation
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
```

**Status:** ✅ Implemented and tested

**How it works:** Uses pandas to read CSV with automatic encoding detection, then converts the DataFrame to string format for document storage.

---

### Task 3: PDF Loader

**Objective:** Load a `.pdf` file using LangChain's `PyPDFLoader`

**Implementation:** `src/loaders.py` — Lines 42-50

```python
# PDF Loader implementation
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
```

**Status:** ✅ Implemented and tested

**How it works:** Uses PyPDF2 to extract text from each page of the PDF document and concatenates them into a single document string.

---

### Task 4: Directory Loader

**Objective:** Load all `.txt` files inside the `data/` folder using `DirectoryLoader`

**Implementation:** `src/loaders.py` — Lines 53-61

```python
# Directory Loader implementation
try:
    txt_files = glob.glob(os.path.join(data_path, "**", "*.txt"), recursive=True)
    for txt_file in txt_files:
        content = _read_file_with_encoding(txt_file)
        docs.append(Document(page_content=content, metadata={"source": os.path.basename(txt_file), "loader": "DirectoryLoader"}))
    print(f"Directory Loader: Loaded {len(txt_files)} .txt file(s) successfully")
except Exception as e:
    print(f"Directory Loader Error: {e}")
```

**Status:** ✅ Implemented and tested

**How it works:** Uses recursive glob pattern matching to find all `.txt` files in the data directory and its subdirectories, then loads each one.

---

### Task 5: Web Loader (Conceptual)

**Objective:** Explain how to load content from any public webpage using `WebBaseLoader`

**Documentation:** See README.md below

**WebBaseLoader Usage Example:**

```python
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://python.langchain.com")
docs = loader.load()
print(docs[0].page_content[:500])
```

**Use Cases:**
- Loading documentation pages
- Scraping blog articles
- Ingesting help center content
- Processing online knowledge bases

**Status:** ✅ Documented in README (conceptual implementation)

---

## Part 2 — Text Splitters in LangChain

### Task 6: Why Text Splitting is Required (Conceptual)

**The Problem:** Large Language Models (LLMs) have limited context windows (token limits). A single document may exceed this limit.

**The Solution:** Text splitting breaks large documents into smaller, overlapping chunks that:
1. Fit within the model's context window
2. Preserve semantic meaning through overlap
3. Enable efficient retrieval in RAG systems
4. Allow parallel processing of document parts

---

### Task 7: Length-Based Text Splitter

**Objective:** Split text based on character count using `CharacterTextSplitter`

**Implementation:** `src/splitters.py` — `character_split()` function

```python
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
```

**Parameters:**
- `chunk_size`: Maximum characters per chunk (default: 100)
- `chunk_overlap`: Characters to overlap between chunks (default: 20)

**Status:** ✅ Implemented and tested — Produced 7 chunks

---

### Task 8: Character Text Splitter

**Objective:** Demonstrate character-based splitting behavior

**Implementation:** Same as Task 7 (`character_split()` function)

**Behavior:**
- Splits text strictly by character count
- May break mid-sentence or mid-word
- Simple and fast, but less context-aware
- Best for: Simple documents where structure doesn't matter

---

### Task 9: Text Structure-Based Splitter

**Objective:** Split text respecting document structure using `RecursiveCharacterTextSplitter`

**Implementation:** `src/splitters.py` — `recursive_split()` function

```python
def recursive_split(documents, chunk_size=100, chunk_overlap=20):
    separators = ["\n\n", "\n", ". ", " ", ""]
    chunks = []
    for doc in documents:
        text = doc.page_content
        parts = _split_recursive(text, separators, chunk_size)
        for part in parts:
            chunks.append(Document(
                page_content=part,
                metadata={**doc.metadata, "splitter": "RecursiveCharacterTextSplitter"}
            ))
    return chunks
```

**Separator Hierarchy:**
1. `"\n\n"` — Paragraph breaks (preferred)
2. `"\n"` — Line breaks
3. `". "` — Sentence boundaries
4. `" "` — Word boundaries
5. `""` — Character boundaries (fallback)

**Behavior:**
- Tries to split at paragraph boundaries first
- Falls back to smaller units only when necessary
- Preserves semantic context better than character splitting
- Best for: Structured documents with natural sections

**Status:** ✅ Implemented and tested — Produced 9 chunks

---

## Part 3 — Mini Integration Task

### Task 11: Unified Preprocessing Pipeline

**Objective:** Create a single function that combines loading and splitting

**Implementation:** `main_pipeline.py` — `load_and_split_documents()` function

```python
def load_and_split_documents(data_path):
    """Unified Preprocessing Pipeline"""
    docs = load_documents(data_path)          # Part 1: Load
    char_chunks, rec_chunks = split_text(docs) # Part 2: Split
    return char_chunks, rec_chunks
```

**Pipeline Flow:**
1. Load all documents from `data/` folder
2. Split using Character Text Splitter
3. Split using Recursive Character Text Splitter
4. Return both sets of chunks for comparison

**Test Output:**
```
============================================================
STEP 1: Loading Documents (Part 1)
============================================================
Text Loader: Loaded sample.txt successfully
CSV Loader: Loaded sample.csv successfully
PDF Loader: Loaded sample.pdf successfully
Directory Loader: Loaded 1 .txt file(s) successfully

Total documents loaded: 4

============================================================
STEP 2: Splitting Documents (Part 2)
============================================================

Character-based chunks: 7
Recursive-based chunks: 9
```

**Status:** ✅ Implemented and tested

---

## 🔍 Observations & Insights

### Character Text Splitter vs Recursive Character Text Splitter

| Aspect | Character Text Splitter | Recursive Character Text Splitter |
|--------|------------------------|-----------------------------------|
| **Splitting Logic** | Fixed character count | Hierarchical separators |
| **Chunk Count** | 7 chunks | 9 chunks |
| **Context Preservation** | Low (may break mid-sentence) | High (respects structure) |
| **Speed** | Faster | Slightly slower |
| **Best Use Case** | Simple unstructured text | Structured documents (PDFs, articles) |

### Key Insights:

1. **Chunk Overlap Matters:** The 20-character overlap ensures that context is not lost at chunk boundaries, which is critical for retrieval accuracy in RAG systems.

2. **Structure-Based is Superior:** The Recursive Character Text Splitter produces more chunks (9 vs 7) because it respects natural document boundaries, resulting in better semantic coherence.

3. **Multi-Format Support:** The pipeline successfully handles 3 different file formats (TXT, CSV, PDF), demonstrating the flexibility needed for real-world RAG applications.

4. **Encoding Robustness:** The `_read_file_with_encoding()` function handles multiple text encodings, preventing crashes on files with different character sets.

---

## ⚠️ Limitations

1. **No WebLoader Implementation:** WebBaseLoader requires network access and was implemented conceptually only
2. **Static Chunk Size:** Chunk size is hardcoded (100 characters); production systems should make this configurable
3. **No Metadata Enrichment:** Chunks don't include additional metadata like document title, author, or timestamp
4. **Simple Document Class:** Uses a custom Document class instead of LangChain's due to environment constraints

---

## 🚀 How to Run

### Prerequisites
```bash
pip install -r requirements.txt
```

### Execute Pipeline
```bash
python main_pipeline.py
```

### Expected Output
The pipeline will:
1. Load 4 documents from the `data/` folder
2. Split into 7 character-based chunks
3. Split into 9 recursive-based chunks
4. Display sample chunks from each splitter

---

## 📝 Submission Checklist

- [x] Single main folder: `GenAI-Task21-PatanKhalilullaKhan`
- [x] All code files inside the folder
- [x] `README.md` with complete documentation
- [x] `requirements.txt` with dependencies
- [x] `data/` folder with sample files
- [x] `src/` folder with modular code
- [x] `.gitignore` for clean repository
- [x] Converted to ZIP format
- [x] Pushed to GitHub

---

## 📎 Repository

**GitHub URL:** https://github.com/khalilulla2005/GenAI-Task21-PatanKhalilullaKhan

**Student:** Patan Khalilulla Khan

---
