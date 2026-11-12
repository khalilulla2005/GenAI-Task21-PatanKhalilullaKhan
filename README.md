# Assignment 21: LangChain Document Loaders & Text Splitters

**Student Name:** Patan Khalilulla Khan

## Project Structure
```
GenAI-Task21-PatanKhalilullaKhan/
├── data/
│   ├── sample.txt          # Text file for TextLoader
│   ├── sample.csv          # CSV file for CSVLoader
│   ├── sample.pdf          # PDF file for PyPDFLoader
├── src/
│   ├── __init__.py
│   ├── loaders.py          # Part 1: Document Loaders (Tasks 1-5)
│   └── splitters.py        # Part 2: Text Splitters (Tasks 7-9)
├── main_pipeline.py        # Part 3: Unified Pipeline (Task 11)
├── README.md
└── requirements.txt
```

## Part 1 — Document Loaders in LangChain

### Task 1: Text Loader
- Loads `.txt` files using `TextLoader`
- File: `data/sample.txt`

### Task 2: CSV Loader
- Loads `.csv` files using `CSVLoader`
- File: `data/sample.csv`

### Task 3: PDF Loader
- Loads `.pdf` files using `PyPDFLoader`
- File: `data/sample.pdf`

### Task 4: Directory Loader
- Loads all `.txt` files inside `data/` folder using `DirectoryLoader`

### Task 5: Web Loader (Conceptual)
- `WebBaseLoader` is used to load content from any public webpage
- Implementation ready for URL input

## Part 2 — Text Splitters in LangChain

### Task 6: Why Text Splitting is Required (Conceptual)
- Large documents cannot be directly passed to LLMs due to context window limits
- Chunking breaks documents into smaller, meaningful pieces
- Overlap between chunks preserves context across boundaries

### Task 7: Length-Based Text Splitter
- Uses `CharacterTextSplitter`
- Configurable `chunk_size` and `chunk_overlap`

### Task 8: Character Text Splitter
- Splits based on character count
- Preserves text flow within chunk size limits

### Task 9: Text Structure-Based Splitter
- Uses `RecursiveCharacterTextSplitter`
- Respects document structure (paragraphs, sentences)
- Tries multiple separators to find optimal split points

## Part 3 — Mini Integration Task

### Task 11: Unified Preprocessing Pipeline
- `load_and_split_documents()` function in `main_pipeline.py`
- Combines all loaders and splitters into one workflow
- Output: Chunked documents ready for embedding/storage

### Task 12: Observations & Insights
- **Character Text Splitter:** Simple, fast, but may break mid-sentence
- **Recursive Character Text Splitter:** Better context preservation, respects natural boundaries
- **Chunk overlap** is critical for maintaining semantic continuity between chunks

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run pipeline: `python main_pipeline.py`
3. Ensure `data/` folder contains `sample.txt`, `sample.csv`, and `sample.pdf`
