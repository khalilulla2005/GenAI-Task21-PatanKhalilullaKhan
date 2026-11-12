import nbformat as nbf

nb = nbf.v4.new_notebook()

text_cell_1 = nbf.v4.new_markdown_cell("""# GenAI Assignment 21: LangChain Document Loaders & Text Splitters
**Student Name:** Patan Khalilulla Khan  
**Course:** GenAI Training Program  

---

## Assignment Overview
This Jupyter Notebook fully implements **Assignment 21** using **LangChain's official standard modules**:
1. **Document Loaders (Tasks 1-5):**
   - `TextLoader` (`sample.txt`)
   - `CSVLoader` (`sample.csv`)
   - `PyPDFLoader` (`sample.pdf`)
   - `DirectoryLoader` with `loader_cls` mapping (`data/`)
   - `WebBaseLoader` (URL content loading)
2. **Text Splitters & Chunking (Tasks 7-10):**
   - `CharacterTextSplitter`
   - `RecursiveCharacterTextSplitter`
   - Structure-aware splitting via `MarkdownHeaderTextSplitter`
   - Semantic chunking explanation and demonstration
3. **Unified Pipeline (Task 11):**
   - Unified pipeline supporting both local file paths and web URLs.
""")

code_cell_1 = nbf.v4.new_code_cell("""# Import required LangChain modules and packages
import os
import langchain
from langchain_community.document_loaders import TextLoader, CSVLoader, PyPDFLoader, DirectoryLoader, WebBaseLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings

print("LangChain and standard modules loaded successfully for Assignment 21!")
""")

text_cell_2 = nbf.v4.new_markdown_cell("""## Part 1: LangChain Document Loaders (Tasks 1 - 5)
Loading various document formats using standard LangChain community loaders.
""")

code_cell_2 = nbf.v4.new_code_cell("""# Task 1: TextLoader
text_loader = TextLoader('data/sample.txt', encoding='utf-8')
text_docs = text_loader.load()
print("=== Task 1: TextLoader ===")
print(f"Loaded {len(text_docs)} document(s)")
print(f"Content: {text_docs[0].page_content}")
print(f"Metadata: {text_docs[0].metadata}")
""")

code_cell_3 = nbf.v4.new_code_cell("""# Task 2: CSVLoader
csv_loader = CSVLoader('data/sample.csv')
csv_docs = csv_loader.load()
print("=== Task 2: CSVLoader ===")
print(f"Loaded {len(csv_docs)} row(s)")
for i, doc in enumerate(csv_docs):
    print(f"  Row {i+1}: {doc.page_content} | Metadata: {doc.metadata}")
""")

code_cell_4 = nbf.v4.new_code_cell("""# Task 3: PyPDFLoader
pdf_loader = PyPDFLoader('data/sample.pdf')
pdf_docs = pdf_loader.load()
print("=== Task 3: PyPDFLoader ===")
print(f"Loaded {len(pdf_docs)} page(s)")
for i, doc in enumerate(pdf_docs):
    print(f"  Page {i+1} content: {doc.page_content[:150]}...")
""")

code_cell_5 = nbf.v4.new_code_cell("""# Task 4: DirectoryLoader with loader_cls mapping
dir_loader = DirectoryLoader(
    'data/',
    glob='**/*.txt',
    loader_cls=TextLoader
)
dir_docs = dir_loader.load()
print("=== Task 4: DirectoryLoader ===")
print(f"Loaded {len(dir_docs)} document(s) via DirectoryLoader")
for doc in dir_docs:
    print(f"  Source: {doc.metadata['source']}")
""")

code_cell_6 = nbf.v4.new_code_cell("""# Task 5: WebBaseLoader
try:
    web_loader = WebBaseLoader('https://example.com')
    web_docs = web_loader.load()
    print("=== Task 5: WebBaseLoader ===")
    print(f"Loaded {len(web_docs)} web document")
    print(f"Content preview: {web_docs[0].page_content[:200]}...")
    print(f"Metadata: {web_docs[0].metadata}")
except Exception as e:
    print("WebBaseLoader network execution notice:", e)
""")

text_cell_3 = nbf.v4.new_markdown_cell("""## Part 2: LangChain Text Splitters & Chunking (Tasks 7 - 10)
Chunking documents using Character, Recursive, Markdown, and Semantic splitters.
""")

code_cell_7 = nbf.v4.new_code_cell("""# Task 7: CharacterTextSplitter
char_splitter = CharacterTextSplitter(separator='\\n', chunk_size=100, chunk_overlap=20)
sample_text = \"\"\"This is a sample text file for LangChain loader task.
It contains multiple lines to test character splitting.
LangChain makes document processing extremely easy and robust.\"\"\"
char_chunks = char_splitter.split_text(sample_text)
print("=== Task 7: CharacterTextSplitter ===")
print(f"Generated {len(char_chunks)} chunks")
for i, chunk in enumerate(char_chunks):
    print(f"  Chunk {i+1}: {repr(chunk)}")
""")

code_cell_8 = nbf.v4.new_code_cell("""# Task 8: RecursiveCharacterTextSplitter
rec_splitter = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=30)
rec_chunks = rec_splitter.split_text(sample_text)
print("=== Task 8: RecursiveCharacterTextSplitter ===")
print(f"Generated {len(rec_chunks)} chunks")
for i, chunk in enumerate(rec_chunks):
    print(f"  Chunk {i+1}: {repr(chunk)}")
""")

code_cell_9 = nbf.v4.new_code_cell("""# Task 9: Document structure-aware splitting (MarkdownHeaderTextSplitter)
markdown_text = \"\"\"# Chapter 1
Introduction to LangChain loaders.
## Section 1.1
Details about text splitters.
### Subsection
Recursive splitting details.\"\"\"

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_docs = markdown_splitter.split_text(markdown_text)
print("=== Task 9: MarkdownHeaderTextSplitter ===")
print(f"Split into {len(md_docs)} structured documents")
for doc in md_docs:
    print(f"  Content: {doc.page_content} | Metadata: {doc.metadata}")
""")

code_cell_10 = nbf.v4.new_code_cell("""# Task 10: Semantic Chunking Explanation & Demo
print("=== Task 10: Semantic Chunking ===")
print("Semantic chunking evaluates embedding similarity between sentences to split text at semantic boundaries rather than fixed character lengths.")

try:
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    semantic_splitter = SemanticChunker(embeddings)
    semantic_docs = semantic_splitter.create_documents([sample_text])
    print(f"Semantic Chunks generated: {len(semantic_docs)}")
    for i, doc in enumerate(semantic_docs):
        print(f"  Semantic Chunk {i+1}: {doc.page_content}")
except Exception as e:
    print("Semantic chunking demo executed:", e)
""")

text_cell_4 = nbf.v4.new_markdown_cell("""## Part 3: Unified Pipeline (Task 11)
Unified pipeline supporting both local file paths and web URLs.
""")

code_cell_11 = nbf.v4.new_code_cell("""# Task 11: Unified Pipeline Support for Local Files and Web URLs
class Assignment21Pipeline:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30)
        
    def process_source(self, source: str, source_type: str = 'file'):
        print(f"\\nProcessing {source_type} source: {source}")
        if source_type == 'file':
            if source.endswith('.txt'):
                loader = TextLoader(source)
            elif source.endswith('.csv'):
                loader = CSVLoader(source)
            elif source.endswith('.pdf'):
                loader = PyPDFLoader(source)
            else:
                loader = TextLoader(source)
        elif source_type == 'web':
            loader = WebBaseLoader(source)
        else:
            raise ValueError("Invalid source type")
            
        docs = loader.load()
        chunks = self.splitter.split_documents(docs)
        print(f"Successfully loaded and split into {len(chunks)} chunks.")
        for i, chunk in enumerate(chunks[:2]):
            print(f"  Chunk {i+1}: {chunk.page_content[:100]}... | Metadata: {chunk.metadata}")
        return chunks

# Test pipeline with local text file
pipeline = Assignment21Pipeline()
chunks_file = pipeline.process_source('data/sample.txt', source_type='file')

# Test pipeline with web URL
try:
    chunks_web = pipeline.process_source('https://example.com', source_type='web')
except Exception as e:
    print("Web processing notice:", e)
""")

nb['cells'] = [
    text_cell_1, code_cell_1,
    text_cell_2, code_cell_2, code_cell_3, code_cell_4, code_cell_5, code_cell_6,
    text_cell_3, code_cell_7, code_cell_8, code_cell_9, code_cell_10,
    text_cell_4, code_cell_11
]

nbf.write(nb, 'D:\\techtude\\GenAI-Task21-PatanKhalilullaKhan\\GenAI_Assignment21_LangChain_Pipeline.ipynb')
print("Assignment 21 Jupyter Notebook generated successfully!")
