# 🔍 RAG From Scratch

A **production-grade Retrieval-Augmented Generation (RAG) system built entirely from scratch** — no LangChain, no LlamaIndex. Every component is hand-rolled so you understand exactly what's happening under the hood.

Built as a portfolio project for AI Engineers who want to demonstrate deep understanding of the RAG stack.

---

## 🏗️ Architecture

```
                          ┌─────────────────────────────────────────┐
                          │              RAG PIPELINE                │
                          │                                          │
  Documents / URLs        │  ┌──────────┐    ┌──────────┐           │
  ──────────────────────► │  │  Loader  │───►│  Chunker │           │
                          │  └──────────┘    └────┬─────┘           │
                          │                       │                  │
                          │               ┌───────▼──────┐          │
                          │               │   Embedder   │          │
                          │               │ (ST / Groq*) │          │
                          │               └───────┬──────┘          │
                          │                       │                  │
                          │          ┌────────────▼──────────────┐  │
                          │          │       Vector Store         │  │
                          │          │   FAISS  │  ChromaDB       │  │
                          │          └────────────┬──────────────-┘  │
                          │                       │                  │
  User Question           │  ┌─────────────────── ▼──────────────┐  │
  ──────────────────────► │  │         Hybrid Retriever           │  │
                          │  │  Dense + BM25 Sparse + RRF Fusion  │  │
                          │  │  + Cross-Encoder Reranker           │  │
                          │  └───────────────────┬───────────────-┘  │
                          │                      │                   │
                          │  ┌───────────────────▼──────────────┐   │
                          │  │          LLM Generator            │   │
                          │  │    Groq (LLMs) │ Ollama (local)   │   │
                          │  └───────────────────┬──────────────-┘   │
                          │                      │                   │
                          └──────────────────────┼───────────────────┘
                                                 │
                                                 ▼
                                          RAGResponse
                                   (answer + sources + tokens)
```


---

## 📁 Project Structure

```
rag_from_scratch/
│
├── config.py                    # Centralised config (Pydantic Settings)
├── main.py                      # CLI entrypoint
├── requirements.txt
├── .env.example
│
├── src/
│   ├── models.py                # Document, Chunk, RetrievedChunk, RAGResponse
│   ├── pipeline.py              # Orchestrates the full pipeline
│   ├── api.py                   # FastAPI REST server
│   │
│   ├── ingestion/
│   │   ├── loaders.py           # PDF, DOCX, TXT, Markdown, URL loaders
│   │   └── chunker.py           # Recursive, Sentence, Semantic chunkers
│   │
│   ├── embeddings/
│   │   └── embedder.py          # SentenceTransformer
│   │
│   ├── vectorstore/
│   │   ├── faiss_store.py       # FAISS vector store (local, fast)
│   │   └── chroma_store.py      # ChromaDB vector store (persistent)
│   │
│   ├── retrieval/
│   │   └── retriever.py         # Dense + BM25 + Hybrid (RRF) + Reranker
│   │
│   ├── generation/
│   │   └── generator.py         # Groq + Ollama generators, prompt builder
│   │
│   └── evaluation/
│       └── evaluator.py         # Hit Rate, MRR, Context Precision, Faithfulness
│
├── tests/
│   ├── test_chunker.py
│   ├── test_loaders.py
│   └── test_pipeline.py
│
├── scripts/
│   ├── ingest_sample.py         # Ingest a Wikipedia article
│   └── demo_query.py            # Run demo queries
│
├── notebooks/
│   └── rag_walkthrough.ipynb    # Step-by-step interactive notebook
│
└── data/
    ├── raw/                     # Drop your documents here
    ├── processed/
    ├── vectorstore/             # Saved FAISS / Chroma indexes
    └── eval_samples.json        # Sample evaluation dataset
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/rag-from-scratch
cd rag-from-scratch

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
# Or use Ollama for a completely free local setup
```

### 3. Ingest Documents

```bash
# Ingest a URL
python main.py ingest "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"

# Ingest a local file
python main.py ingest ./data/raw/my_document.pdf

# Ingest an entire directory
python main.py ingest ./data/raw/
```

### 4. Ask Questions

```bash
python main.py query "What is retrieval-augmented generation?" --show-sources
```

### 5. Start the API Server

```bash
python main.py serve
# API docs available at: http://localhost:8000/docs
```

---

## 🛠️ Key Components Explained

### Chunking Strategies

| Strategy | How It Works | Best For |
|---|---|---|
| `recursive` | Splits on `\n\n` → `\n` → `.` → ` ` in priority order | General purpose (default) |
| `sentence` | Greedy sentence windows with overlap | Conversational text, FAQs |
| `semantic` | Groups sentences by cosine similarity | Dense technical documents |

```python
from src.ingestion.chunker import get_chunker

chunker = get_chunker("recursive", chunk_size=512, chunk_overlap=64)
chunks = chunker.split(document)
```

### Retrieval Modes

| Mode | Description | Trade-off |
|---|---|---|
| `dense` | Pure vector similarity (cosine) | Best for semantic queries |
| `sparse` | BM25 keyword matching | Best for exact-match queries |
| `hybrid` | RRF fusion of dense + sparse | Best overall (default) |

```python
# Set in .env:
RETRIEVAL_MODE=hybrid
RERANK_RESULTS=true   # cross-encoder reranking
```

### Vector Stores

```python
# FAISS — fast, local, no server needed
pipeline = RAGPipeline.build(vector_store_type="faiss")

# ChromaDB — persistent, production-ready
pipeline = RAGPipeline.build(vector_store_type="chroma")
```

### LLM Backends

```python
# GROQ (set GROQ_API_KEY in .env)
LLM_MODEL=llama-3.1-8b-instant

# Ollama (free, local — run: ollama pull llama3.2)
LLM_MODEL=llama3.2
```

---

## 🌐 REST API

Start the server:
```bash
python main.py serve
```

#### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/ingest` | Upload and index a file |
| `POST` | `/ingest/url` | Index a URL |
| `POST` | `/query` | Ask a question |
| `GET` | `/health` | Health check |
| `GET` | `/stats` | Pipeline statistics |
| `GET` | `/docs` | Swagger UI |

#### Example API Usage

```bash
# Index a URL
curl -X POST "http://localhost:8000/ingest/url?url=https://example.com/doc.html"

# Query
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?", "top_k": 5}'
```

---

## 📊 Evaluation

Run built-in evaluation metrics:

```bash
python main.py eval data/eval_samples.json
```

Metrics computed (from scratch, no RAGAS dependency):

| Metric | Description |
|---|---|
| **Hit Rate** | Was a relevant document retrieved at all? |
| **MRR** | Mean Reciprocal Rank of the first correct result |
| **Context Precision** | Fraction of retrieved chunks that are actually relevant |
| **Answer Faithfulness** | How grounded is the answer in the retrieved context? |
| **Answer Relevance** | How relevant is the answer to the original question? |

---

## 🧪 Running Tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## 🔧 Advanced Configuration

All settings are in `.env`. Key options:

```bash
# Chunking
CHUNK_SIZE=512
CHUNK_OVERLAP=64
CHUNKING_STRATEGY=recursive     # recursive | sentence | semantic

# Retrieval
TOP_K=5
RETRIEVAL_MODE=hybrid           # dense | sparse | hybrid
RERANK_RESULTS=true
RERANK_TOP_N=3

# Embeddings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
# or: text-embedding-3-small    (GROQ)

# Vector Store
VECTOR_STORE_TYPE=faiss         # faiss | chroma
```

---

## 💡 Using as a Python Library

```python
from src.pipeline import RAGPipeline

# Build pipeline
pipeline = RAGPipeline.build(
    vector_store_type="faiss",
    retrieval_mode="hybrid",
)

# Ingest
pipeline.ingest("./my_docs/")
pipeline.ingest("https://example.com/article")

# Query
response = pipeline.query("What does this say about X?")
print(response.answer)
print(response.sources)

# Save index
pipeline.save()

# Reload later
pipeline = RAGPipeline.build(load_existing=True)
```

---

## 🗺️ Roadmap / Extension Ideas

- [ ] Multi-modal RAG (images + text)
- [ ] Streaming API responses
- [ ] Parent-child chunk retrieval
- [ ] HyDE (Hypothetical Document Embeddings)
- [ ] Self-RAG with reflection
- [ ] Query rewriting / expansion
- [ ] Conversation memory / multi-turn RAG
- [ ] Async ingestion pipeline
- [ ] Docker + docker-compose setup
- [ ] LLM-based evaluation with RAGAS

---

## 🧠 What This Project Demonstrates

This project is deliberately built **without any RAG framework** to show:

1. **Deep understanding** of every RAG component, not just API calls
2. **Systems thinking** — modular design with clean interfaces
3. **Production awareness** — config management, logging, error handling, API layer
4. **Evaluation discipline** — measuring quality, not just building
5. **Practical trade-off knowledge** — when to use FAISS vs Chroma, BM25 vs dense, etc.

---

## 📄 License

MIT — use freely for personal and commercial projects.
