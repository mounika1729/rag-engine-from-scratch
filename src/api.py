"""
src/api.py — FastAPI REST API for the RAG pipeline.

Endpoints:
  POST /ingest   — Upload and index a file
  POST /query    — Ask a question
  GET  /health   — Health check
  GET  /stats    — Pipeline statistics
"""

from __future__ import annotations
import time
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from loguru import logger
from config import settings
from src.pipeline import RAGPipeline


# ── Global pipeline instance ──────────────────────────────────────────────────

pipeline: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    logger.info("Initialising RAG pipeline...")
    pipeline = RAGPipeline.build()
    logger.info("RAG API ready.")
    yield
    if pipeline:
        pipeline.save()


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG From Scratch",
    description="A production-grade Retrieval-Augmented Generation API built from scratch.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    top_k: int = settings.top_k

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[dict]
    model: str
    latency_ms: float

class IngestResponse(BaseModel):
    message: str
    chunks_indexed: int

class HealthResponse(BaseModel):
    status: str
    vector_store_size: int


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    size = len(pipeline.retriever.vector_store) if pipeline else 0
    return {"status": "ok", "vector_store_size": size}


@app.post("/ingest", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...)):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")
    tmp = Path(f"/tmp/{file.filename}")
    tmp.write_bytes(await file.read())
    try:
        n_chunks = pipeline.ingest(tmp)
        return {"message": f"Indexed '{file.filename}'", "chunks_indexed": n_chunks}
    finally:
        tmp.unlink(missing_ok=True)


@app.post("/ingest/url", response_model=IngestResponse)
def ingest_url(url: str):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")
    n_chunks = pipeline.ingest(url)
    return {"message": f"Indexed URL: {url}", "chunks_indexed": n_chunks}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")
    t0 = time.perf_counter()
    response = pipeline.query(req.question, top_k=req.top_k)
    latency = (time.perf_counter() - t0) * 1000
    return QueryResponse(
        question=response.question,
        answer=response.answer,
        sources=response.sources,
        model=response.model,
        latency_ms=round(latency, 2),
    )


@app.get("/stats")
def stats():
    if pipeline is None:
        return {"error": "Pipeline not ready"}
    return {
        "vector_store_type": settings.vector_store_type,
        "vector_store_size": len(pipeline.retriever.vector_store),
        "retrieval_mode": settings.retrieval_mode,
        "embedding_model": settings.embedding_model,
        "llm_model": settings.llm_model,
        "reranking_enabled": settings.rerank_results,
    }
