"""Integration tests for the RAG pipeline (no LLM calls)."""
import pytest
import tempfile
from pathlib import Path
import sys; sys.path.insert(0, ".")
from src.models import Document, Chunk
from src.ingestion.chunker import chunk_documents


def make_chunks(n: int = 10) -> list[Chunk]:
    docs = [Document(content=f"This is document {i}. It talks about topic {i} in detail.", metadata={"source": f"doc{i}.txt"}) for i in range(n)]
    return chunk_documents(docs, strategy="sentence")


def test_chunk_documents_returns_chunks():
    chunks = make_chunks(5)
    assert len(chunks) > 0
    assert all(isinstance(c, Chunk) for c in chunks)


def test_chunk_unique_ids():
    chunks = make_chunks(5)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))  # all unique


def test_chunk_doc_id_propagation():
    chunks = make_chunks(3)
    assert all(c.doc_id for c in chunks)
