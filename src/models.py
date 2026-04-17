"""
src/models.py — Core data models shared across the entire pipeline.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


@dataclass
class Document:
    """A raw document before chunking."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    doc_id: str = field(default_factory=lambda: str(uuid4()))

    def __repr__(self) -> str:
        preview = self.content[:80].replace("\n", " ")
        return f"Document(id={self.doc_id[:8]}, chars={len(self.content)}, preview='{preview}...')"


@dataclass
class Chunk:
    """A single chunk derived from a Document."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_id: str = field(default_factory=lambda: str(uuid4()))
    doc_id: str = ""
    embedding: list[float] | None = None

    def __repr__(self) -> str:
        preview = self.content[:60].replace("\n", " ")
        return f"Chunk(id={self.chunk_id[:8]}, doc={self.doc_id[:8]}, chars={len(self.content)}, preview='{preview}...')"


@dataclass
class RetrievedChunk:
    """A chunk returned by the retriever with a relevance score."""

    chunk: Chunk
    score: float
    retrieval_method: str = "dense"  # dense | sparse | hybrid | reranked

    def __repr__(self) -> str:
        return f"RetrievedChunk(score={self.score:.4f}, method={self.retrieval_method}, chunk={self.chunk})"


@dataclass
class RAGResponse:
    """Final response from the RAG pipeline."""

    question: str
    answer: str
    retrieved_chunks: list[RetrievedChunk]
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def sources(self) -> list[dict[str, Any]]:
        return [
            {
                "chunk_id": rc.chunk.chunk_id,
                "doc_id": rc.chunk.doc_id,
                "score": rc.score,
                "method": rc.retrieval_method,
                "metadata": rc.chunk.metadata,
                "preview": rc.chunk.content[:200],
            }
            for rc in self.retrieved_chunks
        ]
