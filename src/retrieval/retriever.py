"""
src/retrieval/retriever.py — Hybrid retriever with optional reranking.

Modes:
  dense   — pure vector similarity
  sparse  — BM25 keyword search
  hybrid  — RRF (Reciprocal Rank Fusion) of dense + sparse scores
  
Reranking: cross-encoder model re-scores the top-k results.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from loguru import logger
from src.models import Chunk, RetrievedChunk
from config import settings


# ── BM25 sparse retriever ─────────────────────────────────────────────────────

@dataclass
class BM25Retriever:
    _bm25: object = field(default=None, init=False, repr=False)
    _chunks: list[Chunk] = field(default_factory=list, init=False)

    def index(self, chunks: list[Chunk]) -> None:
        from rank_bm25 import BM25Okapi
        self._chunks = chunks
        tokenized = [c.content.lower().split() for c in chunks]
        self._bm25 = BM25Okapi(tokenized)
        logger.info(f"BM25 index built with {len(chunks)} chunks")

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(query.lower().split())
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            RetrievedChunk(chunk=self._chunks[i], score=float(scores[i]), retrieval_method="sparse")
            for i in top_indices if scores[i] > 0
        ]


# ── Reranker (cross-encoder) ──────────────────────────────────────────────────

@dataclass
class CrossEncoderReranker:
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    _model: object = field(default=None, init=False, repr=False)

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading cross-encoder: {self.model_name}")
            self._model = CrossEncoder(self.model_name)

    def rerank(self, query: str, results: list[RetrievedChunk], top_n: int = 3) -> list[RetrievedChunk]:
        self._load()
        pairs = [[query, r.chunk.content] for r in results]
        scores = self._model.predict(pairs)
        reranked = sorted(
            [RetrievedChunk(chunk=r.chunk, score=float(s), retrieval_method="reranked")
             for r, s in zip(results, scores)],
            key=lambda x: x.score, reverse=True,
        )
        return reranked[:top_n]


# ── RRF fusion ────────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    dense_results: list[RetrievedChunk],
    sparse_results: list[RetrievedChunk],
    k: int = 60,
) -> list[RetrievedChunk]:
    """Combine two ranked lists using Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    chunk_map: dict[str, Chunk] = {}

    for rank, r in enumerate(dense_results):
        cid = r.chunk.chunk_id
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
        chunk_map[cid] = r.chunk

    for rank, r in enumerate(sparse_results):
        cid = r.chunk.chunk_id
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
        chunk_map[cid] = r.chunk

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [
        RetrievedChunk(chunk=chunk_map[cid], score=scores[cid], retrieval_method="hybrid")
        for cid in sorted_ids
    ]


# ── Main Retriever ────────────────────────────────────────────────────────────

@dataclass
class HybridRetriever:
    """
    Unified retriever supporting dense, sparse, and hybrid modes.
    Optionally reranks results with a cross-encoder.
    """
    vector_store: object           # FAISSVectorStore | ChromaVectorStore
    bm25: BM25Retriever = field(default_factory=BM25Retriever)
    reranker: CrossEncoderReranker = field(default_factory=CrossEncoderReranker)
    mode: str = settings.retrieval_mode
    top_k: int = settings.top_k
    rerank: bool = settings.rerank_results
    rerank_top_n: int = settings.rerank_top_n

    def index_chunks(self, chunks: list[Chunk]) -> None:
        """Index chunks into both dense and sparse stores."""
        self.vector_store.add_chunks(chunks)
        self.bm25.index(chunks)

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        k = top_k or self.top_k
        mode = self.mode.lower()

        if mode == "dense":
            results = self.vector_store.search(query, top_k=k)
        elif mode == "sparse":
            results = self.bm25.search(query, top_k=k)
        elif mode == "hybrid":
            dense_results = self.vector_store.search(query, top_k=k)
            sparse_results = self.bm25.search(query, top_k=k)
            results = reciprocal_rank_fusion(dense_results, sparse_results)[:k]
        else:
            raise ValueError(f"Unknown retrieval mode: '{mode}'")

        logger.debug(f"Retrieved {len(results)} chunks via {mode} search")

        if self.rerank and results:
            results = self.reranker.rerank(query, results, top_n=self.rerank_top_n)
            logger.debug(f"Reranked to top {len(results)} chunks")

        return results
