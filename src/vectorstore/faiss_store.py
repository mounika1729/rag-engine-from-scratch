"""
src/vectorstore/faiss_store.py — FAISS-backed vector store (from scratch, no LangChain).

Features:
  • add / search / delete chunks
  • persist to disk and reload
  • inner-product (cosine) similarity search
"""

from __future__ import annotations
import json
import pickle
from pathlib import Path
from dataclasses import dataclass, field
from loguru import logger
import numpy as np
from src.models import Chunk, RetrievedChunk
from src.embeddings.embedder import EmbeddingModel


@dataclass
class FAISSVectorStore:
    embedding_model: EmbeddingModel
    index_path: Path = Path("./data/vectorstore/faiss")
    _index: object = field(default=None, init=False, repr=False)
    _chunks: list[Chunk] = field(default_factory=list, init=False)

    def _get_index(self, dim: int):
        """Lazily create FAISS index."""
        import faiss
        if self._index is None:
            self._index = faiss.IndexFlatIP(dim)  # Inner Product = cosine for normalised vecs
        return self._index

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        texts = [c.content for c in chunks]
        embeddings = self.embedding_model.embed_texts(texts)
        dim = len(embeddings[0])
        index = self._get_index(dim)
        import faiss
        vectors = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(vectors)
        index.add(vectors)
        self._chunks.extend(chunks)
        # Store embeddings on chunks
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb
        logger.info(f"Added {len(chunks)} chunks. Total: {index.ntotal}")

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if self._index is None or self._index.ntotal == 0:
            logger.warning("Vector store is empty.")
            return []
        import faiss
        query_vec = np.array([self.embedding_model.embed_text(query)], dtype=np.float32)
        faiss.normalize_L2(query_vec)
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query_vec, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append(RetrievedChunk(chunk=self._chunks[idx], score=float(score), retrieval_method="dense"))
        return results

    def save(self) -> None:
        import faiss
        self.index_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.index_path / "index.faiss"))
        with open(self.index_path / "chunks.pkl", "wb") as f:
            pickle.dump(self._chunks, f)
        logger.info(f"Saved FAISS store to {self.index_path}")

    def load(self) -> None:
        import faiss
        idx_file = self.index_path / "index.faiss"
        chunks_file = self.index_path / "chunks.pkl"
        if not idx_file.exists():
            raise FileNotFoundError(f"No saved index at {idx_file}")
        self._index = faiss.read_index(str(idx_file))
        with open(chunks_file, "rb") as f:
            self._chunks = pickle.load(f)
        logger.info(f"Loaded FAISS store: {self._index.ntotal} vectors")

    def __len__(self) -> int:
        return self._index.ntotal if self._index else 0
