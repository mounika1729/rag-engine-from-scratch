"""
src/vectorstore/chroma_store.py — ChromaDB-backed vector store.
Persists automatically to disk; good for production use.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from loguru import logger
from src.models import Chunk, RetrievedChunk
from src.embeddings.embedder import EmbeddingModel
from config import settings


@dataclass
class ChromaVectorStore:
    embedding_model: EmbeddingModel
    collection_name: str = settings.chroma_collection_name
    persist_directory: str = str(settings.vector_store_path / "chroma")
    _collection: object = field(default=None, init=False, repr=False)

    def _get_collection(self):
        if self._collection is None:
            import chromadb
            client = chromadb.PersistentClient(path=self.persist_directory)
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        col = self._get_collection()
        texts = [c.content for c in chunks]
        embeddings = self.embedding_model.embed_texts(texts)
        col.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{**c.metadata, "doc_id": c.doc_id} for c in chunks],
        )
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb
        logger.info(f"Added {len(chunks)} chunks to ChromaDB collection '{self.collection_name}'")

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        col = self._get_collection()
        query_emb = self.embedding_model.embed_text(query)
        results = col.query(query_embeddings=[query_emb], n_results=min(top_k, col.count()))
        retrieved = []
        if not results["ids"][0]:
            return []
        for i, chunk_id in enumerate(results["ids"][0]):
            chunk = Chunk(
                content=results["documents"][0][i],
                metadata=results["metadatas"][0][i],
                chunk_id=chunk_id,
                doc_id=results["metadatas"][0][i].get("doc_id", ""),
            )
            score = 1.0 - results["distances"][0][i]  # chroma returns distance
            retrieved.append(RetrievedChunk(chunk=chunk, score=score, retrieval_method="dense"))
        return retrieved

    def __len__(self) -> int:
        return self._get_collection().count()
