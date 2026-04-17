"""
src/pipeline.py — Orchestrates the full RAG pipeline.

Usage:
    pipeline = RAGPipeline.build()          # uses config defaults
    pipeline.ingest("./data/raw")           # or a file path / URL
    response = pipeline.query("What is X?")
    print(response.answer)
"""

from __future__ import annotations
from pathlib import Path
from loguru import logger
from config import settings
from src.models import RAGResponse
from src.ingestion.loaders import load_document, load_directory
from src.ingestion.chunker import chunk_documents
from src.embeddings.embedder import get_embedding_model, EmbeddingModel
from src.retrieval.retriever import HybridRetriever, BM25Retriever, CrossEncoderReranker
from src.generation.generator import get_generator
from src.generation.generator import GroqGenerator

class RAGPipeline:
    def __init__(self, retriever: HybridRetriever, generator):
        self.retriever = retriever
        self.generator = generator
        logger.info("RAGPipeline ready.")

    # ── factory ───────────────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        vector_store_type: str = settings.vector_store_type,
        embedding_model: EmbeddingModel | None = None,
        llm_model: str | None = None,
        retrieval_mode: str = settings.retrieval_mode,
        load_existing: bool = False,
    ) -> "RAGPipeline":
        """
        Build a RAGPipeline from config defaults.

        Args:
            vector_store_type: 'faiss' or 'chroma'
            embedding_model:   Custom EmbeddingModel instance (optional)
            llm_model:         Model name string (optional)
            retrieval_mode:    'dense' | 'sparse' | 'hybrid'
            load_existing:     Load a previously saved FAISS index from disk
        """
        emb = embedding_model or get_embedding_model()

        # Vector store
        if vector_store_type == "faiss":
            from src.vectorstore.faiss_store import FAISSVectorStore
            store = FAISSVectorStore(embedding_model=emb, index_path=settings.vector_store_path / "faiss")
            if load_existing:
                store.load()
        elif vector_store_type == "chroma":
            from src.vectorstore.chroma_store import ChromaVectorStore
            store = ChromaVectorStore(embedding_model=emb)
        else:
            raise ValueError(f"Unknown vector store: '{vector_store_type}'")

        retriever = HybridRetriever(
            vector_store=store,
            bm25=BM25Retriever(),
            reranker=CrossEncoderReranker(),
            mode=retrieval_mode,
        )

        generator = GroqGenerator()
        return cls(retriever=retriever, generator=generator)

    # ── public API ────────────────────────────────────────────────────────────

    def ingest(self, source: str | Path, chunking_strategy: str = settings.chunking_strategy) -> int:
        """
        Load and index a file, URL, or directory.

        Returns:
            Number of chunks indexed.
        """
        src = Path(source) if not str(source).startswith("http") else source

        if isinstance(src, Path) and src.is_dir():
            docs = load_directory(src)
        else:
            docs = [load_document(source)]

        chunks = chunk_documents(docs, strategy=chunking_strategy)
        self.retriever.index_chunks(chunks)
        logger.info(f"Ingested {len(docs)} doc(s) → {len(chunks)} chunks")
        return len(chunks)

    def query(self, question: str, top_k: int | None = None) -> RAGResponse:
        """
        Full RAG query: retrieve → generate → return.

        Args:
            question: Natural language question.
            top_k:    Override default top_k for this query.

        Returns:
            RAGResponse with answer and source chunks.
        """
        logger.info(f"Query: '{question}'")
        chunks = self.retriever.retrieve(question, top_k=top_k)

        if not chunks:
            from src.models import RAGResponse
            return RAGResponse(
                question=question,
                answer="I could not find any relevant information to answer your question.",
                retrieved_chunks=[],
                model="none",
            )

        response = self.generator.generate(question, chunks)
        logger.info(f"Answer generated ({response.completion_tokens} tokens)")
        return response

    def save(self) -> None:
        """Persist the vector store to disk (FAISS only)."""
        if hasattr(self.retriever.vector_store, "save"):
            self.retriever.vector_store.save()
            logger.info("Pipeline state saved.")
        else:
            logger.warning("Current vector store does not support manual saving (Chroma auto-saves).")
