"""
Reranker
--------
Re-scores retrieval results using a cross-encoder model.

Cross-encoders jointly encode the query AND each candidate chunk,
giving much more accurate relevance scores than bi-encoder retrieval.
The trade-off: slower (N forward passes vs 1 for bi-encoders), so
we only rerank the top-K already retrieved — not the full corpus.

Usage:
    reranker = Reranker(model="cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranked = reranker.rerank(query="What is RAG?", results=retrieved_results, top_k=3)
"""

import logging
from typing import List, Optional
from src.vectorstore.faiss_store import SearchResult

logger = logging.getLogger(__name__)


class Reranker:
    """
    Cross-encoder reranker that improves retrieval precision.

    Recommended models (all free, from HuggingFace):
      - cross-encoder/ms-marco-MiniLM-L-6-v2   (fast, good quality)
      - cross-encoder/ms-marco-MiniLM-L-12-v2  (slower, better quality)
      - BAAI/bge-reranker-base                 (strong multilingual)
    """

    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model
        self._model = None
        self._load_model()

    def _load_model(self):
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
            logger.info(f"Reranker loaded: {self.model_name}")
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for reranking. "
                "Run: pip install sentence-transformers"
            )

    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Rerank a list of SearchResults using cross-encoder scores.

        Args:
            query:   The user's query string
            results: Initial retrieval results (from Retriever)
            top_k:   How many top results to return after reranking

        Returns:
            Reranked SearchResult list (best first)
        """
        if not results:
            return []

        top_k = top_k or len(results)

        # Create (query, passage) pairs for cross-encoder
        pairs = [(query, result.chunk.content) for result in results]

        # Score all pairs in one batch
        scores = self._model.predict(pairs)

        # Attach new scores and resort
        for result, score in zip(results, scores):
            result.score = float(score)

        reranked = sorted(results, key=lambda r: r.score, reverse=True)

        # Update ranks
        for i, result in enumerate(reranked):
            result.rank = i + 1

        logger.debug(
            f"Reranked {len(results)} results → returning top {top_k}. "
            f"Score range: {min(scores):.3f} – {max(scores):.3f}"
        )

        return reranked[:top_k]
