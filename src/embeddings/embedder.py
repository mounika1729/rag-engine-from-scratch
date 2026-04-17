"""
src/embeddings/embedder.py — Embedding model wrappers.

Supports:
  • SentenceTransformerEmbedder  (local, free — default)
  • OpenAIEmbedder               (API, paid)
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from loguru import logger
from tqdm import tqdm
from config import settings


class EmbeddingModel(ABC):
    @abstractmethod
    def embed_text(self, text: str) -> list[float]: ...
    @abstractmethod
    def embed_texts(self, texts: list[str], batch_size: int = 64) -> list[list[float]]: ...
    @property
    @abstractmethod
    def dimension(self) -> int: ...


@dataclass
class SentenceTransformerEmbedder(EmbeddingModel):
    """Local embeddings via sentence-transformers (free, no API key needed)."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    _model: object = field(default=None, init=False, repr=False)

    def _load(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)

    def embed_text(self, text: str) -> list[float]:
        self._load()
        return self._model.encode(text, normalize_embeddings=True).tolist()

    def embed_texts(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        self._load()
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding", unit="batch"):
            batch = texts[i:i+batch_size]
            vecs = self._model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
            all_embeddings.extend(vecs.tolist())
        return all_embeddings

    @property
    def dimension(self) -> int:
        self._load()
        return self._model.get_sentence_embedding_dimension()


@dataclass
class OpenAIEmbedder(EmbeddingModel):
    """OpenAI text-embedding-3-small or text-embedding-3-large."""
    model_name: str = "text-embedding-3-small"
    _client: object = field(default=None, init=False, repr=False)

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=settings.openai_api_key)
        return self._client

    def embed_text(self, text: str) -> list[float]:
        resp = self._get_client().embeddings.create(input=[text], model=self.model_name)
        return resp.data[0].embedding

    def embed_texts(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="OpenAI Embed", unit="batch"):
            batch = texts[i:i+batch_size]
            resp = self._get_client().embeddings.create(input=batch, model=self.model_name)
            all_embeddings.extend([r.embedding for r in sorted(resp.data, key=lambda x: x.index)])
        return all_embeddings

    @property
    def dimension(self) -> int:
        return {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}.get(self.model_name, 1536)


def get_embedding_model(model_name: str | None = None) -> EmbeddingModel:
    name = model_name or settings.embedding_model
    if name.startswith("text-embedding"):
        return OpenAIEmbedder(model_name=name)
    return SentenceTransformerEmbedder(model_name=name)
