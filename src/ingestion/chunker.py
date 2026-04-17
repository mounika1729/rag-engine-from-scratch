"""
src/ingestion/chunker.py — Text chunking strategies.

Three strategies:
  1. recursive  — paragraph → sentence → word boundary (default)
  2. sentence   — strict sentence windows
  3. semantic   — cosine-similarity-based grouping
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from loguru import logger
from config import settings
from src.models import Chunk, Document


def _split_sentences(text: str) -> list[str]:
    raw = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in raw if s.strip()]


def _merge_sentences(sentences: list[str], max_size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for sent in sentences:
        if current_len + len(sent) + 1 > max_size and current:
            chunks.append(" ".join(current))
            tail, tail_len = [], 0
            for s in reversed(current):
                if tail_len + len(s) < overlap:
                    tail.insert(0, s); tail_len += len(s)
                else:
                    break
            current, current_len = tail, tail_len
        current.append(sent); current_len += len(sent) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


@dataclass
class RecursiveChunker:
    chunk_size: int = settings.chunk_size
    chunk_overlap: int = settings.chunk_overlap
    SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        if not text: return []
        if len(text) <= self.chunk_size: return [text]
        sep = separators[0]; next_seps = separators[1:]
        if sep == "":
            return [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]
        parts = text.split(sep); chunks = []; current = []; current_len = 0
        for part in parts:
            part_len = len(part) + len(sep)
            if current_len + part_len > self.chunk_size and current:
                merged = sep.join(current)
                chunks.extend([merged] if len(merged) <= self.chunk_size else self._recursive_split(merged, next_seps))
                overlap_parts, overlap_len = [], 0
                for p in reversed(current):
                    if overlap_len + len(p) < self.chunk_overlap:
                        overlap_parts.insert(0, p); overlap_len += len(p)
                    else: break
                current, current_len = overlap_parts, overlap_len
            current.append(part); current_len += part_len
        if current:
            merged = sep.join(current)
            chunks.extend([merged] if len(merged) <= self.chunk_size else self._recursive_split(merged, next_seps))
        return [c.strip() for c in chunks if c.strip()]

    def split(self, document: Document) -> list[Chunk]:
        raw = self._recursive_split(document.content, self.SEPARATORS)
        return [Chunk(content=c, metadata={**document.metadata, "chunk_index": i}, doc_id=document.doc_id) for i, c in enumerate(raw)]


@dataclass
class SentenceChunker:
    chunk_size: int = settings.chunk_size
    chunk_overlap: int = settings.chunk_overlap

    def split(self, document: Document) -> list[Chunk]:
        sentences = _split_sentences(document.content)
        raw = _merge_sentences(sentences, self.chunk_size, self.chunk_overlap)
        return [Chunk(content=c, metadata={**document.metadata, "chunk_index": i}, doc_id=document.doc_id) for i, c in enumerate(raw)]


@dataclass
class SemanticChunker:
    embedding_model: object
    breakpoint_threshold: float = 0.85
    chunk_size: int = settings.chunk_size

    def split(self, document: Document) -> list[Chunk]:
        import numpy as np
        sentences = _split_sentences(document.content)
        if len(sentences) < 2:
            return [Chunk(content=document.content, metadata=document.metadata, doc_id=document.doc_id)]
        embeddings = self.embedding_model.embed_texts(sentences)
        def cosine(a, b):
            a, b = np.array(a), np.array(b)
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        groups = [[sentences[0]]]
        for i in range(1, len(sentences)):
            sim = cosine(embeddings[i-1], embeddings[i])
            if sim >= self.breakpoint_threshold and sum(len(s) for s in groups[-1]) < self.chunk_size:
                groups[-1].append(sentences[i])
            else:
                groups.append([sentences[i]])
        return [Chunk(content=" ".join(g), metadata={**document.metadata, "chunk_index": i}, doc_id=document.doc_id) for i, g in enumerate(groups)]


def get_chunker(strategy: str = settings.chunking_strategy, **kwargs):
    s = strategy.lower()
    if s == "recursive": return RecursiveChunker(**kwargs)
    if s == "sentence": return SentenceChunker(**kwargs)
    if s == "semantic":
        if "embedding_model" not in kwargs: raise ValueError("SemanticChunker needs 'embedding_model'.")
        return SemanticChunker(**kwargs)
    raise ValueError(f"Unknown strategy '{strategy}'")


def chunk_documents(documents: list[Document], strategy: str = settings.chunking_strategy, **kwargs) -> list[Chunk]:
    chunker = get_chunker(strategy, **kwargs)
    all_chunks = []
    for doc in documents:
        chunks = chunker.split(doc); all_chunks.extend(chunks)
        logger.debug(f"Doc {doc.doc_id[:8]} -> {len(chunks)} chunks")
    logger.info(f"Total chunks: {len(all_chunks)} from {len(documents)} docs")
    return all_chunks
