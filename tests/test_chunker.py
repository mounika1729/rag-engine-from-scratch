"""Tests for chunking strategies."""
import pytest
import sys; sys.path.insert(0, ".")
from src.models import Document
from src.ingestion.chunker import RecursiveChunker, SentenceChunker, get_chunker

SAMPLE_TEXT = """
Artificial intelligence (AI) is intelligence demonstrated by machines. 
Unlike the natural intelligence displayed by animals including humans, AI is intelligence demonstrated by machines.

Machine learning is a subset of AI. It gives systems the ability to automatically learn and improve from experience.
Deep learning is a subset of machine learning. It uses neural networks with many layers to process data.

Natural language processing (NLP) is a subfield of AI. It focuses on the interaction between computers and humans through language.
""".strip()


@pytest.fixture
def doc():
    return Document(content=SAMPLE_TEXT, metadata={"source": "test"})


def test_recursive_chunker_basic(doc):
    chunker = RecursiveChunker(chunk_size=200, chunk_overlap=20)
    chunks = chunker.split(doc)
    assert len(chunks) > 0
    for c in chunks:
        assert len(c.content) <= 300  # allow some slack
        assert c.doc_id == doc.doc_id


def test_sentence_chunker_basic(doc):
    chunker = SentenceChunker(chunk_size=200, chunk_overlap=30)
    chunks = chunker.split(doc)
    assert len(chunks) > 0


def test_chunk_overlap_content(doc):
    chunker = RecursiveChunker(chunk_size=150, chunk_overlap=50)
    chunks = chunker.split(doc)
    # All chunks should be non-empty
    assert all(len(c.content.strip()) > 0 for c in chunks)


def test_get_chunker_factory():
    assert isinstance(get_chunker("recursive"), RecursiveChunker)
    assert isinstance(get_chunker("sentence"), SentenceChunker)
    with pytest.raises(ValueError):
        get_chunker("unknown")


def test_metadata_propagation(doc):
    chunker = RecursiveChunker(chunk_size=200)
    chunks = chunker.split(doc)
    for i, c in enumerate(chunks):
        assert c.metadata["source"] == "test"
        assert c.metadata["chunk_index"] == i
