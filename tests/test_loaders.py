"""Tests for document loaders."""
import pytest
import tempfile
from pathlib import Path
import sys; sys.path.insert(0, ".")
from src.ingestion.loaders import load_text, load_document, load_directory


def test_load_text_file():
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
        f.write("Hello world. This is a test document.")
        tmp = Path(f.name)
    doc = load_text(tmp)
    assert "Hello world" in doc.content
    assert doc.metadata["file_type"] == "txt"
    tmp.unlink()


def test_load_document_dispatch_txt():
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
        f.write("Test content")
        tmp = Path(f.name)
    doc = load_document(tmp)
    assert doc.content == "Test content"
    tmp.unlink()


def test_load_markdown_file():
    with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
        f.write("# Title\nSome markdown content.")
        tmp = Path(f.name)
    doc = load_document(tmp)
    assert "Title" in doc.content
    tmp.unlink()


def test_load_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(3):
            Path(tmpdir, f"doc{i}.txt").write_text(f"Document {i} content.")
        docs = load_directory(tmpdir)
        assert len(docs) == 3


def test_unsupported_extension():
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_document(Path("file.xyz"))


def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_document(Path("/nonexistent/path/file.txt"))
