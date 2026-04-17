"""
Document Loader
---------------
Supports loading from:
  - PDF files
  - Plain text (.txt)
  - Markdown (.md)
  - Web URLs
  - Entire directories

Returns a list of Document objects with content + metadata.
"""

import os
import re
import logging
import requests
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a loaded document with its text and metadata."""
    content: str
    metadata: dict = field(default_factory=dict)

    def __repr__(self):
        preview = self.content[:80].replace("\n", " ")
        return f"Document(chars={len(self.content)}, source={self.metadata.get('source', 'unknown')}, preview='{preview}...')"


class DocumentLoader:
    """
    Loads documents from various sources into a unified Document format.

    Usage:
        loader = DocumentLoader()
        docs = loader.load("data/raw/report.pdf")
        docs = loader.load_directory("data/raw/")
        docs = loader.load_url("https://example.com/article")
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}

    def load(self, source: str) -> List[Document]:
        """
        Auto-detect source type and load accordingly.
        Accepts: file path, directory path, or URL.
        """
        if source.startswith("http://") or source.startswith("https://"):
            return self.load_url(source)

        path = Path(source)
        if path.is_dir():
            return self.load_directory(source)
        if path.is_file():
            return self._load_file(path)

        raise FileNotFoundError(f"Source not found: {source}")

    def load_directory(self, directory: str, recursive: bool = True) -> List[Document]:
        """Load all supported files from a directory."""
        docs = []
        path = Path(directory)
        pattern = "**/*" if recursive else "*"

        for file_path in path.glob(pattern):
            if file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                try:
                    loaded = self._load_file(file_path)
                    docs.extend(loaded)
                    logger.info(f"Loaded {file_path.name} → {len(loaded)} document(s)")
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")

        logger.info(f"Total documents loaded from directory: {len(docs)}")
        return docs

    def _load_file(self, path: Path) -> List[Document]:
        """Route file to the correct loader based on extension."""
        ext = path.suffix.lower()
        if ext == ".pdf":
            return self._load_pdf(path)
        elif ext in {".txt", ".md", ".markdown"}:
            return self._load_text(path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def _load_pdf(self, path: Path) -> List[Document]:
        """Load a PDF file, extracting text page by page."""
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("pypdf is required for PDF loading. Run: pip install pypdf")

        reader = PdfReader(str(path))
        docs = []

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = self._clean_text(text)
            if text.strip():
                docs.append(Document(
                    content=text,
                    metadata={
                        "source": str(path),
                        "filename": path.name,
                        "page": page_num + 1,
                        "total_pages": len(reader.pages),
                        "type": "pdf"
                    }
                ))

        return docs

    def _load_text(self, path: Path) -> List[Document]:
        """Load a plain text or markdown file."""
        encodings = ["utf-8", "utf-8-sig", "latin-1"]
        for enc in encodings:
            try:
                text = path.read_text(encoding=enc)
                text = self._clean_text(text)
                return [Document(
                    content=text,
                    metadata={
                        "source": str(path),
                        "filename": path.name,
                        "type": path.suffix.lstrip(".")
                    }
                )]
            except UnicodeDecodeError:
                continue

        raise UnicodeDecodeError(f"Could not decode {path} with any supported encoding.")

    def load_url(self, url: str, timeout: int = 10) -> List[Document]:
        """Fetch and extract text from a web URL."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("beautifulsoup4 is required for URL loading. Run: pip install beautifulsoup4")

        headers = {"User-Agent": "Mozilla/5.0 (RAG-Loader/1.0)"}
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove boilerplate elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        text = self._clean_text(text)

        return [Document(
            content=text,
            metadata={
                "source": url,
                "type": "web",
                "title": soup.title.string if soup.title else url
            }
        )]

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalize whitespace and remove junk characters."""
        text = re.sub(r"\r\n", "\n", text)           # normalize line endings
        text = re.sub(r"\n{3,}", "\n\n", text)        # collapse excessive newlines
        text = re.sub(r"[ \t]{2,}", " ", text)        # collapse excessive spaces
        text = re.sub(r"[^\x00-\x7F]+", " ", text)   # remove non-ASCII
        return text.strip()
