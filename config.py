"""
config.py — Centralised settings loaded from environment variables.
All modules import from here; never read os.environ directly.
"""

from __future__ import annotations
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────────────────────
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")
    llm_temperature: float = Field(default=0.0, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=1024, alias="LLM_MAX_TOKENS")

    # ── Embeddings ────────────────────────────────────────────────────────
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL",
    )

    # ── Vector Store ──────────────────────────────────────────────────────
    vector_store_type: str = Field(default="faiss", alias="VECTOR_STORE_TYPE")
    vector_store_path: Path = Field(
        default=Path("./data/vectorstore"), alias="VECTOR_STORE_PATH"
    )
    chroma_collection_name: str = Field(
        default="rag_collection", alias="CHROMA_COLLECTION_NAME"
    )

    # ── Chunking ──────────────────────────────────────────────────────────
    chunk_size: int = Field(default=512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=64, alias="CHUNK_OVERLAP")
    chunking_strategy: str = Field(default="recursive", alias="CHUNKING_STRATEGY")

    # ── Retrieval ─────────────────────────────────────────────────────────
    top_k: int = Field(default=5, alias="TOP_K")
    retrieval_mode: str = Field(default="hybrid", alias="RETRIEVAL_MODE")
    rerank_results: bool = Field(default=True, alias="RERANK_RESULTS")
    rerank_top_n: int = Field(default=3, alias="RERANK_TOP_N")

    # ── API ───────────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    api_reload: bool = Field(default=True, alias="API_RELOAD")

    # ── Logging ───────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_file: Path = Field(default=Path("./logs/rag.log"), alias="LOG_FILE")


settings = Settings()
