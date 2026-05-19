"""
Central configuration for the Legal RAG Assistant.

All settings are loaded from environment variables or a .env file.
Uses Pydantic BaseSettings for typed, validated configuration.
"""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


# Resolve project root relative to this file
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application-wide settings loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Device Configuration ──────────────────────────────────────────
    device: Literal["cpu", "cuda"] = "cpu"

    # ── Embedding Configuration ───────────────────────────────────────
    embedding_provider: Literal["local", "voyage", "openrouter"] = "openrouter"

    # Local (FastEmbed / ONNX) model names
    dense_model_name: str = "BAAI/bge-large-en-v1.5"
    sparse_model_name: str = "prithivida/Splade_PP_en_v1"

    # Voyage AI (optional, used when embedding_provider="voyage")
    voyage_api_key: str = ""
    voyage_model_name: str = "voyage-law-2"

    # OpenRouter Embedding (used when embedding_provider="openrouter")
    # Options: "openai/text-embedding-3-large", "openai/text-embedding-3-small"
    openrouter_embedding_model: str = "openai/text-embedding-3-large"
    # Optional dimension control (useful for models like text-embedding-3-small that support reduced dimensions)
    openrouter_embedding_dimensions: int | None = None  # None = model default

    # ── Reranker Configuration ────────────────────────────────────────
    # Provider options:
    #   "local"      - Runs a cross-encoder locally (e.g. BAAI/bge-reranker-base)
    #   "openrouter" - Uses OpenRouter's rerank API (e.g. cohere/rerank-v3.5)
    reranker_provider: Literal["local", "openrouter"] = "openrouter"
    
    # Local options: "BAAI/bge-reranker-large", "BAAI/bge-reranker-base"
    reranker_model_name: str = "BAAI/bge-reranker-base"
    openrouter_reranker_model: str = "cohere/rerank-v3.5"
    reranker_top_n: int = 8
    reranker_relevance_threshold: float = 0.1

    # ── LLM Configuration (OpenRouter) ────────────────────────────────
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "openai/gpt-4o"
    llm_model_small: str = "openai/gpt-4o-mini"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 4096

    # ── Evaluation ─────────────────────────────────────────────────────
    # Enable/disable the faithfulness guard (RAGAS evaluation) stage.
    # Set to false to skip post-generation verification and reduce latency.
    enable_faithfulness_guard: bool = True

    # ── Chunking Configuration ────────────────────────────────────────
    chunk_max_tokens: int | None = None  # None = auto-derive from tokenizer
    chunk_merge_peers: bool = True

    # ── Retrieval Configuration ───────────────────────────────────────
    retrieval_top_k: int = 20
    # Set to None to disable context window overflow protection
    context_max_tokens: int | None = None

    # ── Qdrant Configuration ─────────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_name: str = "legal_docs"
    # Set to ":memory:" for in-memory mode (desktop, no Docker needed)
    qdrant_path: str = str(PROJECT_ROOT / "data" / "qdrant_storage")

    # ── Session / Conversation Configuration ──────────────────────────
    session_db_path: str = str(PROJECT_ROOT / "data" / "sessions.db")
    checkpointer_db_uri: str = "sqlite:///"

    # ── File Storage ──────────────────────────────────────────────────
    upload_dir: str = str(PROJECT_ROOT / "data" / "uploads")

    # ── Logging ───────────────────────────────────────────────────────
    log_level: str = "INFO"
    log_file: str = str(PROJECT_ROOT / "logs" / "legal_rag.log")


def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
