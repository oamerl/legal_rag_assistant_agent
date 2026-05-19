"""
Custom exception hierarchy for the Legal RAG Assistant.

All application-specific exceptions inherit from LegalRAGError
so callers can catch broad or narrow as needed.
"""


class LegalRAGError(Exception):
    """Base exception for the Legal RAG Assistant."""


# ── Ingestion Errors ──────────────────────────────────────────────────

class IngestionError(LegalRAGError):
    """Base for all ingestion-pipeline errors."""


class UnsupportedFileTypeError(IngestionError):
    """Raised when a file type is not supported by any parser."""


class ParsingError(IngestionError):
    """Raised when document parsing fails."""


class ChunkingError(IngestionError):
    """Raised when chunking a parsed document fails."""


class EmbeddingError(IngestionError):
    """Raised when embedding generation fails."""


class IndexingError(IngestionError):
    """Raised when storing chunks in the vector store fails."""


# ── Retrieval Errors ──────────────────────────────────────────────────

class RetrievalError(LegalRAGError):
    """Base for all retrieval-pipeline errors."""


class QueryTransformError(RetrievalError):
    """Raised when query transformation (HyDE, decomposition, etc.) fails."""


class SearchError(RetrievalError):
    """Raised when vector/sparse search fails."""


class RerankingError(RetrievalError):
    """Raised when the reranker encounters an error."""


class NoRelevantChunksError(RetrievalError):
    """Raised when no chunks pass the relevance threshold."""


# ── Generation Errors ─────────────────────────────────────────────────

class GenerationError(LegalRAGError):
    """Base for all generation errors."""


class LLMClientError(GenerationError):
    """Raised when the LLM API call fails."""


class EvaluationError(GenerationError):
    """Raised when the evaluation fails."""


# ── Configuration Errors ──────────────────────────────────────────────

class ConfigurationError(LegalRAGError):
    """Raised for invalid or missing configuration."""
