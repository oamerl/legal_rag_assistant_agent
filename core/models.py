"""
Core data models for the Legal RAG Assistant.

These dataclasses define the shared vocabulary used across all pipeline
stages — ingestion, retrieval, and generation.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


# ── Ingestion Models ──────────────────────────────────────────────────


@dataclass
class ParsedDocument:
    """Result of parsing a raw file (PDF / DOCX) with Docling."""

    doc_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    doc_name: str = ""
    file_path: str = ""
    # The raw Docling `DoclingDocument` object — kept opaque so the
    # chunker can consume it without the rest of the app importing Docling.
    docling_document: Any = None
    page_count: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class ChunkMetadata:
    """Rich metadata attached to every chunk for filtering and citations."""

    document_id: str = ""
    document_name: str = ""
    page_number: int = 0
    section_id: str = ""            # e.g. "2.3.1"
    section_title: str = ""         # e.g. "Limitation of Liability"
    chunk_level: int = 0            # 1=article, 2=section, 3=clause
    parent_chunk_id: str | None = None
    heading_path: list[str] = field(default_factory=list)
    char_start: int = 0
    char_end: int = 0
    raw_text: str = ""              # Original chunk text before contextualize()


@dataclass
class Chunk:
    """A single text chunk ready for embedding and storage."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    metadata: ChunkMetadata = field(default_factory=ChunkMetadata)

    # Populated after embedding (not stored — transient)
    dense_embedding: list[float] | None = None
    sparse_indices: list[int] | None = None
    sparse_values: list[float] | None = None


# ── Retrieval Models ──────────────────────────────────────────────────


@dataclass
class RankedChunk:
    """A chunk that has been scored by the reranker."""

    chunk: Chunk
    score: float = 0.0


@dataclass
class RetrievalResult:
    """Output of the full retrieval pipeline."""

    query: str = ""
    sub_queries: list[str] = field(default_factory=list)
    ranked_chunks: list[RankedChunk] = field(default_factory=list)
    assembled_context: str = ""
    is_relevant: bool = True  # False if no chunks pass threshold


# ── Generation Models ─────────────────────────────────────────────────


@dataclass
class Citation:
    """A single citation linking an answer claim to its source."""

    document_name: str = ""
    page_number: int = 0
    section_id: str = ""
    section_title: str = ""
    quoted_text: str = ""


@dataclass
class LegalAnswer:
    """Structured output from the LLM generator."""

    answer: str = ""
    citations: list[Citation] = field(default_factory=list)
    confidence: float = 0.0
    unanswered_parts: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)


@dataclass
class GuardResult:
    """Output of the faithfulness guard."""

    faithfulness_score: float | None = None
    answer_relevancy_score: float | None = None
    context_precision_score: float | None = None


# ── RAGAS evaluation Models ──────────────────────────────────────────

@dataclass
class EvaluationSample:
    """A single evaluation sample (question + ground truth + prediction)."""

    question: str = ""
    answer: str = ""                   # Generated answer
    contexts: list[str] = field(default_factory=list)  # Retrieved chunks
    ground_truth: str = ""             # Expected answer (for recall)


@dataclass
class EvaluationResult:
    """Results from a RAGAS evaluation run."""

    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    num_samples: int = 0


# ── Pipeline Data Container ──────────────────────────────────────────


@dataclass
class PipelineData:
    """
    Generic data bag passed between pipeline stages.

    Each stage reads what it needs and writes its outputs into this
    container so downstream stages can consume them.
    """

    # Ingestion
    file_path: str = ""
    parsed_document: ParsedDocument | None = None
    chunks: list[Chunk] = field(default_factory=list)

    # Retrieval
    query: str = ""
    sub_queries: list[str] = field(default_factory=list)
    query_type: str = ""               # SIMPLE / COMPLEX / SPECIFIC / AMBIGUOUS
    query_embeddings: list[list[float]] = field(default_factory=list)
    retrieved_chunks: list[Chunk] = field(default_factory=list)
    ranked_chunks: list[RankedChunk] = field(default_factory=list)
    assembled_context: str = ""
    retrieval_is_relevant: bool = True

    # Generation
    legal_answer: LegalAnswer | None = None
    guard_result: GuardResult | None = None

    # Session
    conversation_id: str = ""
    conversation_history: list[dict] = field(default_factory=list)

    # Metadata / diagnostics
    diagnostics: dict = field(default_factory=dict)
