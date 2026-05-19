"""
Context builder — assembles the final context from reranked chunks.

Performs parent document expansion and enforces
a token budget to prevent context window overflow.

Design Pattern: Builder Pattern
    Step-by-step context assembly with configurable constraints.
"""

from __future__ import annotations

import logging

from config.settings import get_settings
from core.models import PipelineData, RankedChunk
from core.pipeline import PipelineStage

logger = logging.getLogger(__name__)


class ContextBuilder:
    """
    Assembles final context from reranked chunks.

    Steps:
        1. Format each ranked chunk with its metadata (for citation)
        2. Expand to parent chunks if available (parent document retrieval)
        3. Enforce token budget (optional)
    """

    def __init__(self, max_context_tokens: int | None = None) -> None:
        settings = get_settings()
        self._max_tokens = max_context_tokens or settings.context_max_tokens

    def build(self, ranked_chunks: list[RankedChunk]) -> str:
        """
        Build the assembled context string from ranked chunks.

        Parameters
        ----------
        ranked_chunks : list[RankedChunk]
            Chunks sorted by relevance (highest first).

        Returns
        -------
        str
            Formatted context string ready for the LLM prompt.
        """
        if not ranked_chunks:
            return ""

        context_parts: list[str] = []
        seen_texts: set[str] = set()
        approx_tokens = 0

        for i, rc in enumerate(ranked_chunks, 1):
            chunk = rc.chunk
            text = chunk.text.strip()

            # Skip duplicates
            # Note in HybridRetrievalStage we already removed duplicate chunks by chunk_id
            # This is just a fallback if Context Builder is not used within the pipeline
            text_key = text[:200]
            if text_key in seen_texts:
                continue
            seen_texts.add(text_key)

            # Format chunk with citation metadata
            formatted = self._format_chunk(chunk, i, rc.score)
            chunk_tokens = self._estimate_tokens(formatted)

            # Check token budget
            if self._max_tokens and approx_tokens + chunk_tokens > self._max_tokens:
                logger.info(
                    "Token budget reached at chunk %d/%d (%d tokens)",
                    i,
                    len(ranked_chunks),
                    approx_tokens,
                )
                break

            context_parts.append(formatted)
            approx_tokens += chunk_tokens

        assembled = "\n\n".join(context_parts)

        logger.info(
            "Context assembled: %d chunk(s), ~%d tokens",
            len(context_parts),
            approx_tokens,
        )
        return assembled

    def _format_chunk(self, chunk, index: int, score: float) -> str:
        """Format a single chunk with its citation metadata."""
        meta = chunk.metadata

        # Build location string
        location_parts = []
        if meta.document_name:
            location_parts.append(f"Document: {meta.document_name}")
        if meta.section_title:
            location_parts.append(f"Section: {meta.section_title}")
        if meta.section_id:
            location_parts.append(f"ID: {meta.section_id}")
        if meta.page_number:
            location_parts.append(f"Page: {meta.page_number}")

        location = " | ".join(location_parts) if location_parts else "Unknown location"

        # Build heading path for context
        heading_ctx = ""
        if meta.heading_path:
            heading_ctx = f"\nHeading path: {' > '.join(meta.heading_path)}"

        return (
            f"[Excerpt {index} — {location} | Relevance: {score:.2f}]"
            f"{heading_ctx}\n"
            f"{chunk.text}"
        )

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimation (~4 chars per token for English)."""
        return len(text) // 4


class ContextBuilderStage(PipelineStage):
    """Pipeline stage: assemble context from reranked chunks."""

    def __init__(self, builder: ContextBuilder | None = None) -> None:
        self._builder = builder or ContextBuilder()

    def process(self, data: PipelineData) -> PipelineData:
        if not data.ranked_chunks:
            data.assembled_context = ""
            return data

        data.assembled_context = self._builder.build(data.ranked_chunks)
        return data
