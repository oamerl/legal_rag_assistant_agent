"""
Query router — classifies queries and selects the retrieval strategy.

Uses a lightweight LLM call (OpenRouter, small model) to classify
queries into types, then routes to the appropriate query transformer.
"""

from __future__ import annotations

import logging
from enum import Enum

from langchain_core.messages import HumanMessage, SystemMessage

from core.models import PipelineData
from core.pipeline import PipelineStage
from generation.llm_client import create_llm_client

logger = logging.getLogger(__name__)


class QueryType(str, Enum):
    """Classification of user queries for routing."""

    SIMPLE = "SIMPLE"          # Direct factual question → direct retrieval
    COMPLEX = "COMPLEX"        # Multi-part or comparison → query decomposition
    SPECIFIC = "SPECIFIC"      # References a specific section → metadata filter
    AMBIGUOUS = "AMBIGUOUS"    # Vague or broad → HyDE + expansion


# Classification prompt for the LLM
ROUTER_SYSTEM_PROMPT = """You are a query classifier for a legal document assistant.
Classify the user's query into exactly ONE of these categories:

- SIMPLE: Direct factual question about a specific topic (e.g., "What is the notice period?")
- COMPLEX: Multi-part question or comparison (e.g., "Compare termination and renewal clauses")
- SPECIFIC: References a specific section, clause, or page (e.g., "What does Section 4.2 say?")
- AMBIGUOUS: Vague or broad question that needs expansion (e.g., "Tell me about liability")

Respond with ONLY the category name, nothing else."""

# Valid query types for response validation
_VALID_TYPES = {qt.value for qt in QueryType}


class QueryRouterStage(PipelineStage):
    """
    Pipeline stage that classifies the query type using an LLM.

    Uses the small model (e.g. gpt-4o-mini) for cost-efficient,
    intelligent query classification. Falls back to SIMPLE on errors.
    """

    def __init__(self) -> None:
        self._llm = None  # Lazy-initialized on first use

    def _get_llm(self):
        """Lazily create the LLM client (small model for routing)."""
        if self._llm is None:
            self._llm = create_llm_client(
                use_small_model=True,
                max_tokens=20,  # Only need a single word response
            )
        return self._llm

    def process(self, data: PipelineData) -> PipelineData:
        query = data.query.strip()
        if not query:
            data.query_type = QueryType.SIMPLE.value
            return data

        query_type = self._classify(query)
        data.query_type = query_type.value

        logger.info("Query classified as: %s — '%s'", query_type.value, query[:80])
        return data

    def _classify(self, query: str) -> QueryType:
        """
        LLM-based query classification.

        Sends the query to the small model with the classification prompt
        and parses the response. Falls back to SIMPLE on any error.
        """
        try:
            llm = self._get_llm()
            messages = [
                SystemMessage(content=ROUTER_SYSTEM_PROMPT),
                HumanMessage(content=query),
            ]

            response = llm.invoke(messages)
            raw = response.content.strip().upper()

            # Parse the response — accept the label even if the LLM
            # adds minor surrounding text (e.g. "COMPLEX.")
            for qt in QueryType:
                if qt.value in raw:
                    return qt

            logger.warning(
                "LLM returned unexpected classification '%s', defaulting to SIMPLE",
                raw,
            )
            return QueryType.SIMPLE

        except Exception:
            logger.exception("LLM classification failed, defaulting to SIMPLE")
            return QueryType.SIMPLE
