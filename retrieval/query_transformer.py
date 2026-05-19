"""
Query transformer — rewrites, decomposes, or expands queries.

Design Pattern: Strategy Pattern
    Different transformation strategies are selected by the QueryRouter
    based on query type. Each strategy implements QueryTransformStrategy.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from config.settings import get_settings
from core.exceptions import QueryTransformError
from core.models import PipelineData
from core.pipeline import PipelineStage
from retrieval.query_router import QueryType

logger = logging.getLogger(__name__)


class QueryTransformStrategy(ABC):
    """Abstract base for query transformation strategies."""

    @abstractmethod
    def transform(self, query: str) -> list[str]:
        """
        Transform a query into one or more sub-queries.

        Returns
        -------
        list[str]
            One or more queries to use for retrieval.
        """


class DirectQueryStrategy(QueryTransformStrategy):
    """Pass-through — returns the original query unchanged."""

    def transform(self, query: str) -> list[str]:
        return [query]


class DecompositionStrategy(QueryTransformStrategy):
    """
    Decompose a complex query into simpler sub-queries using the LLM.

    Used for COMPLEX queries (comparisons, multi-part questions).
    """

    def __init__(self, llm_client=None) -> None:
        self._llm = llm_client

    def _get_llm(self):
        if self._llm is None:
            from generation.llm_client import create_llm_client
            self._llm = create_llm_client(use_small_model=True)
        return self._llm

    def transform(self, query: str) -> list[str]:
        try:
            llm = self._get_llm()
            prompt = (
                "Break down this legal document question into 2-4 simpler, "
                "self-contained sub-questions. Return each sub-question on "
                "a separate line, with no numbering or bullets.\n\n"
                f"Question: {query}"
            )
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)

            sub_queries = [
                line.strip()
                for line in content.strip().split("\n")
                if line.strip()
            ]

            # Always include the original query
            if query not in sub_queries:
                sub_queries.insert(0, query)

            logger.info("Decomposed query into %d sub-queries", len(sub_queries))
            return sub_queries

        except Exception as exc:
            logger.warning("Decomposition failed, falling back to original: %s", exc)
            return [query]


class HyDEStrategy(QueryTransformStrategy):
    """
    Hypothetical Document Embeddings — generates a hypothetical answer
    and uses both it and the original query for retrieval.

    Used for AMBIGUOUS queries where answer-space embeddings align
    better with chunk embeddings than question-space embeddings.
    """

    def __init__(self, llm_client=None) -> None:
        self._llm = llm_client

    def _get_llm(self):
        if self._llm is None:
            from generation.llm_client import create_llm_client
            self._llm = create_llm_client(use_small_model=True, 
                                          temperature=0.1) # we need to have some temperature for hyDE strategy to generate some creative hypothetical documents.
        return self._llm

    def transform(self, query: str) -> list[str]:
        try:
            llm = self._get_llm()
            prompt = (
                "You are a legal document expert. Write a short paragraph "
                "(2-3 sentences) that would be a typical clause or section "
                "in a legal agreement that answers this question. "
                "Write it as if it were text from an actual legal document.\n\n"
                f"Question: {query}"
            )
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)

            # Return both original query AND hypothetical document
            # to mitigate HyDE drift
            queries = [query, content.strip()]
            logger.info("HyDE generated hypothetical document (%d chars)", len(content))
            return queries

        except Exception as exc:
            logger.warning("HyDE failed, falling back to original: %s", exc)
            return [query]


class MetadataFilterStrategy(QueryTransformStrategy):
    """
    Extract section/page references from the query for metadata filtering.

    Used for SPECIFIC queries that reference known document structure.
    Returns the cleaned query for semantic search.
    """

    def transform(self, query: str) -> list[str]:
        # The actual metadata filter will be applied in the retriever
        # Here we just return the original query
        return [query]


# ── Pipeline Stage ────────────────────────────────────────────────────


class QueryTransformStage(PipelineStage):
    """
    Pipeline stage that selects and applies the appropriate
    query transformation strategy based on the query type.
    """

    def __init__(self, strategies: dict[str, QueryTransformStrategy] | None = None):
        self._strategies = strategies or {
            QueryType.SIMPLE.value: DirectQueryStrategy(),
            QueryType.COMPLEX.value: DecompositionStrategy(),
            QueryType.SPECIFIC.value: MetadataFilterStrategy(),
            QueryType.AMBIGUOUS.value: HyDEStrategy(),
        }

    def process(self, data: PipelineData) -> PipelineData:
        query_type = data.query_type or QueryType.SIMPLE.value
        strategy = self._strategies.get(query_type, DirectQueryStrategy())

        logger.info("Applying %s for query type %s", strategy.__class__.__name__, query_type)

        try:
            sub_queries = strategy.transform(data.query)
            data.sub_queries = sub_queries
            logger.info("Query transformed into %d sub-query(ies)", len(sub_queries))
        except Exception as exc:
            raise QueryTransformError(f"Query transformation failed: {exc}") from exc

        return data
