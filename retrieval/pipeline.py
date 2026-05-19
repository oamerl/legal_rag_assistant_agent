"""
Retrieval pipeline — orchestrates: Route → Transform → Retrieve → Rerank → Build Context.
"""

from __future__ import annotations

import logging

from core.models import PipelineData
from core.pipeline import Pipeline, PipelineStage
from retrieval.context_builder import ContextBuilderStage
from retrieval.query_router import QueryRouterStage
from retrieval.query_transformer import QueryTransformStage
from retrieval.reranker import RerankingStage
from retrieval.retriever import HybridRetrievalStage

logger = logging.getLogger(__name__)


def create_retrieval_pipeline() -> Pipeline:
    """
    Build the default retrieval pipeline.

    Stages:
        1. QueryRouterStage         — classify query type
        2. QueryTransformStage      — rewrite / decompose / expand
        3. HybridRetrievalStage     — dense + sparse search + RRF
        4. RerankingStage           — cross-encoder reranking + threshold filtering
        5. ContextBuilderStage      — assemble final context with citations

    Returns
    -------
    Pipeline
        Ready-to-run retrieval pipeline.
    """
    stages: list[PipelineStage] = [
        QueryRouterStage(),
        QueryTransformStage(),
        HybridRetrievalStage(),
        RerankingStage(),
        ContextBuilderStage(),
    ]
    return Pipeline(stages, name="RetrievalPipeline")


def retrieve(query: str, conversation_id: str = "") -> PipelineData:
    """
    Convenience function: run the retrieval pipeline for a query.

    Parameters
    ----------
    query : str
        The user's natural language question.
    conversation_id : str, optional
        For session tracking.

    Returns
    -------
    PipelineData
        Contains ranked_chunks, assembled_context, and relevance flag.
    """
    pipeline = create_retrieval_pipeline()
    data = PipelineData(query=query, conversation_id=conversation_id)
    return pipeline.run(data)
