"""
Hybrid retriever — dense + sparse search with RRF fusion on Qdrant.

Executes both dense (semantic) and sparse (SPLADE++) searches,
then merges results using Reciprocal Rank Fusion (RRF) via
Qdrant's native prefetch/fusion mechanism.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from config.settings import get_settings
from core.exceptions import SearchError
from core.models import Chunk, ChunkMetadata, PipelineData
from core.pipeline import PipelineStage
from ingestion.embedder import EmbeddingServiceFactory, EmbeddingStrategy

logger = logging.getLogger(__name__)


class BaseRetriever(ABC):
    """Abstract interface for retrieval."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        document_filter: str | None = None,
    ) -> list[Chunk]:
        """Retrieve relevant chunks for a query."""


class QdrantHybridRetriever(BaseRetriever):
    """
    Hybrid retriever using Qdrant's native prefetch + RRF fusion.

    Retrieves top_k candidates from both dense and sparse indices,
    then fuses using Reciprocal Rank Fusion for the final ranking.
    """

    def __init__(
        self,
        embedding_strategy: EmbeddingStrategy | None = None,
    ) -> None:
        self._settings = get_settings()
        self._embedding = embedding_strategy or EmbeddingServiceFactory.create()
        self._client = None

    def _get_client(self):
        """Return the shared Qdrant client singleton."""
        if self._client is None:
            try:
                from core.qdrant_client_manager import get_qdrant_client

                self._client = get_qdrant_client()
            except RuntimeError as exc:
                raise SearchError(str(exc)) from exc
        return self._client

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        document_filter: str | None = None,
    ) -> list[Chunk]:
        """
        Execute hybrid search: dense + sparse with RRF fusion.

        Parameters
        ----------
        query : str
            The search query.
        top_k : int
            Number of results per method before fusion.
        document_filter : str, optional
            If provided, restrict search to this document_id.

        Returns
        -------
        list[Chunk]
            Ranked chunks after RRF fusion.
        """
        top_k = top_k or self._settings.retrieval_top_k

        try:
            from qdrant_client import models

            client = self._get_client()
            collection_name = self._settings.qdrant_collection_name

            # Generate query embeddings
            query_emb = self._embedding.embed_query(query)

            # Build optional document filter
            qdrant_filter = None
            if document_filter:
                qdrant_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_filter),
                        )
                    ]
                )

            # Build sparse query vector
            sparse_vector = None
            if query_emb.sparse_indices and query_emb.sparse_values:
                sparse_vector = models.SparseVector(
                    indices=query_emb.sparse_indices,
                    values=query_emb.sparse_values,
                )

            # Hybrid search using Qdrant prefetch + RRF
            prefetch = [
                # Dense search
                models.Prefetch(
                    query=query_emb.dense,
                    using="dense",
                    limit=top_k,
                    filter=qdrant_filter,
                ),
            ]

            # Add sparse prefetch if available
            if sparse_vector:
                prefetch.append(
                    models.Prefetch(
                        query=sparse_vector,
                        using="sparse",
                        limit=top_k,
                        filter=qdrant_filter,
                    )
                )

            # Execute hybrid query with RRF fusion
            results = client.query_points(
                collection_name=collection_name,
                prefetch=prefetch,
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=top_k,
                with_payload=True,
            )

            # Convert Qdrant results to Chunk objects
            chunks = []
            for point in results.points:
                payload = point.payload or {}
                chunk = Chunk(
                    id=payload.get("chunk_id", str(point.id)),
                    text=payload.get("text", ""),
                    metadata=ChunkMetadata(
                        document_id=payload.get("document_id", ""),
                        document_name=payload.get("document_name", ""),
                        page_number=payload.get("page_number", 0),
                        section_id=payload.get("section_id", ""),
                        section_title=payload.get("section_title", ""),
                        chunk_level=payload.get("chunk_level", 0),
                        parent_chunk_id=payload.get("parent_chunk_id"),
                        heading_path=payload.get("heading_path", []),
                    ),
                )
                chunks.append(chunk)

            logger.info(
                "Hybrid search for '%s' returned %d chunk(s)",
                query[:60],
                len(chunks),
            )
            return chunks

        except SearchError:
            raise
        except Exception as exc:
            raise SearchError(f"Hybrid search failed: {exc}") from exc


class HybridRetrievalStage(PipelineStage):
    """
    Pipeline stage: retrieve chunks for all sub-queries and deduplicate.
    """

    def __init__(self, retriever: BaseRetriever | None = None) -> None:
        self._retriever = retriever or QdrantHybridRetriever()

    def process(self, data: PipelineData) -> PipelineData:
        queries = data.sub_queries or [data.query]
        all_chunks: dict[str, Chunk] = {}

        for query in queries:
            chunks = self._retriever.retrieve(query)
            for chunk in chunks:
                # Deduplicate by chunk ID
                if chunk.id not in all_chunks:
                    all_chunks[chunk.id] = chunk

        data.retrieved_chunks = list(all_chunks.values())
        logger.info(
            "Retrieved %d unique chunk(s) across %d sub-query(ies)",
            len(data.retrieved_chunks),
            len(queries),
        )
        return data
