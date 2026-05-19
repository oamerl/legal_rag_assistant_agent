"""
Cross-encoder reranker — re-scores retrieved chunks for precision.

Uses BAAI/bge-reranker-large to compute query-document relevance
scores. Runs on CPU by default; GPU supported via config.

Applies a relevance threshold — if no chunks pass, the query
is flagged as "not found in documents".
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import requests

from config.settings import get_settings
from core.exceptions import NoRelevantChunksError, RerankingError
from core.models import Chunk, PipelineData, RankedChunk
from core.pipeline import PipelineStage

logger = logging.getLogger(__name__)


class BaseReranker(ABC):
    """Abstract interface for reranking."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        chunks: list[Chunk],
        top_n: int = 6,
    ) -> list[RankedChunk]:
        """Re-score and re-order chunks by relevance to the query."""


class CrossEncoderReranker(BaseReranker):
    """
    Cross-encoder reranker using sentence-transformers.

    Processes (query, chunk) pairs through a cross-encoder model
    for deep relevance scoring. Much more accurate than bi-encoder
    similarity but slower — hence applied only to pre-filtered
    candidates (top 20-30 from retrieval).
    """

    def __init__(
        self,
        model_name: str | None = None,
    ) -> None:
        settings = get_settings()
        self._model_name = model_name or settings.reranker_model_name
        self._device = settings.device
        self._model = None

    def _get_model(self):
        """Lazy-load the cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                logger.info(
                    "Loading reranker: %s (device=%s)",
                    self._model_name,
                    self._device,
                )
                self._model = CrossEncoder(
                    self._model_name,
                    device=self._device,
                )
                logger.info("Reranker model loaded")
            except ImportError as exc:
                raise RerankingError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                ) from exc
        return self._model

    def rerank(
        self,
        query: str,
        chunks: list[Chunk],
        top_n: int | None = None,
    ) -> list[RankedChunk]:
        """
        Re-score chunks using the cross-encoder.

        Parameters
        ----------
        query : str
            The user's query.
        chunks : list[Chunk]
            Pre-filtered chunks from hybrid retrieval.
        top_n : int
            Number of top results to return after reranking.

        Returns
        -------
        list[RankedChunk]
            Chunks sorted by relevance score (descending).
        """
        settings = get_settings()
        top_n = top_n or settings.reranker_top_n

        if not chunks:
            return []

        try:
            model = self._get_model()

            # Create (query, chunk_text) pairs for the cross-encoder
            pairs = [(query, chunk.text) for chunk in chunks]

            logger.info("Reranking %d chunk(s)", len(pairs))
            scores = model.predict(pairs)

            # Pair chunks with their scores and sort descending
            ranked = sorted(
                zip(chunks, scores),
                key=lambda x: float(x[1]),
                reverse=True,
            )

            # Take top N
            results = [
                RankedChunk(chunk=chunk, score=float(score))
                for chunk, score in ranked[:top_n]
            ]

            logger.info(
                "Reranked: top score=%.3f, bottom score=%.3f (%d results)",
                results[0].score if results else 0,
                results[-1].score if results else 0,
                len(results),
            )

            return results

        except RerankingError:
            raise
        except Exception as exc:
            raise RerankingError(f"Reranking failed: {exc}") from exc


class OpenRouterReranker(BaseReranker):
    """
    Reranker using OpenRouter's rerank API (e.g., for Cohere models).
    Offloads cross-encoder computation to a remote API.
    """

    def __init__(
        self,
        model_name: str | None = None,
    ) -> None:
        settings = get_settings()
        self._model_name = model_name or settings.openrouter_reranker_model
        self._api_key = settings.openrouter_api_key
        # Ensure base URL is clean (OpenRouter base is usually https://openrouter.ai/api/v1)
        self._base_url = settings.openrouter_base_url.rstrip('/')

    def rerank(
        self,
        query: str,
        chunks: list[Chunk],
        top_n: int | None = None,
    ) -> list[RankedChunk]:
        settings = get_settings()
        top_n = top_n or settings.reranker_top_n

        if not chunks:
            return []

        if not self._api_key:
            raise RerankingError("OpenRouter API key is missing. Cannot use OpenRouterReranker.")

        url = f"{self._base_url}/rerank"

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json"
        }

        # Prepare request payload
        documents = [chunk.text for chunk in chunks]
        payload = {
            "model": self._model_name,
            "query": query,
            "documents": documents,
            "top_n": top_n
        }

        try:
            logger.info("Reranking %d chunk(s) via OpenRouter (%s)", len(chunks), self._model_name)
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            
            if response.status_code != 200:
                logger.error("OpenRouter rerank failed: %s", response.text)
                response.raise_for_status()

            data = response.json()
            results = data.get("results", [])
            
            # OpenRouter returns a list of results with indices and scores
            # Map back to RankedChunk
            ranked_results = []
            for item in results:
                idx = item.get("index")
                score = item.get("relevance_score")
                if idx is not None and score is not None and 0 <= idx < len(chunks):
                    ranked_results.append(RankedChunk(chunk=chunks[idx], score=float(score)))
            
            # Ensure they are sorted by score descending
            ranked_results.sort(key=lambda x: x.score, reverse=True)

            logger.info(
                "Reranked via OpenRouter: top score=%.3f, bottom score=%.3f (%d results)",
                ranked_results[0].score if ranked_results else 0,
                ranked_results[-1].score if ranked_results else 0,
                len(ranked_results),
            )

            return ranked_results

        except requests.RequestException as exc:
            raise RerankingError(f"OpenRouter API request failed: {exc}") from exc
        except Exception as exc:
            raise RerankingError(f"OpenRouter reranking failed: {exc}") from exc


class RerankerFactory:
    """Factory to create the appropriate reranker based on configuration."""

    _strategies: dict[str, type[BaseReranker]] = {
        "local": CrossEncoderReranker,
        "openrouter": OpenRouterReranker,
    }

    @classmethod
    def create(cls, reranker_provider: str | None = None) -> BaseReranker:
        """
        Create an reranker strategy based on the provider name.

        Parameters
        ----------
        provider : str, optional
            One of "local", "openrouter".
            Defaults to settings.reranker_provider.
        """
        settings = get_settings()
        provider = reranker_provider or settings.reranker_provider

        strategy_class = cls._strategies.get(provider)
        if strategy_class is None:
            raise RerankingError(
                f"Unknown reranker provider: {provider}"
                f"Available: {list(cls._strategies.keys())}"
                )
        logger.info("Creating reranker strategy: %s", provider)
        return strategy_class()


class RerankingStage(PipelineStage):
    """
    Pipeline stage: rerank retrieved chunks and apply relevance threshold.

    If no chunks pass the threshold, sets `retrieval_is_relevant = False`
    so downstream stages can handle "not found" gracefully.
    """

    def __init__(self, reranker: BaseReranker | None = None) -> None:
        self._reranker = reranker or RerankerFactory.create()

    def process(self, data: PipelineData) -> PipelineData:
        settings = get_settings()

        if not data.retrieved_chunks:
            data.retrieval_is_relevant = False
            data.ranked_chunks = []
            logger.warning("No chunks to rerank")
            return data

        ranked = self._reranker.rerank(
            query=data.query,
            chunks=data.retrieved_chunks,
        )

        # Apply relevance threshold
        threshold = settings.reranker_relevance_threshold
        filtered = [r for r in ranked if r.score >= threshold]

        if not filtered:
            data.retrieval_is_relevant = False
            data.ranked_chunks = ranked[:3]  # Keep top 3 anyway for diagnostics
            logger.warning(
                "No chunks above threshold %.2f — highest score was %.3f",
                threshold,
                ranked[0].score if ranked else 0,
            )
        else:
            data.retrieval_is_relevant = True
            data.ranked_chunks = filtered

        logger.info(
            "After reranking: %d/%d chunk(s) above threshold (%.2f)",
            len(filtered),
            len(ranked),
            threshold,
        )
        return data
