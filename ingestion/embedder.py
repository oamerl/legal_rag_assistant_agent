"""
Embedding module — generates dense and sparse embeddings for chunks.

Design Pattern: Strategy Pattern
    The EmbeddingStrategy ABC allows swapping between local (FastEmbed),
    Voyage AI, or OpenRouter embedding providers via configuration.

All local models run on CPU by default (ONNX Runtime).
Set `device="cuda"` in settings to use GPU acceleration.

OpenRouter Strategy:
    Uses the OpenAI-compatible /v1/embeddings endpoint via OpenRouter for
    dense embeddings (e.g. text-embedding-3-small) while still running
    SPLADE++ locally for sparse embeddings.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from config.settings import get_settings
from core.exceptions import EmbeddingError
from core.models import Chunk

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Container for dense + sparse embedding outputs."""

    dense: list[float]
    sparse_indices: list[int]
    sparse_values: list[float]


class EmbeddingStrategy(ABC):
    """Abstract base for embedding providers."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[EmbeddingResult]:
        """Generate embeddings for a batch of texts."""

    @abstractmethod
    def embed_query(self, query: str) -> EmbeddingResult:
        """Generate embeddings for a single query."""

    @abstractmethod
    def get_dense_dimension(self) -> int:
        """Return the dimensionality of the dense embedding model."""

    @abstractmethod
    def get_tokenizer_model_id(self) -> str:
        """Return the model identifier used to load the matching tokenizer."""

    @abstractmethod
    def get_tokenizer_type(self) -> str:
        """Return the tokenizer backend type: 'huggingface' or 'openai'."""


class LocalFastEmbedStrategy(EmbeddingStrategy):
    """
    Local embedding using FastEmbed (ONNX Runtime).

    Generates both dense (bge-large-en-v1.5) and sparse (SPLADE++)
    embeddings on CPU. GPU is supported if CUDA is available and
    configured.
    """

    def __init__(
        self,
        dense_model: str | None = None, # names of the models
        sparse_model: str | None = None,
    ) -> None:
        settings = get_settings()
        self._dense_model_name = dense_model or settings.dense_model_name
        self._sparse_model_name = sparse_model or settings.sparse_model_name
        self._dense_encoder = None
        self._sparse_encoder = None
        self._dense_dim: int | None = None

    def _init_dense(self):
        """Lazy-load the dense embedding model."""
        if self._dense_encoder is None:
            try:
                from fastembed import TextEmbedding

                logger.info("Loading dense model: %s", self._dense_model_name)
                self._dense_encoder = TextEmbedding(model_name=self._dense_model_name,)
                logger.info("Dense model loaded")
            except ImportError as exc:
                raise EmbeddingError(
                    "fastembed not installed. Run: pip install fastembed"
                ) from exc
        return self._dense_encoder

    def _init_sparse(self):
        """Lazy-load the sparse embedding model."""
        if self._sparse_encoder is None:
            try:
                from fastembed import SparseTextEmbedding

                logger.info("Loading sparse model: %s", self._sparse_model_name)
                self._sparse_encoder = SparseTextEmbedding(model_name=self._sparse_model_name,)
                logger.info("Sparse model loaded")
            except ImportError as exc:
                raise EmbeddingError(
                    "fastembed not installed. Run: pip install fastembed"
                ) from exc
        return self._sparse_encoder

    def embed_texts(self, texts: list[str]) -> list[EmbeddingResult]:
        """
        Embed a batch of texts with both dense and sparse models.

        Parameters
        ----------
        texts : list[str]
            The texts to embed.

        Returns
        -------
        list[EmbeddingResult]
            One result per input text, each containing dense + sparse vectors.
        """
        if not texts:
            return []

        try:
            dense_encoder = self._init_dense()
            sparse_encoder = self._init_sparse()

            logger.info("Embedding %d text(s)", len(texts))

            # Generate dense embeddings
            dense_embeddings = list(dense_encoder.embed(texts))

            # Generate sparse embeddings
            sparse_embeddings = list(sparse_encoder.embed(texts))

            # Capture dense dimension from first result
            if dense_embeddings and self._dense_dim is None:
                self._dense_dim = len(dense_embeddings[0])
                logger.info("Dense embedding dimension: %d", self._dense_dim)

            results = []
            for dense_vec, sparse_vec in zip(dense_embeddings, sparse_embeddings):
                results.append(
                    EmbeddingResult(
                        dense=dense_vec.tolist() if hasattr(dense_vec, "tolist") else list(dense_vec),
                        sparse_indices=sparse_vec.indices.tolist() if hasattr(sparse_vec.indices, "tolist") else list(sparse_vec.indices),
                        sparse_values=sparse_vec.values.tolist() if hasattr(sparse_vec.values, "tolist") else list(sparse_vec.values),
                    )
                )

            logger.info("Embedded %d text(s) successfully", len(results))
            return results

        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"Embedding failed: {exc}") from exc

    def embed_query(self, query: str) -> EmbeddingResult:
        """Embed a single query string."""
        results = self.embed_texts([query])
        if not results:
            raise EmbeddingError("No embedding returned for query")
        return results[0]

    def get_dense_dimension(self) -> int:
        """Return dense embedding dimensionality (initialises model if needed)."""
        if self._dense_dim is None:
            # Embed a dummy text to discover the dimension
            self._init_dense()
            dummy = list(self._init_dense().embed(["dimension probe"]))
            self._dense_dim = len(dummy[0])
        return self._dense_dim

    def get_tokenizer_model_id(self) -> str:
        """Return the HuggingFace model name for tokenizer loading."""
        return self._dense_model_name

    def get_tokenizer_type(self) -> str:
        """Local models use HuggingFace tokenizers."""
        return "huggingface"


class VoyageEmbeddingStrategy(EmbeddingStrategy):
    """
    Placeholder for Voyage AI embeddings (voyage-law-2).

    Enable by setting `embedding_provider=voyage` and providing
    a `voyage_api_key` in the .env file.
    """

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.voyage_api_key:
            raise EmbeddingError(
                "Voyage API key not set. Add VOYAGE_API_KEY to your .env file."
            )
        self._voyage_model_name = settings.voyage_model_name
        logger.info("VoyageEmbeddingStrategy initialised (model: %s)", self._voyage_model_name)
        # TODO: Implement Voyage AI client integration
        raise NotImplementedError("Voyage AI embedding is not yet implemented")

    def embed_texts(self, texts: list[str]) -> list[EmbeddingResult]:
        raise NotImplementedError

    def embed_query(self, query: str) -> EmbeddingResult:
        raise NotImplementedError

    def get_dense_dimension(self) -> int:
        raise NotImplementedError

    def get_tokenizer_model_id(self) -> str:
        return self._voyage_model_name

    def get_tokenizer_type(self) -> str:
        return "huggingface"


class OpenRouterEmbeddingStrategy(EmbeddingStrategy):
    """
    Remote dense embeddings via OpenRouter + local SPLADE++ for sparse.

    Uses the OpenAI-compatible ``/v1/embeddings`` endpoint exposed by
    OpenRouter so you can use models like ``openai/text-embedding-3-small``
    or ``openai/text-embedding-3-large`` without downloading anything
    locally.  Sparse embeddings still run through FastEmbed (SPLADE++)
    because no remote sparse model is available.

    Enable by setting ``EMBEDDING_PROVIDER=openrouter`` in your ``.env``
    file.  The existing ``OPENROUTER_API_KEY`` and ``OPENROUTER_BASE_URL``
    settings are reused.
    """

    def __init__(self) -> None:
        settings = get_settings()

        if not settings.openrouter_api_key:
            raise EmbeddingError(
                "OpenRouter API key not set. Add OPENROUTER_API_KEY to your .env file."
            )

        self._model = settings.openrouter_embedding_model
        self._dimensions = settings.openrouter_embedding_dimensions
        self._base_url = settings.openrouter_base_url
        self._api_key = settings.openrouter_api_key

        # Lazy-initialised clients
        self._openai_client = None
        self._sparse_encoder = None
        self._dense_dim: int | None = None

        # Local sparse model name (SPLADE++)
        self._sparse_model_name = settings.sparse_model_name

        logger.info(
            "OpenRouterEmbeddingStrategy initialised (model: %s, dimensions: %s)",
            self._model,
            self._dimensions or "default",
        )

    # ── Private helpers ───────────────────────────────────────────────

    def _get_openai_client(self):
        """Lazy-load the OpenAI client pointed at OpenRouter."""
        if self._openai_client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise EmbeddingError(
                    "openai package not installed. Run: pip install openai"
                ) from exc

            self._openai_client = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
            )
            logger.info("OpenAI client created for OpenRouter embeddings")
        return self._openai_client

    def _init_sparse(self):
        """Lazy-load the local SPLADE++ sparse model."""
        if self._sparse_encoder is None:
            try:
                from fastembed import SparseTextEmbedding
            except ImportError as exc:
                raise EmbeddingError(
                    "fastembed not installed. Run: pip install fastembed"
                ) from exc

            logger.info("Loading sparse model: %s", self._sparse_model_name)
            self._sparse_encoder = SparseTextEmbedding(
                model_name=self._sparse_model_name,
            )
            logger.info("Sparse model loaded")
        return self._sparse_encoder

    def _embed_dense_batch(self, texts: list[str]) -> list[list[float]]:
        """Call the OpenRouter /v1/embeddings endpoint for dense vectors."""
        client = self._get_openai_client()

        kwargs: dict = {
            "model": self._model,
            "input": texts,
        }
        if self._dimensions is not None:
            kwargs["dimensions"] = self._dimensions

        response = client.embeddings.create(**kwargs)

        # Sort by index to guarantee ordering matches input
        sorted_data = sorted(response.data, key=lambda d: d.index)
        embeddings = [item.embedding for item in sorted_data]

        # Capture dimension from the first result
        if embeddings and self._dense_dim is None:
            self._dense_dim = len(embeddings[0])
            logger.info("Dense embedding dimension (OpenRouter): %d", self._dense_dim)

        return embeddings

    # ── Public interface ──────────────────────────────────────────────

    def embed_texts(self, texts: list[str]) -> list[EmbeddingResult]:
        """
        Embed a batch of texts using OpenRouter (dense) + local SPLADE++ (sparse).

        Parameters
        ----------
        texts : list[str]
            The texts to embed.

        Returns
        -------
        list[EmbeddingResult]
            One result per input text, each containing dense + sparse vectors.
        """
        if not texts:
            return []

        try:
            logger.info("Embedding %d text(s) via OpenRouter + local SPLADE++", len(texts))

            # Dense embeddings from OpenRouter
            dense_embeddings = self._embed_dense_batch(texts)

            # Sparse embeddings from local SPLADE++
            sparse_encoder = self._init_sparse()
            sparse_embeddings = list(sparse_encoder.embed(texts))

            results = []
            for dense_vec, sparse_vec in zip(dense_embeddings, sparse_embeddings):
                results.append(
                    EmbeddingResult(
                        dense=list(dense_vec),
                        sparse_indices=(
                            sparse_vec.indices.tolist()
                            if hasattr(sparse_vec.indices, "tolist")
                            else list(sparse_vec.indices)
                        ),
                        sparse_values=(
                            sparse_vec.values.tolist()
                            if hasattr(sparse_vec.values, "tolist")
                            else list(sparse_vec.values)
                        ),
                    )
                )

            logger.info("Embedded %d text(s) successfully", len(results))
            return results

        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"OpenRouter embedding failed: {exc}") from exc

    def embed_query(self, query: str) -> EmbeddingResult:
        """Embed a single query string."""
        results = self.embed_texts([query])
        if not results:
            raise EmbeddingError("No embedding returned for query")
        return results[0]

    def get_dense_dimension(self) -> int:
        """Return dense embedding dimensionality (calls API if needed)."""
        if self._dense_dim is None:
            self._embed_dense_batch(["dimension probe"])
        return self._dense_dim

    def get_tokenizer_model_id(self) -> str:
        """Return the OpenRouter model name for tiktoken loading."""
        return self._model

    def get_tokenizer_type(self) -> str:
        """OpenRouter models use OpenAI-compatible (tiktoken) tokenizers."""
        return "openai"


# ── Factory ───────────────────────────────────────────────────────────


class EmbeddingServiceFactory:
    """Factory to create the appropriate embedding strategy from config."""

    _strategies: dict[str, type[EmbeddingStrategy]] = {
        "local": LocalFastEmbedStrategy,
        "voyage": VoyageEmbeddingStrategy,
        "openrouter": OpenRouterEmbeddingStrategy,
    }

    @classmethod
    def create(cls, provider: str | None = None) -> EmbeddingStrategy:
        """
        Create an embedding strategy based on the provider name.

        Parameters
        ----------
        provider : str, optional
            One of "local", "voyage", "openrouter".
            Defaults to settings.embedding_provider.
        """
        settings = get_settings()
        provider = provider or settings.embedding_provider

        strategy_class = cls._strategies.get(provider)
        if strategy_class is None:
            raise EmbeddingError(
                f"Unknown embedding provider '{provider}'. "
                f"Available: {list(cls._strategies.keys())}"
            )

        logger.info("Creating embedding strategy: %s", provider)
        return strategy_class()
