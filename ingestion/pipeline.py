"""
Ingestion pipeline — orchestrates: Parse → Chunk → Embed → Index.

Each step is a PipelineStage that reads from and writes to the shared
PipelineData object. The pipeline is composed using the Pipeline class
from core.pipeline.
"""

from __future__ import annotations

import logging

from config.settings import get_settings
from core.exceptions import IngestionError
from core.models import PipelineData
from core.pipeline import Pipeline, PipelineStage
from ingestion.chunker import ChunkingStrategy, DoclingHybridChunkerStrategy
from ingestion.embedder import EmbeddingServiceFactory, EmbeddingStrategy
from ingestion.indexer import BaseVectorStoreRepository, QdrantVectorStoreRepository
from ingestion.parser import BaseDocumentParser, DocumentParserFactory

logger = logging.getLogger(__name__)


# ── Pipeline Stages ───────────────────────────────────────────────────


class ParseStage(PipelineStage):
    """Stage 1: Parse the raw file into a structured document."""

    def __init__(self, parser: BaseDocumentParser | None = None) -> None:
        self._parser = parser

    def process(self, data: PipelineData) -> PipelineData:
        parser = self._parser or DocumentParserFactory.create(data.file_path)
        data.parsed_document = parser.parse(data.file_path)
        return data


class ChunkStage(PipelineStage):
    """Stage 2: Split the parsed document into chunks with metadata."""

    def __init__(self, strategy: ChunkingStrategy | None = None) -> None:
        self._strategy = strategy or DoclingHybridChunkerStrategy()

    def process(self, data: PipelineData) -> PipelineData:
        if data.parsed_document is None:
            raise IngestionError("No parsed document available for chunking")
        data.chunks = self._strategy.chunk(data.parsed_document)
        return data


class EmbedStage(PipelineStage):
    """Stage 3: Generate dense + sparse embeddings for each chunk."""

    def __init__(self, strategy: EmbeddingStrategy | None = None) -> None:
        self._strategy = strategy or EmbeddingServiceFactory.create()

    def process(self, data: PipelineData) -> PipelineData:
        if not data.chunks:
            raise IngestionError("No chunks available for embedding")

        texts = [c.text for c in data.chunks]
        results = self._strategy.embed_texts(texts)

        for chunk, emb in zip(data.chunks, results):
            chunk.dense_embedding = emb.dense
            chunk.sparse_indices = emb.sparse_indices
            chunk.sparse_values = emb.sparse_values

        return data

    @property
    def strategy(self) -> EmbeddingStrategy:
        """Expose the embedding strategy for dimension queries."""
        return self._strategy


class IndexStage(PipelineStage):
    """Stage 4: Store embedded chunks in the vector database."""

    def __init__(self, repository: BaseVectorStoreRepository | None = None) -> None:
        self._repository = repository or QdrantVectorStoreRepository()

    def process(self, data: PipelineData) -> PipelineData:
        if not data.chunks:
            raise IngestionError("No chunks available for indexing")

        # Ensure collection exists with the correct dimensions
        if data.chunks[0].dense_embedding:
            dim = len(data.chunks[0].dense_embedding)
        else:
            raise IngestionError("Chunks have no embeddings — was EmbedStage run?")

        self._repository.ensure_collection(dense_dim=dim)
        count = self._repository.upsert_chunks(data.chunks)
        data.diagnostics["chunks_indexed"] = count
        return data

    @property
    def repository(self) -> BaseVectorStoreRepository:
        """Expose the repository for retrieval pipeline reuse."""
        return self._repository


# ── Tokenizer Bridge ─────────────────────────────────────────────────


def _build_tokenizer_from_embedder(
    embedder: EmbeddingStrategy,
) -> tuple:
    """
    Load the raw tokenizer matching the embedding model.

    Returns
    -------
    tuple[tokenizer_object | None, str]
        (raw_tokenizer, tokenizer_type) ready for
        DoclingHybridChunkerStrategy.__init__.
    """
    tok_type = embedder.get_tokenizer_type()
    model_id = embedder.get_tokenizer_model_id()

    if tok_type == "huggingface":
        try:
            from transformers import AutoTokenizer

            logger.info("Loading HuggingFace tokenizer for chunker: %s", model_id)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            return tokenizer, tok_type
        except ImportError:
            logger.warning(
                "transformers not installed — chunker will use default tokenizer. "
                "Run: pip install transformers"
            )
            return None, tok_type
        except Exception as exc:
            logger.warning(
                "Failed to load tokenizer '%s': %s — using default", model_id, exc
            )
            return None, tok_type

    elif tok_type == "openai":
        try:
            import tiktoken

            # Strip provider prefix (e.g. "openai/text-embedding-3-small" → "text-embedding-3-small")
            model_name = model_id.split("/")[-1] if "/" in model_id else model_id
            logger.info("Loading tiktoken tokenizer for chunker: %s", model_name)
            tokenizer = tiktoken.encoding_for_model(model_name)
            return tokenizer, tok_type
        except ImportError:
            logger.warning(
                "tiktoken not installed — chunker will use default tokenizer. "
                "Run: pip install tiktoken"
            )
            return None, tok_type
        except Exception as exc:
            logger.warning(
                "Failed to load tiktoken for '%s': %s — using default", model_id, exc
            )
            return None, tok_type

    else:
        logger.warning("Unknown tokenizer type '%s' — using default", tok_type)
        return None, tok_type


# ── Composed Pipeline ─────────────────────────────────────────────────


def create_ingestion_pipeline(
    parser: BaseDocumentParser | None = None,
    chunker: ChunkingStrategy | None = None,
    embedder: EmbeddingStrategy | None = None,
    repository: BaseVectorStoreRepository | None = None,
) -> Pipeline:
    """
    Build the default ingestion pipeline.

    When no explicit chunker is provided, one is created using
    DoclingHybridChunkerStrategy with the tokenizer aligned to
    the embedding model for token-aware splitting.

    Parameters
    ----------
    parser, chunker, embedder, repository
        Optional overrides for each stage. Defaults are created from config.

    Returns
    -------
    Pipeline
        Ready-to-run ingestion pipeline.
    """
    embedder = embedder or EmbeddingServiceFactory.create()

    if chunker is None:
        tokenizer, tok_type = _build_tokenizer_from_embedder(embedder)
        chunker = DoclingHybridChunkerStrategy(
            tokenizer=tokenizer,
            tokenizer_type=tok_type,
        )

    stages = [
        ParseStage(parser),
        ChunkStage(chunker),
        EmbedStage(embedder),
        IndexStage(repository),
    ]
    return Pipeline(stages, name="IngestionPipeline")


def ingest_file(file_path: str) -> PipelineData:
    """
    Convenience function: ingest a single file end-to-end.

    Parameters
    ----------
    file_path : str
        Path to a PDF or DOCX file.

    Returns
    -------
    PipelineData
        Contains parsed document, chunks, and diagnostics.
    """
    pipeline = create_ingestion_pipeline()
    data = PipelineData(file_path=file_path)
    return pipeline.run(data)
