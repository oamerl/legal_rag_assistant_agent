"""
Chunking module — structure-aware chunking using Docling HybridChunker.

Design Pattern: Strategy Pattern
    The ChunkingStrategy ABC allows swapping chunking implementations
    via configuration without changing the pipeline.

The default DoclingHybridChunkerStrategy uses Docling's HybridChunker
which respects document structure (headings, paragraphs, tables) and
enforces token limits — ideal for legal documents.

When a tokenizer is provided (aligned to the embedding model), the
HybridChunker performs token-aware splitting using the exact same
tokenization as the downstream model, and max_tokens is derived
automatically from the tokenizer's context window.
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from typing import Any

from config.settings import get_settings
from core.exceptions import ChunkingError
from core.models import Chunk, ChunkMetadata, ParsedDocument

logger = logging.getLogger(__name__)


class ChunkingStrategy(ABC):
    """Abstract base for chunking strategies."""

    @abstractmethod
    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        """Split a parsed document into chunks with metadata."""


class DoclingHybridChunkerStrategy(ChunkingStrategy):
    """
    Structure-aware chunking using Docling's HybridChunker.

    Combines structural segmentation (respects headings, paragraphs,
    tables) with token-based size limits. Preserves parent-child
    relationships in metadata for parent document retrieval.

    When a tokenizer is provided, the HybridChunker uses it for
    token-aware splitting aligned to the embedding model.  The
    ``contextualize()`` method enriches each chunk with parent
    headings, captions, and table headers for improved retrieval.
    """

    def __init__(
        self,
        tokenizer: Any | None = None,
        tokenizer_type: str = "huggingface",
        max_tokens: int | None = None,
        merge_peers: bool | None = None,
    ) -> None:
        settings = get_settings()
        self._max_tokens = max_tokens or settings.chunk_max_tokens
        self._merge_peers = merge_peers if merge_peers is not None else settings.chunk_merge_peers
        self._raw_tokenizer = tokenizer
        self._tokenizer_type = tokenizer_type

    def _build_docling_tokenizer(self) -> Any | None:
        """Wrap the raw tokenizer in the appropriate Docling tokenizer wrapper."""
        if self._raw_tokenizer is None:
            return None

        if self._tokenizer_type == "huggingface":
            try:
                from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
                return HuggingFaceTokenizer(tokenizer=self._raw_tokenizer)
            except ImportError:
                logger.warning(
                    "docling-core[chunking] not installed — "
                    "falling back to default tokenizer. "
                    "Run: pip install 'docling-core[chunking]'"
                )
                return None

        elif self._tokenizer_type == "openai":
            try:
                from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer

                # OpenAITokenizer requires max_tokens (the tokenizer's context window).
                # Default to 8191 (standard for OpenAI embedding models like
                # text-embedding-3-small/large). The HybridChunker's own max_tokens
                # param can further limit chunk size below this ceiling.
                tok_max = self._max_tokens or 8191
                return OpenAITokenizer(
                    tokenizer=self._raw_tokenizer,
                    max_tokens=tok_max,
                )
            except ImportError:
                logger.warning(
                    "docling-core[chunking-openai] not installed — "
                    "falling back to default tokenizer. "
                    "Run: pip install 'docling-core[chunking-openai]'"
                )
                return None

        else:
            logger.warning("Unknown tokenizer_type '%s' — using default", self._tokenizer_type)
            return None

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        """
        Chunk a ParsedDocument using Docling's HybridChunker.

        Parameters
        ----------
        document : ParsedDocument
            Must have a valid ``docling_document`` attribute.

        Returns
        -------
        list[Chunk]
            Chunks enriched with structural metadata and contextualized
            text for optimal embedding quality.

        Raises
        ------
        ChunkingError
            If the Docling document is missing or chunking fails.
        """
        if document.docling_document is None:
            raise ChunkingError("ParsedDocument has no docling_document — was it parsed?")

        logger.info(
            "Chunking '%s' with HybridChunker (max_tokens=%s, merge_peers=%s, tokenizer=%s)",
            document.doc_name,
            self._max_tokens or "auto",
            self._merge_peers,
            self._tokenizer_type if self._raw_tokenizer else "default",
        )

        try:
            from docling.chunking import HybridChunker

            # Build the Docling-wrapped tokenizer
            docling_tokenizer = self._build_docling_tokenizer()

            # Assemble HybridChunker kwargs
            kwargs: dict[str, Any] = {"merge_peers": self._merge_peers}
            if docling_tokenizer is not None:
                kwargs["tokenizer"] = docling_tokenizer
            if self._max_tokens is not None:
                kwargs["max_tokens"] = self._max_tokens

            chunker = HybridChunker(**kwargs)

            docling_chunks = list(chunker.chunk(document.docling_document))
            chunks: list[Chunk] = []

            for i, dc in enumerate(docling_chunks):
                # Get raw text from the chunk
                raw_text = dc.text if hasattr(dc, "text") else str(dc)

                if not raw_text or not raw_text.strip():
                    continue

                # Use contextualize() for embedding-quality text
                try:
                    text = chunker.contextualize(chunk=dc)
                except Exception:
                    # Fall back to raw text if contextualize fails
                    logger.debug(
                        "contextualize() failed for chunk %d, using raw text",
                        i,
                    )
                    text = raw_text

                if not text or not text.strip():
                    text = raw_text

                # Build metadata from Docling chunk
                meta = self._extract_metadata(dc, document, i)
                # Store the raw text for clean citations
                meta.raw_text = raw_text

                chunk = Chunk(
                    id=self._generate_chunk_id(document.doc_id, i, text),
                    text=text,
                    metadata=meta,
                )
                chunks.append(chunk)

            logger.info(
                "Chunked '%s' into %d chunk(s)",
                document.doc_name,
                len(chunks),
            )
            return chunks

        except ImportError as exc:
            raise ChunkingError(
                "Docling is not installed. Run: pip install docling"
            ) from exc
        except ChunkingError:
            raise
        except Exception as exc:
            raise ChunkingError(
                f"Chunking failed for '{document.doc_name}': {exc}"
            ) from exc

    def _extract_metadata(self, docling_chunk, document: ParsedDocument, index: int) -> ChunkMetadata:
        """Extract metadata from a Docling chunk object.

        Docling's HybridChunker yields ``DocChunk`` Pydantic models with
        the following structure::

            DocChunk
            ├── text: str
            └── meta: DocMeta
                ├── doc_items: list[DocItem]
                │   └── prov: list[ProvenanceItem]
                │       ├── page_no: int
                │       ├── bbox: BoundingBox
                │       └── charspan: [int, int]
                ├── headings: Optional[list[str]]
                └── origin: Optional[DocumentOrigin]

        Page numbers, headings, and doc-items all live under ``chunk.meta``,
        **not** directly on the chunk object.
        """
        # ── Access the DocMeta object ─────────────────────────────────
        meta = getattr(docling_chunk, "meta", None)

        # ── Extract heading path ──────────────────────────────────────
        heading_path: list[str] = []
        if meta is not None and getattr(meta, "headings", None):
            heading_path = list(meta.headings)

        # ── Collect page numbers from doc_items → prov → page_no ──────
        page_numbers: list[int] = []
        if meta is not None and getattr(meta, "doc_items", None):
            for item in meta.doc_items:
                for prov in getattr(item, "prov", []):
                    page_no = getattr(prov, "page_no", None)
                    if page_no is not None and page_no > 0:
                        page_numbers.append(page_no)

        # De-duplicate and sort
        page_numbers = sorted(set(page_numbers))

        # Primary page number = first page the chunk appears on
        page_number = page_numbers[0] if page_numbers else 0

        if page_numbers:
            logger.debug(
                "Chunk %d: pages=%s (primary=%d)", index, page_numbers, page_number,
            )

        # Build section info from heading path
        section_title = heading_path[-1] if heading_path else ""

        # Determine chunk level from heading depth
        chunk_level = len(heading_path) if heading_path else 0

        return ChunkMetadata(
            document_id=document.doc_id,
            document_name=document.doc_name,
            page_number=page_number,
            section_id="",
            section_title=section_title,
            chunk_level=chunk_level,
            parent_chunk_id=None,
            heading_path=heading_path,
        )

    @staticmethod
    def _generate_chunk_id(doc_id: str, index: int, text: str) -> str:
        """Generate a deterministic chunk ID from document ID, index, and content."""
        content = f"{doc_id}:{index}:{text[:100]}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
