"""
Vector store indexer module — stores embedded chunks in Qdrant.

Design Pattern: Repository Pattern
    VectorStoreRepository abstracts the Qdrant client so the rest
    of the application never imports qdrant_client directly.
    This allows swapping to Pinecone, Weaviate, etc. in the future.

Qdrant is used in local-file mode (no Docker required for desktop)
or can connect to a running Qdrant server.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from config.settings import get_settings
from core.exceptions import IndexingError
from core.models import Chunk

logger = logging.getLogger(__name__)


class BaseVectorStoreRepository(ABC):
    """Abstract interface for vector store operations."""

    @abstractmethod
    def collection_exists(self) -> bool:
        """Check whether the target collection exists."""

    @abstractmethod
    def ensure_collection(self, dense_dim: int) -> None:
        """Create the collection if it does not exist."""

    @abstractmethod
    def upsert_chunks(self, chunks: list[Chunk]) -> int:
        """Store chunks with their embeddings. Return count stored."""

    @abstractmethod
    def delete_document(self, document_id: str) -> int:
        """Delete all chunks belonging to a document. Return count deleted."""



class QdrantVectorStoreRepository(BaseVectorStoreRepository):
    """
    Qdrant implementation of the vector store repository.

    Supports:
        - Dense + sparse named vectors in a single collection
        - Rich metadata payloads for filtering and citations
        - Local file-based storage (no Docker) or remote server
    """

    def __init__(self) -> None:
        self._client = None
        self._settings = get_settings()

    def _get_client(self):
        """Return the shared Qdrant client singleton."""
        if self._client is None:
            try:
                from core.qdrant_client_manager import get_qdrant_client

                self._client = get_qdrant_client()
            except RuntimeError as exc:
                raise IndexingError(str(exc)) from exc
        return self._client

    def collection_exists(self) -> bool:
        """Check if the collection exists."""
        try:
            client = self._get_client()
            return client.collection_exists(self._settings.qdrant_collection_name)
        except Exception:
            return False

    def ensure_collection(self, dense_dim: int) -> None:
        """
        Create the collection with dense + sparse vector configuration
        if it does not already exist.

        Parameters
        ----------
        dense_dim : int
            Dimensionality of the dense embedding model (e.g. 1024 for bge-large).
        """
        try:
            from qdrant_client import models

            client = self._get_client()
            collection_name = self._settings.qdrant_collection_name

            if client.collection_exists(collection_name):
                # ── Dimension-mismatch guard ──────────────────────────
                # If the user switched embedding models, the existing
                # collection will have an incompatible vector size.
                # Detect this and recreate the collection automatically.
                collection_info = client.get_collection(collection_name)
                existing_dense_cfg = collection_info.config.params.vectors.get("dense")

                if existing_dense_cfg is not None and existing_dense_cfg.size != dense_dim:
                    logger.warning(
                        "Collection '%s' has dense_dim=%d but the current "
                        "embedding model produces dim=%d. Recreating the "
                        "collection to match the new model. "
                        "⚠️  All previously indexed vectors will be lost — "
                        "please re-ingest your documents.",
                        collection_name,
                        existing_dense_cfg.size,
                        dense_dim,
                    )
                    client.delete_collection(collection_name)
                    logger.info("Deleted stale collection '%s'", collection_name)
                else:
                    logger.info("Collection '%s' already exists (dim=%d ✓)", collection_name, dense_dim)
                    return

            client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=dense_dim,
                        distance=models.Distance.COSINE,
                    ),
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(),
                },
            )

            logger.info(
                "Created collection '%s' (dense_dim=%d, sparse=SPLADE++)",
                collection_name,
                dense_dim,
            )

        except IndexingError:
            raise
        except Exception as exc:
            raise IndexingError(f"Failed to create collection: {exc}") from exc

    def upsert_chunks(self, chunks: list[Chunk]) -> int:
        """
        Store chunks with their dense + sparse embeddings and metadata.

        Parameters
        ----------
        chunks : list[Chunk]
            Chunks must have `dense_embedding`, `sparse_indices`, and
            `sparse_values` populated.

        Returns
        -------
        int
            Number of chunks stored.
        """
        if not chunks:
            return 0

        try:
            from qdrant_client import models

            client = self._get_client()
            collection_name = self._settings.qdrant_collection_name

            points = []
            for chunk in chunks:
                if chunk.dense_embedding is None:
                    logger.warning("Chunk %s has no dense embedding — skipping", chunk.id)
                    continue

                # Build payload from metadata
                payload = {
                    "text": chunk.text,
                    "document_id": chunk.metadata.document_id,
                    "document_name": chunk.metadata.document_name,
                    "page_number": chunk.metadata.page_number,
                    "section_id": chunk.metadata.section_id,
                    "section_title": chunk.metadata.section_title,
                    "chunk_level": chunk.metadata.chunk_level,
                    "parent_chunk_id": chunk.metadata.parent_chunk_id or "",
                    "heading_path": chunk.metadata.heading_path,
                    "chunk_id": chunk.id,
                }

                # Build sparse vector
                sparse_vector = None
                if chunk.sparse_indices and chunk.sparse_values:
                    sparse_vector = models.SparseVector(
                        indices=chunk.sparse_indices,
                        values=chunk.sparse_values,
                    )

                point = models.PointStruct(
                    id=abs(hash(chunk.id)) % (2**63),  # Qdrant expects int or UUID
                    vector={
                        "dense": chunk.dense_embedding,
                        **({"sparse": sparse_vector} if sparse_vector else {}),
                    },
                    payload=payload,
                )
                points.append(point)

            if not points:
                logger.warning("No valid points to upsert")
                return 0

            # Upsert in batches of 100
            batch_size = 100
            total = 0
            for i in range(0, len(points), batch_size):
                batch = points[i : i + batch_size]
                client.upsert(
                    collection_name=collection_name,
                    points=batch,
                )
                total += len(batch)
                logger.debug("Upserted batch %d-%d", i, i + len(batch))

            logger.info("Upserted %d chunk(s) into '%s'", total, collection_name)
            return total

        except IndexingError:
            raise
        except Exception as exc:
            raise IndexingError(f"Failed to upsert chunks: {exc}") from exc

    def delete_document(self, document_id: str) -> int:
        """Delete all chunks belonging to a specific document."""
        try:
            from qdrant_client import models

            client = self._get_client()
            collection_name = self._settings.qdrant_collection_name

            result = client.delete(
                collection_name=collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="document_id",
                                match=models.MatchValue(value=document_id),
                            )
                        ]
                    )
                ),
            )

            logger.info("Deleted chunks for document_id='%s'", document_id)
            return 0  # Qdrant delete doesn't return count

        except Exception as exc:
            raise IndexingError(
                f"Failed to delete document '{document_id}': {exc}"
            ) from exc


