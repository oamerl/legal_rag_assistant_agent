"""
Singleton Qdrant client manager.

Qdrant's local-file mode (QdrantClient(path=...)) takes an exclusive
file-lock on the storage directory, so only ONE client instance can
exist per process.  This module provides a thread-safe singleton that
both the ingestion indexer and the retrieval retriever share.

If a Qdrant *server* is used instead (via URL), there is no lock
contention, but we still benefit from reusing a single connection.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

# ── module-level singleton state ──────────────────────────────────────
_lock = threading.Lock() # to prevent multiple threads from creating the client simultaneously
_client = None          # the single QdrantClient instance
_client_path = None     # the storage path it was opened with


def get_qdrant_client():
    """
    Return the process-wide QdrantClient singleton.

    The client is lazily created on first call and reused thereafter.
    Thread-safe via a module-level lock.

    Returns
    -------
    qdrant_client.QdrantClient
        Shared Qdrant client instance.

    Raises
    ------
    RuntimeError
        If ``qdrant-client`` is not installed.
    """
    global _client, _client_path

    # Check 1 — OUTSIDE the lock (fast path, no waiting)
    if _client is not None:
        return _client # Already exists, skip the lock entirely

    # Check 2 — INSIDE the lock (safe path)
    with _lock:
        if _client is not None: # Another thread may have just created it
            return _client

        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise RuntimeError(
                "qdrant-client not installed. Run: pip install qdrant-client"
            ) from exc

        from config.settings import get_settings

        settings = get_settings()
        storage_path = settings.qdrant_path
        Path(storage_path).mkdir(parents=True, exist_ok=True)

        _client = QdrantClient(path=storage_path) # Only runs ONCE ever
        _client_path = storage_path
        logger.info("Qdrant client singleton initialised (path: %s)", storage_path)

    return _client


def close_qdrant_client() -> None:
    """
    Explicitly close the Qdrant client and release the file lock.

    Safe to call even if no client was created.
    """
    global _client, _client_path

    with _lock:
        if _client is not None:
            try:
                _client.close()
                logger.info("Qdrant client closed (path: %s)", _client_path)
            except Exception:
                logger.warning("Error closing Qdrant client", exc_info=True)
            finally:
                _client = None
                _client_path = None
