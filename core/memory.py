"""
Memory provider — manages checkpointers.

Supports backends controlled by the ``CHECKPOINTER_DB_URI`` env var:

  - ``memory://``                           → in-memory (no persistence, volatile, dev only) (default)
  - ``sqlite:///``                          → SQLite (persistence, reads from config)
  - ``postgresql://user:pass@host:5432/db`` → Postgres (production) (for persistence)
"""

from __future__ import annotations

import logging
import os

from config.settings import get_settings
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _get_uri() -> str:
    return get_settings().checkpointer_db_uri


def _parse_backend(uri: str) -> str:
    """Return 'memory', 'sqlite', or 'postgres' from a URI string."""
    if uri.startswith("memory://") or uri == "memory":
        return "memory"
    if uri.startswith("sqlite"):
        return "sqlite"
    if uri.startswith("postgresql://") or uri.startswith("postgres://"):
        return "postgres"
    raise ValueError(
        f"Unsupported CHECKPOINTER_DB_URI scheme: {uri!r}. "
        "Supported schemes: memory://, sqlite:///, postgresql://..."
    )


def _create_sqlite_checkpointer(tenant_id: str) -> BaseCheckpointSaver:
    import sqlite3
    from langgraph.checkpoint.sqlite import SqliteSaver

    settings = get_settings()
    db_path = settings.session_db_path
    
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn=conn)
    checkpointer.setup()  # auto-create tables
    logger.info("SQLite checkpointer ready for tenant '%s' at %s", tenant_id, db_path)
    return checkpointer


def _create_postgres_checkpointer(tenant_id: str) -> BaseCheckpointSaver:
    import psycopg
    from langgraph.checkpoint.postgres import PostgresSaver

    conn = psycopg.connect(_get_uri())
    checkpointer = PostgresSaver(conn=conn)
    checkpointer.setup()  # auto-create tables
    logger.info("Postgres checkpointer ready for tenant '%s'", tenant_id)
    return checkpointer


class MemoryProvider:
    def __init__(self) -> None:
        self._backend = _parse_backend(_get_uri())
        self._tenant_checkpointers: dict[str, BaseCheckpointSaver] = {}
        logger.info("MemoryProvider initialised — backend=%s", self._backend)

    def get_checkpointer(self, tenant_id: str = "default") -> BaseCheckpointSaver:
        """Return (or create) the checkpointer for *tenant_id*."""
        if tenant_id not in self._tenant_checkpointers:
            self._tenant_checkpointers[tenant_id] = self._create(tenant_id)
        return self._tenant_checkpointers[tenant_id]

    def _create(self, tenant_id: str) -> BaseCheckpointSaver:
        """Instantiate the right checkpointer for the configured backend."""
        if self._backend == "sqlite":
            return _create_sqlite_checkpointer(tenant_id)
        if self._backend == "postgres":
            return _create_postgres_checkpointer(tenant_id)
        # fallback: in-memory
        logger.info("Fallen back to InMemorySaver checkpointer for tenant '%s' (no persistence)", tenant_id)
        return InMemorySaver()

    def close(self) -> None:
        """Close all checkpointer DB connections."""
        for tenant_id, cp in self._tenant_checkpointers.items():
            conn = getattr(cp, "conn", None)
            if conn is not None:
                try:
                    conn.close()
                    logger.info("Closed checkpointer connection for tenant '%s'", tenant_id)
                except Exception:
                    logger.warning("Failed to close connection for tenant '%s'", tenant_id, exc_info=True)
        self._tenant_checkpointers.clear()

    @staticmethod
    def make_thread_id(user_id: str, conversation_id: str) -> str:
        """Build a globally-unique thread key for a user+conversation triple."""
        return f"{user_id}::{conversation_id}"

# Global memory provider
memory_provider = MemoryProvider()
