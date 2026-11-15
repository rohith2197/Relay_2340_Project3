import logging

from django.apps import AppConfig
from django.db.backends.signals import connection_created

logger = logging.getLogger(__name__)


def _load_sqlite_vec(sender, connection, **kwargs):
    """Load the sqlite-vec extension into every new SQLite connection."""
    if connection.vendor != "sqlite":
        return
    from .vector_store import load_extension
    try:
        load_extension(connection.connection)
    except Exception as exc:  # noqa: BLE001 - retrieval falls back to keyword search
        logger.warning("sqlite-vec could not be loaded; vector search is off (%s).", type(exc).__name__)


class RagConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "rag"
    verbose_name = "RAG engine"

    def ready(self):
        connection_created.connect(_load_sqlite_vec, dispatch_uid="rag_load_sqlite_vec")
