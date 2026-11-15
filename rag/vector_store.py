"""Vector database: chunk embeddings stored in SQLite with the sqlite-vec extension.

Vectors live in a `vec0` virtual table (`rag_chunk_vectors`) inside the app's
own SQLite database, keyed by `knowledge.Chunk.id`, with cosine distance.
`sync()` keeps it in step with the chunk table: it embeds new chunks, deletes
vectors for chunks that no longer exist, and rebuilds everything when the
embedding model changes. Search is k-nearest-neighbour inside SQLite.

    sync()                    # after loading or changing documents
    search([query_vector])    # -> {chunk_id: cosine similarity}
"""
import logging
import threading

from django.db import connection

from knowledge.models import Chunk

from . import embeddings

logger = logging.getLogger(__name__)

VECTOR_TABLE = "rag_chunk_vectors"
META_TABLE = "rag_vector_meta"

_lock = threading.Lock()
_state = {"synced_fingerprint": None}


class VectorStoreUnavailable(Exception):
    """sqlite-vec could not be loaded, so vector search is off."""


def load_extension(sqlite_connection):
    """Load sqlite-vec into a raw sqlite3 connection (called for every new DB connection)."""
    import sqlite_vec
    sqlite_connection.enable_load_extension(True)
    try:
        sqlite_vec.load(sqlite_connection)
    finally:
        sqlite_connection.enable_load_extension(False)


def is_available():
    if connection.vendor != "sqlite":
        return False
    try:
        with connection.cursor() as cursor:
            cursor.execute("select vec_version()")
        return True
    except Exception:  # extension not loaded on this connection
        return False


def _serialize(vector):
    import sqlite_vec
    return sqlite_vec.serialize_float32(vector)


def _table_exists(cursor, name):
    cursor.execute("select 1 from sqlite_master where name = %s", [name])
    return cursor.fetchone() is not None


def _read_meta(cursor):
    cursor.execute(f"create table if not exists {META_TABLE} (key text primary key, value text not null)")
    cursor.execute(f"select key, value from {META_TABLE}")
    return dict(cursor.fetchall())


def _write_meta(cursor, **values):
    for key, value in values.items():
        cursor.execute(f"insert or replace into {META_TABLE} (key, value) values (%s, %s)", [key, str(value)])


def _drop(cursor):
    cursor.execute(f"drop table if exists {VECTOR_TABLE}")
    cursor.execute(f"delete from {META_TABLE}")


def chunk_embedding_text(chunk):
    # The document title gives each chunk context ("How to appeal a FEMA decision").
    return f"{chunk.document.title}\n\n{chunk.text}"


def sync(*, rebuild=False):
    """Bring the vector table in line with the chunk table. Returns counts of what changed."""
    if not is_available():
        raise VectorStoreUnavailable("The sqlite-vec extension is not loaded.")
    with _lock, connection.cursor() as cursor:
        meta = _read_meta(cursor)
        if rebuild or (meta and meta.get("model") != embeddings.embedding_model()):
            _drop(cursor)
            meta = {}
        exists = _table_exists(cursor, VECTOR_TABLE)

        current_ids = set(Chunk.objects.values_list("id", flat=True))
        stored_ids = set()
        if exists:
            cursor.execute(f"select chunk_id from {VECTOR_TABLE}")
            stored_ids = {row[0] for row in cursor.fetchall()}

        stale = sorted(stored_ids - current_ids)
        if stale:
            cursor.executemany(f"delete from {VECTOR_TABLE} where chunk_id = %s", [[cid] for cid in stale])

        missing = sorted(current_ids - stored_ids)
        if missing:
            chunks = list(Chunk.objects.filter(id__in=missing).select_related("document").order_by("id"))
            vectors = embeddings.embed_texts([chunk_embedding_text(c) for c in chunks])
            if not exists:
                dimensions = len(vectors[0])
                cursor.execute(f"create virtual table {VECTOR_TABLE} using vec0("
                               f"chunk_id integer primary key, embedding float[{dimensions}] distance_metric=cosine)")
                _write_meta(cursor, model=embeddings.embedding_model(), dimensions=dimensions)
            cursor.executemany(f"insert into {VECTOR_TABLE} (chunk_id, embedding) values (%s, %s)",
                               [[c.id, _serialize(v)] for c, v in zip(chunks, vectors)])
        return {"embedded": len(missing), "deleted": len(stale), "total": len(current_ids)}


def sync_if_changed(fingerprint):
    """Run sync() only when the knowledge base changed since the last sync in this process."""
    if _state["synced_fingerprint"] == fingerprint:
        return None
    result = sync()
    _state["synced_fingerprint"] = fingerprint
    if result["embedded"] or result["deleted"]:
        logger.info("Vector store synced: %s", result)
    return result


def forget_sync():
    _state["synced_fingerprint"] = None


def search(query_vectors, *, limit=30):
    """Return {chunk_id: cosine similarity} for the nearest chunks to any of the query vectors."""
    best = {}
    with connection.cursor() as cursor:
        if not _table_exists(cursor, VECTOR_TABLE):
            return best
        for vector in query_vectors:
            cursor.execute(f"select chunk_id, distance from {VECTOR_TABLE} "
                           f"where embedding match %s and k = %s order by distance",
                           [_serialize(vector), limit])
            for chunk_id, distance in cursor.fetchall():
                similarity = 1.0 - float(distance)
                if similarity > best.get(chunk_id, -1.0):
                    best[chunk_id] = similarity
    return best
