"""Embed knowledge-base chunks into the vector store (sqlite-vec).

    python manage.py embed_chunks             # embed new chunks, drop vectors of deleted ones
    python manage.py embed_chunks --rebuild   # re-embed everything (e.g. after changing model)

Run after load_dataset. Answering also syncs automatically when the knowledge
base changes, but running this first keeps the first question fast.
"""
from django.core.management.base import BaseCommand, CommandError

from rag import embeddings, retrieval, vector_store


class Command(BaseCommand):
    help = "Create or update chunk embeddings in the vector store."

    def add_arguments(self, parser):
        parser.add_argument("--rebuild", action="store_true", help="Drop all vectors and re-embed every chunk.")

    def handle(self, *args, **options):
        if not embeddings.is_configured():
            raise CommandError("OPENROUTER_API_KEY is not set, so chunks cannot be embedded.")
        try:
            result = vector_store.sync(rebuild=options["rebuild"])
        except (vector_store.VectorStoreUnavailable, embeddings.EmbeddingError) as exc:
            raise CommandError(str(exc))
        retrieval.invalidate()
        self.stdout.write(f"Model {embeddings.embedding_model()}: {result['embedded']} embedded, "
                          f"{result['deleted']} deleted, {result['total']} chunks in the store.")
