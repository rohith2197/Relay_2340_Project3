"""Fetch a public source and reuse the US 5/7 ingestion pipeline."""
from django.core.exceptions import ValidationError
from django.core.management.base import BaseCommand, CommandError
from django.core.validators import validate_slug
from django.db import transaction

from knowledge.ingestion import ingest_document
from knowledge.models import Document
from knowledge.scraping import fetch_url


class Command(BaseCommand):
    help = "Ingest public HTML/plain text; repeat the same slug and final URL to update."

    def add_arguments(self, parser):
        parser.add_argument("url")
        parser.add_argument("--slug", required=True)
        parser.add_argument("--title", required=True)
        parser.add_argument("--disaster-type", choices=Document.DisasterType.values, default="general")
        parser.add_argument("--phase", choices=Document.Phase.values, default="general")

    def handle(self, *args, **options):
        slug, title = options["slug"], options["title"].strip()
        try:
            validate_slug(slug)
            if len(slug) > 100 or not 1 <= len(title) <= 255:
                raise ValueError("Slug must be at most 100 characters; title must contain 1–255 characters.")
            raw_text, final_url = fetch_url(options["url"])
            with transaction.atomic():
                existing = Document.objects.filter(slug=slug).first()
                if existing and (existing.source_type != Document.SourceType.SCRAPE
                                 or existing.source_url != final_url):
                    raise ValueError("Slug belongs to a different source. Choose a new slug; no document was changed.")
                result = ingest_document(
                    slug=slug, title=title, raw_text=raw_text, source_url=final_url,
                    source_type=Document.SourceType.SCRAPE,
                    disaster_type=options["disaster_type"], phase=options["phase"],
                    is_active=existing.is_active if existing else True,
                )
        except (ValueError, ValidationError) as exc:
            raise CommandError(str(exc)) from exc
        self.stdout.write(f"{result.status.upper()} {slug} ({result.chunk_count} chunks) — {final_url}")
        self.stdout.write("Keyword retrieval refreshes on the next query. Run embed_chunks to prepare vectors.")
