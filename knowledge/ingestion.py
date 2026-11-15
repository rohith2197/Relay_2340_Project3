"""ingest_document() is the single entry point for adding/updating documents.
Story 6's extra ingestion methods should call it too."""
from dataclasses import dataclass

from django.core.exceptions import ValidationError
from django.core.validators import validate_slug
from django.db import transaction

from .models import Chunk, Document
from .preprocessing import chunk_text, clean_text, content_hash


class IngestionError(ValueError):
    """The document's metadata is invalid."""


@dataclass
class IngestResult:
    document: Document
    status: str  # "created", "updated" or "unchanged"
    chunk_count: int


def rebuild_chunks(document):
    document.chunks.all().delete()
    pieces = chunk_text(document.clean_text)
    Chunk.objects.bulk_create(
        Chunk(document=document, position=i, text=piece) for i, piece in enumerate(pieces)
    )
    return len(pieces)


@transaction.atomic
def refresh_document(document):
    """Re-run cleaning and chunking from raw_text."""
    document.clean_text = clean_text(document.raw_text)
    document.content_hash = content_hash(document.clean_text)
    document.save()
    return rebuild_chunks(document)


@transaction.atomic
def ingest_document(*, slug, title, raw_text, source_url="", source_type=Document.SourceType.MANUAL,
                    disaster_type=Document.DisasterType.GENERAL, phase=Document.Phase.GENERAL,
                    is_active=True):
    """Create the document, or update it if `slug` already exists."""
    try:
        if not isinstance(slug, str):
            raise TypeError
        validate_slug(slug)
    except (ValidationError, TypeError):
        raise IngestionError(f"Invalid slug {slug!r}: use letters, numbers, hyphens and underscores.")
    if len(slug) > 100:
        raise IngestionError("Slug must be 100 characters or fewer.")
    if not isinstance(title, str) or not title.strip():
        raise IngestionError("A title is required.")
    if disaster_type not in Document.DisasterType.values:
        raise IngestionError(f"Unknown disaster_type {disaster_type!r}. Use one of: {', '.join(Document.DisasterType.values)}.")
    if phase not in Document.Phase.values:
        raise IngestionError(f"Unknown phase {phase!r}. Use one of: {', '.join(Document.Phase.values)}.")
    if source_type not in Document.SourceType.values:
        raise IngestionError(f"Unknown source_type {source_type!r}.")

    clean = clean_text(raw_text)
    digest = content_hash(clean)
    metadata = {
        "title": title.strip(), "source_url": source_url or "", "source_type": source_type,
        "disaster_type": disaster_type, "phase": phase, "is_active": is_active,
    }

    document = Document.objects.filter(slug=slug).first()
    if document is None:
        document = Document.objects.create(slug=slug, raw_text=raw_text, clean_text=clean,
                                           content_hash=digest, **metadata)
        return IngestResult(document, "created", rebuild_chunks(document))

    metadata_changed = any(getattr(document, field) != value for field, value in metadata.items())
    content_changed = document.content_hash != digest
    for field, value in metadata.items():
        setattr(document, field, value)
    document.raw_text = raw_text
    document.clean_text = clean
    document.content_hash = digest
    document.save()
    if content_changed:
        return IngestResult(document, "updated", rebuild_chunks(document))
    status = "updated" if metadata_changed else "unchanged"
    return IngestResult(document, status, document.chunks.count())