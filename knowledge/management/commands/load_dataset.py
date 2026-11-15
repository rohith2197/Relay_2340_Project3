"""Load or update the dataset from a manifest file.

    python manage.py load_dataset               # reads data/dataset.json
    python manage.py load_dataset --dry-run     # show changes, save nothing
    python manage.py load_dataset --prune       # delete manifest docs no longer listed
"""
import json
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from knowledge.ingestion import IngestionError, ingest_document
from knowledge.models import Document

ALLOWED_KEYS = {"slug", "title", "file", "text", "source_url", "disaster_type", "phase"}


class Command(BaseCommand):
    help = "Create or update documents from a dataset manifest (default: data/dataset.json)."

    def add_arguments(self, parser):
        parser.add_argument("--manifest", default=str(Path(settings.BASE_DIR) / "data" / "dataset.json"))
        parser.add_argument("--prune", action="store_true",
                            help="Delete manifest-sourced documents that are no longer in the manifest.")
        parser.add_argument("--dry-run", action="store_true", help="Report changes without saving them.")

    def handle(self, *args, **options):
        manifest_path = Path(options["manifest"]).resolve()
        entries = self._read_manifest(manifest_path)
        base_dir = manifest_path.parent
        manifest_slugs = {entry.get("slug") for entry in entries}
        counts = {"created": 0, "updated": 0, "unchanged": 0, "failed": 0, "deleted": 0}

        with transaction.atomic():
            for entry in entries:
                label = entry.get("slug", "<missing slug>")
                try:
                    result = self._ingest_entry(entry, base_dir)
                except (ValueError, OSError) as exc:
                    counts["failed"] += 1
                    self.stderr.write(f"  FAILED     {label}: {exc}")
                    continue
                counts[result.status] += 1
                self.stdout.write(f"  {result.status.upper():<10} {label} ({result.chunk_count} chunks)")

            if options["prune"]:
                # Uses ALL manifest slugs (even failed ones) so a typo never deletes a good document.
                stale = Document.objects.filter(source_type=Document.SourceType.MANIFEST).exclude(slug__in=manifest_slugs)
                for document in stale:
                    self.stdout.write(f"  DELETED    {document.slug}")
                    counts["deleted"] += 1
                stale.delete()

            if options["dry_run"]:
                transaction.set_rollback(True)

        prefix = "DRY RUN (nothing saved): " if options["dry_run"] else ""
        self.stdout.write(prefix + ", ".join(f"{n} {name}" for name, n in counts.items()))
        if counts["failed"]:
            raise CommandError(f"{counts['failed']} document(s) failed. Fix them and run the command again.")

    def _read_manifest(self, path):
        if not path.is_file():
            raise CommandError(f"Manifest not found: {path}")
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except ValueError as exc:
            raise CommandError(f"Manifest is not valid JSON: {exc}")
        entries = data.get("documents") if isinstance(data, dict) else None
        if not isinstance(entries, list) or not all(isinstance(e, dict) for e in entries):
            raise CommandError('Manifest must look like {"documents": [{...}, {...}]}.')
        slugs = [e.get("slug") for e in entries]
        duplicates = {s for s in slugs if slugs.count(s) > 1}
        if duplicates:
            raise CommandError(f"Duplicate slug(s) in manifest: {', '.join(sorted(map(str, duplicates)))}")
        return entries

    def _ingest_entry(self, entry, base_dir):
        unknown = set(entry) - ALLOWED_KEYS
        if unknown:
            raise IngestionError(f"Unknown key(s): {', '.join(sorted(unknown))}")
        if ("file" in entry) == ("text" in entry):
            raise IngestionError('Provide exactly one of "file" or "text".')

        if "file" in entry:
            path = (base_dir / entry["file"]).resolve()
            if not path.is_relative_to(base_dir):
                raise IngestionError('"file" must be inside the manifest folder.')
            raw_text = path.read_text(encoding="utf-8-sig")  # tolerates Windows Notepad BOMs
        else:
            raw_text = entry["text"]

        return ingest_document(
            slug=entry.get("slug", ""),
            title=entry.get("title", ""),
            raw_text=raw_text,
            source_url=entry.get("source_url", ""),
            source_type=Document.SourceType.MANIFEST,
            disaster_type=entry.get("disaster_type", Document.DisasterType.GENERAL),
            phase=entry.get("phase", Document.Phase.GENERAL),
        )