"""Run reference questions through the configured chat backend, without reference leakage."""
import hashlib
import json
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from chat.services import get_answer
from knowledge.models import Document
from rag.evaluation import evaluate, load_cases


class Command(BaseCommand):
    help = "Evaluate the configured chatbot backend; JSON heuristics require manual correctness review."

    def add_arguments(self, parser):
        parser.add_argument("--questions", default=str(Path(settings.BASE_DIR) / "data/evaluation/questions.json"))
        parser.add_argument("--case", action="append", dest="case_ids", help="Run only these case IDs (repeatable).")
        parser.add_argument("--live", action="store_true", help="Enable backend calls, which may incur provider charges.")

    def handle(self, *args, **options):
        if not options["live"]:
            raise CommandError("Pass --live to call the configured backend. Use a disposable local database; provider calls may cost money.")
        if settings.RELAY_ANSWER_BACKEND == "chat.services.unavailable_backend":
            raise CommandError("Configure RELAY_ANSWER_BACKEND before evaluating.")
        try:
            cases = load_cases(options["questions"])
        except (OSError, ValueError) as exc:
            raise CommandError(f"Cannot load evaluation questions: {exc}") from exc
        if options["case_ids"]:
            selected = set(options["case_ids"])
            unknown = selected - {c["id"] for c in cases}
            if unknown:
                raise CommandError(f"Unknown case IDs: {', '.join(sorted(unknown))}")
            cases = [c for c in cases if c["id"] in selected]
        # Snapshot the actual database corpus; do not load or modify source documents.
        corpus = list(Document.objects.order_by("slug").values(
            "slug", "content_hash", "title", "source_url", "is_active", "disaster_type", "phase"))
        if not any(d["is_active"] for d in corpus):
            raise CommandError("No active documents loaded. Prepare a disposable evaluation database first.")
        report = evaluate(cases, get_answer)
        report["run"] = {
            "completed_at": timezone.now().isoformat(), "backend": settings.RELAY_ANSWER_BACKEND,
            "configured_model": settings.RELAY_LLM_MODEL,
            "embedding_model": settings.RELAY_EMBEDDING_MODEL,
            "embeddings_configured": bool(settings.RELAY_EMBEDDING_API_KEY),
            "retrieval_k": settings.RELAY_RETRIEVAL_K, "min_similarity": settings.RELAY_MIN_SIMILARITY,
            "query_rewrite": settings.RELAY_QUERY_REWRITE,
            "corpus_sha256": hashlib.sha256(json.dumps(corpus, sort_keys=True).encode()).hexdigest(),
            "cases_sha256": hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest(),
            "note": "Configured models are not proof of actual provider/model or vector use; inspect runtime logs for fallback.",
        }
        self.stdout.write(json.dumps(report, indent=2, ensure_ascii=False))
        if report["summary"]["errors"]:
            raise CommandError("Evaluation completed with backend errors; see the JSON report.")
