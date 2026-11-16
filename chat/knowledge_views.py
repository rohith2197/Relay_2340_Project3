"""Admin view for the knowledge-base management page (US 11).

Provides a web UI for staff to:
  - View the current knowledge-base document count and chunk count.
  - Clear all documents and chunks (DELETE + invalidate).
  - Reload documents from the dataset manifest (runs load_dataset).
  - Rebuild vectors from scratch (runs embed_chunks --rebuild, if the
    embedding key is configured).

All mutating actions are POST-only and staff-only (CSRF enforced by Django).
"""
import io
import logging

from django.contrib.admin.views.decorators import staff_member_required
from django.contrib import messages
from django.core.management import call_command
from django.db import transaction
from django.shortcuts import redirect, render
from django.views.decorators.cache import never_cache
from django.views.decorators.http import require_http_methods

from knowledge.models import Chunk, Document
from rag import retrieval, vector_store

logger = logging.getLogger(__name__)


def _kb_stats():
    return {
        "doc_count": Document.objects.count(),
        "active_count": Document.objects.filter(is_active=True).count(),
        "chunk_count": Chunk.objects.count(),
    }


@staff_member_required
@never_cache
@require_http_methods(["GET", "POST"])
def knowledge_base(request):
    if request.method == "GET":
        return render(request, "chat/admin_knowledge.html", _kb_stats())

    action = request.POST.get("action")

    if action == "clear":
        try:
            with transaction.atomic():
                deleted_docs = Document.objects.all().delete()[0]
            retrieval.invalidate()
            vector_store.sync()          # deletes orphaned vectors
            messages.success(request, f"Knowledge base cleared: {deleted_docs} row(s) deleted.")
            logger.info("admin: knowledge base cleared by %s", request.user)
        except Exception as exc:
            logger.error("admin: clear failed: %s", exc)
            messages.error(request, f"Clear failed: {exc}")

    elif action == "reload":
        try:
            out = io.StringIO()
            call_command("load_dataset", stdout=out)
            retrieval.invalidate()
            msg = out.getvalue().strip().splitlines()[-1]  # summary line
            messages.success(request, f"Dataset reloaded: {msg}")
            logger.info("admin: load_dataset run by %s — %s", request.user, msg)
        except Exception as exc:
            logger.error("admin: reload failed: %s", exc)
            messages.error(request, f"Reload failed: {exc}")

    elif action == "rebuild_vectors":
        try:
            out = io.StringIO()
            call_command("embed_chunks", "--rebuild", stdout=out)
            msg = out.getvalue().strip().splitlines()[-1]
            messages.success(request, f"Vectors rebuilt: {msg}")
            logger.info("admin: embed_chunks --rebuild run by %s — %s", request.user, msg)
        except Exception as exc:
            logger.error("admin: rebuild_vectors failed: %s", exc)
            messages.error(request, f"Vector rebuild failed: {exc}")

    elif action == "clear_and_reload":
        try:
            with transaction.atomic():
                Document.objects.all().delete()
            retrieval.invalidate()
            out = io.StringIO()
            call_command("load_dataset", stdout=out)
            retrieval.invalidate()
            msg = out.getvalue().strip().splitlines()[-1]
            messages.success(request, f"Knowledge base cleared and reloaded: {msg}")
            logger.info("admin: clear_and_reload by %s — %s", request.user, msg)
        except Exception as exc:
            logger.error("admin: clear_and_reload failed: %s", exc)
            messages.error(request, f"Clear and reload failed: {exc}")

    else:
        messages.error(request, "Unknown action.")

    return redirect("chat:admin_knowledge")
