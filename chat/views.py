import json
import logging

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.cache import never_cache
from django.views.decorators.http import require_GET, require_POST

from .services import BackendUnavailable, get_answer

logger = logging.getLogger(__name__)


@require_GET
@never_cache
def index(request):
    return render(request, "chat/index.html", {
        "backend_pending": settings.RELAY_ANSWER_BACKEND == "chat.services.unavailable_backend",
    })


@require_GET
def health(request):
    # Process liveness only; this does not assert that the RAG backend is ready.
    return JsonResponse({"status": "ok"})


@require_POST
@never_cache
def ask(request):
    if request.content_type != "application/json":
        return JsonResponse({"error": "Send a JSON request."}, status=415)
    try:
        data = json.loads(request.body)
    except (ValueError, UnicodeDecodeError):
        return JsonResponse({"error": "Invalid JSON."}, status=400)
    if not isinstance(data, dict):
        return JsonResponse({"error": "Expected a JSON object."}, status=400)
    question = data.get("question")
    if not isinstance(question, str) or not 1 <= len(question.strip()) <= 2000:
        return JsonResponse({"error": "Enter a question between 1 and 2,000 characters."}, status=400)
    history = data.get("history", [])
    if not isinstance(history, list) or len(history) > 20 or any(
        not isinstance(message, dict)
        or message.get("role") not in ("user", "assistant")
        or not isinstance(message.get("content"), str)
        or not 1 <= len(message["content"]) <= 12000
        for message in history
    ):
        return JsonResponse({"error": "Invalid conversation history. Start a new chat."}, status=400)
    history = [{"role": m["role"], "content": m["content"]} for m in history]
    try:
        return JsonResponse(get_answer(question.strip(), history))
    except BackendUnavailable:
        return JsonResponse({"error": "Relay’s answer service is not connected yet. Please try again once the project backend is ready."}, status=503)
    except Exception as exc:
        # Do not log questions, history, credentials, or backend exception text.
        logger.error("Answer service failure: %s", type(exc).__name__)
        return JsonResponse({"error": "Relay could not get an answer. Please try again."}, status=502)
