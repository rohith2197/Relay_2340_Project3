"""Integration boundary owned jointly with the US 3 teammate.

Backend contract: answer(question: str, history: list[dict]) -> dict
Return {"answer": "...", "sources": [{"title": "...", "url": "https://..."}]}.
History is untrusted user input. Apply RAG grounding in the backend.
"""
from django.conf import settings
from django.utils.module_loading import import_string


class BackendUnavailable(Exception):
    pass


def unavailable_backend(question, history):
    raise BackendUnavailable("The answer service is not connected yet.")


def get_answer(question, history):
    backend = import_string(settings.RELAY_ANSWER_BACKEND)
    result = backend(question=question, history=history)
    if not isinstance(result, dict) or not isinstance(result.get("answer"), str) or not result["answer"].strip():
        raise ValueError("Backend must return a nonempty answer string.")
    sources = result.get("sources", [])
    if not isinstance(sources, list) or any(
        not isinstance(source, dict)
        or not isinstance(source.get("title"), str)
        or not isinstance(source.get("url", ""), str)
        for source in sources
    ):
        raise ValueError("Backend returned invalid sources.")
    return {"answer": result["answer"], "sources": sources}
