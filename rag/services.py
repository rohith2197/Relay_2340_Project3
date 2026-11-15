"""The RAG pipeline behind the chat page (US 3 + US 4).

Set RELAY_ANSWER_BACKEND=rag.services.answer_question to connect it.

    answer_question(question, history) -> {"answer": str, "sources": [{"title", "url", "snippet"}]}

Pipeline: for follow-ups, rewrite the question into a standalone one ->
hybrid retrieval (embedding vector search + BM25 keywords) over active
documents -> if nothing relevant, a fixed
"not in my sources" reply with no model call -> otherwise Claude answers from
the retrieved chunks with citations -> only cited documents become sources.
An answer with no citations is treated as ungrounded and replaced by the
not-found reply.
"""
import logging
import time

from django.conf import settings
from django.dispatch import Signal

from . import retrieval
from .generation import GenerationRefused, generate_answer, rewrite_queries

logger = logging.getLogger(__name__)

# Sent after every answer, for monitoring and usage logs (US 10 / US 12).
# Keyword arguments: outcome ("answered", "no_match", "ungrounded" or "declined"),
# timings_ms (dict), retrieved_chunks, cited_sources, model. Never the question text.
answer_completed = Signal()

NOT_FOUND_ANSWER = (
    "I couldn't find information about this in Relay's sources, so I can't answer it reliably. "
    "You can try asking in a different way, or ask about applying for aid, eligibility, "
    "documents, or appeals after a flood."
)

DECLINED_ANSWER = (
    "I can't help with that request. I can answer questions about disaster recovery aid using "
    "Relay's sources."
)


def _ms(start):
    return round((time.perf_counter() - start) * 1000)


def _finish(outcome, answer, sources, timings, *, retrieved=0, model=""):
    timings["total"] = sum(v for k, v in timings.items() if k != "total")
    answer_completed.send(sender=answer_question, outcome=outcome, timings_ms=dict(timings),
                          retrieved_chunks=retrieved, cited_sources=len(sources), model=model)
    logger.info("rag answer outcome=%s retrieved=%d cited=%d timings_ms=%s",
                outcome, retrieved, len(sources), timings)
    return {"answer": answer, "sources": sources}


def answer_question(question, history=None, *, disaster_type=None, phase=None):
    history = history or []
    timings = {}

    start = time.perf_counter()
    queries = rewrite_queries(question, history)
    timings["rewrite"] = _ms(start)

    start = time.perf_counter()
    chunks = retrieval.retrieve(queries, k=getattr(settings, "RELAY_RETRIEVAL_K", retrieval.DEFAULT_K),
                                disaster_type=disaster_type, phase=phase)
    timings["retrieval"] = _ms(start)
    if not chunks:
        return _finish("no_match", NOT_FOUND_ANSWER, [], timings)

    start = time.perf_counter()
    try:
        generated = generate_answer(question, history, chunks)
    except GenerationRefused:
        timings["generation"] = _ms(start)
        return _finish("declined", DECLINED_ANSWER, [], timings, retrieved=len(chunks))
    timings["generation"] = _ms(start)

    if not generated.sources or not generated.answer:
        return _finish("ungrounded", NOT_FOUND_ANSWER, [], timings,
                       retrieved=len(chunks), model=generated.model)
    return _finish("answered", generated.answer, generated.sources, timings,
                   retrieved=len(chunks), model=generated.model)
