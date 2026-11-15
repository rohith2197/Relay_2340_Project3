import logging
import math
import time
from datetime import timedelta

from django.conf import settings
from django.db.models import Avg, Count
from django.dispatch import receiver
from django.utils import timezone

from rag.services import answer_completed

from .models import QueryLog

logger = logging.getLogger(__name__)
Outcome = QueryLog.Outcome

WINDOWS = [
    ("Last hour", timedelta(hours=1)),
    ("Last 24 hours", timedelta(hours=24)),
    ("Last 7 days", timedelta(days=7)),
]


@receiver(answer_completed)
def log_completed_answer(sender, outcome, timings_ms, retrieved_chunks, cited_sources, model, **kwargs):
    """Runs after every answer from the RAG engine. Logging must never break a chat response."""
    try:
        QueryLog.objects.create(
            outcome=outcome,
            response_time_ms=timings_ms.get("total", 0),
            rewrite_ms=timings_ms.get("rewrite"),
            retrieval_ms=timings_ms.get("retrieval"),
            generation_ms=timings_ms.get("generation"),
            retrieved_chunks=retrieved_chunks,
            cited_sources=cited_sources,
            model=(model or "")[:100],
        )
    except Exception as exc:
        logger.error("Usage log write failed: %s", type(exc).__name__)


def record_failure(started, status_code, error_type):
    """Called by the chat view when the backend raises (the signal does not fire then)."""
    try:
        QueryLog.objects.create(
            outcome=Outcome.ERROR,
            response_time_ms=int((time.perf_counter() - started) * 1000),
            status_code=status_code,
            error_type=error_type[:100],
        )
    except Exception as exc:
        logger.error("Usage log write failed: %s", type(exc).__name__)


def latency_target_ms():
    return getattr(settings, "RELAY_LATENCY_TARGET_MS", 12000)


def _percentile(sorted_values, pct):
    if not sorted_values:
        return None
    return sorted_values[max(0, math.ceil(pct / 100 * len(sorted_values)) - 1)]


def performance_stats():
    """Latency is measured over queries that did not error (what users actually waited for)."""
    now = timezone.now()
    target = latency_target_ms()
    rows = []
    for label, delta in WINDOWS:
        qs = QueryLog.objects.filter(created_at__gte=now - delta)
        total = qs.count()
        failures = qs.filter(outcome=Outcome.ERROR).count()
        times = sorted(qs.exclude(outcome=Outcome.ERROR).values_list("response_time_ms", flat=True))
        p95 = _percentile(times, 95)
        rows.append({
            "label": label,
            "total": total,
            "failures": failures,
            "failure_rate": round(100 * failures / total, 1) if total else None,
            "avg": round(sum(times) / len(times)) if times else None,
            "p50": _percentile(times, 50),
            "p95": p95,
            "slow": p95 is not None and p95 > target,
        })
    return rows


def breakdown_24h():
    since = timezone.now() - timedelta(hours=24)
    qs = QueryLog.objects.filter(created_at__gte=since)
    agg = qs.exclude(outcome=Outcome.ERROR).aggregate(
        rewrite=Avg("rewrite_ms"), retrieval=Avg("retrieval_ms"), generation=Avg("generation_ms"))
    stages = [
        ("Rewrite follow-up question", agg["rewrite"]),
        ("Search knowledge base", agg["retrieval"]),
        ("Claude answer", agg["generation"]),
    ]
    stages = [(label, round(v) if v is not None else None) for label, v in stages]
    counts = dict(qs.values_list("outcome").annotate(n=Count("id")))
    outcomes = [(label, counts.get(value, 0)) for value, label in Outcome.choices]
    return stages, outcomes
