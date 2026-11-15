from datetime import timedelta

from django.contrib.admin.views.decorators import staff_member_required
from django.core.paginator import Paginator
from django.db.models import Avg
from django.shortcuts import render
from django.utils import timezone
from django.views.decorators.cache import never_cache
from django.views.decorators.http import require_GET

from .models import QueryLog
from .usage import breakdown_24h, latency_target_ms, performance_stats


@staff_member_required
@require_GET
@never_cache
def performance(request):
    stats = performance_stats()
    stages, outcomes = breakdown_24h()
    return render(request, "chat/admin_performance.html", {
        "stats": stats,
        "stages": stages,
        "outcomes": outcomes,
        "any_slow": any(row["slow"] for row in stats),
        "target_ms": latency_target_ms(),
        "recent_failures": QueryLog.objects.filter(outcome=QueryLog.Outcome.ERROR)[:10],
    })


@staff_member_required
@require_GET
@never_cache
def logs(request):
    outcome = request.GET.get("outcome", "")
    qs = QueryLog.objects.all()
    if outcome in QueryLog.Outcome.values:
        qs = qs.filter(outcome=outcome)
    else:
        outcome = ""
    page = Paginator(qs, 25).get_page(request.GET.get("page"))
    day_ago = timezone.now() - timedelta(hours=24)
    return render(request, "chat/admin_logs.html", {
        "page": page,
        "outcome": outcome,
        "outcome_choices": QueryLog.Outcome.choices,
        "total_all": QueryLog.objects.count(),
        "total_24h": QueryLog.objects.filter(created_at__gte=day_ago).count(),
        "avg_ms": QueryLog.objects.exclude(outcome=QueryLog.Outcome.ERROR)
                  .aggregate(a=Avg("response_time_ms"))["a"],
    })
