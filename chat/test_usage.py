from datetime import timedelta
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from rag.services import answer_completed, answer_question

from .models import QueryLog
from .usage import _percentile, performance_stats

Outcome = QueryLog.Outcome
STATIC = {"staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"}}


def make_log(outcome=Outcome.ANSWERED, ms=1000, **kwargs):
    return QueryLog.objects.create(outcome=outcome, response_time_ms=ms, **kwargs)


class PercentileTests(TestCase):
    def test_empty_returns_none(self):
        self.assertIsNone(_percentile([], 95))

    def test_single_value(self):
        self.assertEqual(_percentile([7], 50), 7)
        self.assertEqual(_percentile([7], 95), 7)

    def test_ten_values(self):
        values = list(range(1, 11))
        self.assertEqual(_percentile(values, 50), 5)
        self.assertEqual(_percentile(values, 95), 10)


class SignalReceiverTests(TestCase):
    def send(self, outcome="answered", **overrides):
        args = dict(
            sender=answer_question,
            outcome=outcome,
            timings_ms={"rewrite": 10, "retrieval": 20, "generation": 300, "total": 330},
            retrieved_chunks=4,
            cited_sources=2,
            model="claude-haiku-5-5",
        )
        args.update(overrides)
        answer_completed.send(**args)

    def test_answer_creates_one_row_with_stage_timings(self):
        self.send()
        row = QueryLog.objects.get()
        self.assertEqual(row.outcome, Outcome.ANSWERED)
        self.assertEqual(row.response_time_ms, 330)
        self.assertEqual((row.rewrite_ms, row.retrieval_ms, row.generation_ms), (10, 20, 300))
        self.assertEqual((row.retrieved_chunks, row.cited_sources), (4, 2))
        self.assertEqual(row.model, "claude-haiku-5-5")

    def test_no_match_has_no_generation_time(self):
        self.send("no_match", timings_ms={"rewrite": 0, "retrieval": 5, "total": 5},
                  retrieved_chunks=0, cited_sources=0, model="")
        row = QueryLog.objects.get()
        self.assertEqual(row.outcome, Outcome.NO_MATCH)
        self.assertIsNone(row.generation_ms)
        self.assertFalse(row.is_error)


@override_settings(STORAGES=STATIC)
class AskLoggingTests(TestCase):
    def post(self, question="What documents do I need?"):
        return self.client.post(reverse("chat:ask"), {"question": question},
                                content_type="application/json")

    def test_backend_exception_logs_error_row(self):
        with patch("chat.views.get_answer", side_effect=RuntimeError("boom")):
            response = self.post()
        self.assertEqual(response.status_code, 502)
        row = QueryLog.objects.get()
        self.assertEqual(row.outcome, Outcome.ERROR)
        self.assertEqual(row.status_code, 502)
        self.assertEqual(row.error_type, "RuntimeError")

    @override_settings(RELAY_ANSWER_BACKEND="chat.services.unavailable_backend")
    def test_unavailable_backend_logs_503(self):
        response = self.post()
        self.assertEqual(response.status_code, 503)
        row = QueryLog.objects.get()
        self.assertEqual((row.outcome, row.status_code, row.error_type),
                         (Outcome.ERROR, 503, "BackendUnavailable"))

    def test_success_is_not_double_logged_by_view(self):
        with patch("chat.views.get_answer", return_value={"answer": "ok", "sources": []}):
            response = self.post()
        self.assertEqual(response.status_code, 200)
        # Success rows come from the answer_completed signal, not the view.
        self.assertEqual(QueryLog.objects.count(), 0)

    def test_validation_errors_are_not_logged(self):
        response = self.client.post(reverse("chat:ask"), "{}", content_type="application/json")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(QueryLog.objects.count(), 0)

    def test_question_text_is_never_stored(self):
        secret = "my-private-question-text"
        with patch("chat.views.get_answer", side_effect=RuntimeError("x")):
            self.post(secret)
        row = QueryLog.objects.get()
        for field in row._meta.get_fields():
            self.assertNotIn(secret, str(getattr(row, field.name, "")))


class PerformanceStatsTests(TestCase):
    def test_counts_and_percentiles(self):
        for ms in (1000, 2000, 3000):
            make_log(ms=ms)
        make_log(Outcome.ERROR, ms=99999, status_code=502, error_type="RuntimeError")
        hour = performance_stats()[0]
        self.assertEqual((hour["total"], hour["failures"], hour["failure_rate"]), (4, 1, 25.0))
        # Errors are excluded from latency numbers.
        self.assertEqual((hour["avg"], hour["p50"], hour["p95"]), (2000, 2000, 3000))
        self.assertFalse(hour["slow"])

    @override_settings(RELAY_LATENCY_TARGET_MS=2500)
    def test_slow_flag_when_p95_over_target(self):
        for ms in (1000, 2000, 3000):
            make_log(ms=ms)
        self.assertTrue(performance_stats()[0]["slow"])

    def test_old_rows_fall_outside_short_windows(self):
        old = make_log()
        QueryLog.objects.filter(pk=old.pk).update(
            created_at=old.created_at - timedelta(days=3))
        hour, day, week = performance_stats()
        self.assertEqual((hour["total"], day["total"], week["total"]), (0, 0, 1))

    def test_empty_database(self):
        hour = performance_stats()[0]
        self.assertEqual(hour["total"], 0)
        self.assertIsNone(hour["failure_rate"])
        self.assertIsNone(hour["p95"])
        self.assertFalse(hour["slow"])


@override_settings(STORAGES=STATIC)
class AdminAccessTests(TestCase):
    URLS = ("chat:admin_performance", "chat:admin_logs")

    @classmethod
    def setUpTestData(cls):
        User = get_user_model()
        cls.staff = User.objects.create_user("staff", password="pw", is_staff=True)
        cls.visitor = User.objects.create_user("visitor", password="pw")

    def test_anonymous_redirected_to_login(self):
        for name in self.URLS:
            response = self.client.get(reverse(name))
            self.assertEqual(response.status_code, 302, name)
            self.assertIn("/admin/login/", response["Location"])

    def test_non_staff_redirected_to_login(self):
        self.client.force_login(self.visitor)
        for name in self.URLS:
            response = self.client.get(reverse(name))
            self.assertEqual(response.status_code, 302, name)
            self.assertIn("/admin/login/", response["Location"])

    def test_staff_can_view_both_pages(self):
        self.client.force_login(self.staff)
        self.assertContains(self.client.get(reverse("chat:admin_performance")), "System performance")
        self.assertContains(self.client.get(reverse("chat:admin_logs")), "Usage logs")

    def test_admin_link_only_visible_to_staff(self):
        self.assertNotContains(self.client.get(reverse("chat:index")), "admin-link")
        self.client.force_login(self.visitor)
        self.assertNotContains(self.client.get(reverse("chat:index")), "admin-link")
        self.client.force_login(self.staff)
        self.assertContains(self.client.get(reverse("chat:index")), "admin-link")


@override_settings(STORAGES=STATIC)
class AdminPageContentTests(TestCase):
    def setUp(self):
        self.client.force_login(get_user_model().objects.create_user("s", password="pw", is_staff=True))

    def test_logs_filter_by_outcome(self):
        make_log(Outcome.ANSWERED)
        make_log(Outcome.ERROR, status_code=502, error_type="RuntimeError")
        response = self.client.get(reverse("chat:admin_logs"), {"outcome": "error"})
        self.assertEqual(len(response.context["page"]), 1)
        self.assertEqual(response.context["outcome"], "error")

    def test_unknown_outcome_filter_is_ignored(self):
        make_log()
        make_log(Outcome.NO_MATCH)
        response = self.client.get(reverse("chat:admin_logs"), {"outcome": "bogus"})
        self.assertEqual(len(response.context["page"]), 2)
        self.assertEqual(response.context["outcome"], "")

    def test_logs_are_paginated(self):
        for _ in range(30):
            make_log()
        response = self.client.get(reverse("chat:admin_logs"))
        self.assertEqual(len(response.context["page"]), 25)
        self.assertTrue(response.context["page"].has_next())

    def test_failures_table_shows_error_type(self):
        make_log(Outcome.ERROR, status_code=502, error_type="TimeoutError")
        self.assertContains(self.client.get(reverse("chat:admin_performance")), "TimeoutError")

    @override_settings(RELAY_LATENCY_TARGET_MS=500)
    def test_slow_warning_banner(self):
        make_log(ms=3000)
        self.assertContains(self.client.get(reverse("chat:admin_performance")),
                            "above the responsiveness target")

    def test_no_warning_when_fast(self):
        make_log(ms=100)
        self.assertNotContains(self.client.get(reverse("chat:admin_performance")),
                               "above the responsiveness target")
