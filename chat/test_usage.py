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


# ---------------------------------------------------------------------------
# US 11 — Clear and reload knowledge base via admin website
# ---------------------------------------------------------------------------

from io import StringIO  # noqa: E402
from unittest.mock import MagicMock, patch, call as mock_call  # noqa: E402

from knowledge.ingestion import ingest_document  # noqa: E402
from knowledge.models import Chunk, Document  # noqa: E402


def _make_doc(slug="test-doc"):
    return ingest_document(
        slug=slug,
        title="Test Document",
        raw_text="This is a test document about FEMA disaster assistance, flood recovery, eligibility criteria, and required documentation for families.",
        disaster_type="flood",
        phase="recovery",
    ).document


@override_settings(STORAGES=STATIC)
class KnowledgeBaseAdminAccessTests(TestCase):
    """US 11: The knowledge-base management page must be staff-only."""

    URL = "chat:admin_knowledge"

    @classmethod
    def setUpTestData(cls):
        User = get_user_model()
        cls.staff = User.objects.create_user("kb_staff", password="pw", is_staff=True)
        cls.visitor = User.objects.create_user("kb_visitor", password="pw")

    def test_anonymous_redirected(self):
        response = self.client.get(reverse(self.URL))
        self.assertEqual(response.status_code, 302)
        self.assertIn("/admin/login/", response["Location"])

    def test_non_staff_redirected(self):
        self.client.force_login(self.visitor)
        response = self.client.get(reverse(self.URL))
        self.assertEqual(response.status_code, 302)
        self.assertIn("/admin/login/", response["Location"])

    def test_staff_can_view_page(self):
        self.client.force_login(self.staff)
        response = self.client.get(reverse(self.URL))
        self.assertContains(response, "Knowledge Base")

    def test_post_without_staff_redirected(self):
        response = self.client.post(reverse(self.URL), {"action": "reload"})
        self.assertEqual(response.status_code, 302)

    def test_get_method_allowed(self):
        self.client.force_login(self.staff)
        self.assertEqual(self.client.get(reverse(self.URL)).status_code, 200)

    def test_delete_not_allowed(self):
        self.client.force_login(self.staff)
        response = self.client.delete(reverse(self.URL))
        self.assertEqual(response.status_code, 405)


@override_settings(STORAGES=STATIC)
class KnowledgeBaseStatsTests(TestCase):
    """US 11: The page must show accurate document / chunk counts."""

    def setUp(self):
        self.client.force_login(
            get_user_model().objects.create_user("stats_staff", password="pw", is_staff=True)
        )

    def test_counts_shown_on_page(self):
        _make_doc("doc-a")
        _make_doc("doc-b")
        response = self.client.get(reverse("chat:admin_knowledge"))
        self.assertContains(response, "2")          # doc count
        chunk_count = Chunk.objects.count()
        self.assertContains(response, str(chunk_count))

    def test_empty_knowledge_base_shows_zeros(self):
        response = self.client.get(reverse("chat:admin_knowledge"))
        self.assertContains(response, "0")


@override_settings(STORAGES=STATIC)
class KnowledgeBaseClearActionTests(TestCase):
    """US 11: The 'clear' action deletes all documents and invalidates the index."""

    def setUp(self):
        self.client.force_login(
            get_user_model().objects.create_user("clear_staff", password="pw", is_staff=True)
        )
        _make_doc("clear-doc-1")
        _make_doc("clear-doc-2")

    def _post(self, action):
        return self.client.post(reverse("chat:admin_knowledge"), {"action": action},
                                follow=True)

    def test_clear_deletes_all_documents(self):
        self.assertEqual(Document.objects.count(), 2)
        with patch("chat.knowledge_views.retrieval.invalidate") as inv, \
                patch("chat.knowledge_views.vector_store.sync"):
            self._post("clear")
        self.assertEqual(Document.objects.count(), 0)
        self.assertEqual(Chunk.objects.count(), 0)
        inv.assert_called_once()

    def test_clear_calls_vector_sync_to_prune_orphans(self):
        with patch("chat.knowledge_views.retrieval.invalidate"), \
                patch("chat.knowledge_views.vector_store.sync") as vs:
            self._post("clear")
        vs.assert_called_once()

    def test_clear_shows_success_message(self):
        with patch("chat.knowledge_views.retrieval.invalidate"), \
                patch("chat.knowledge_views.vector_store.sync"):
            response = self._post("clear")
        messages_list = list(response.context["messages"])
        self.assertTrue(any("cleared" in str(m) for m in messages_list))

    def test_clear_redirects_to_knowledge_page(self):
        with patch("chat.knowledge_views.retrieval.invalidate"), \
                patch("chat.knowledge_views.vector_store.sync"):
            response = self.client.post(reverse("chat:admin_knowledge"), {"action": "clear"})
        self.assertRedirects(response, reverse("chat:admin_knowledge"))


@override_settings(STORAGES=STATIC)
class KnowledgeBaseReloadActionTests(TestCase):
    """US 11: The 'reload' action runs load_dataset and invalidates the index."""

    def setUp(self):
        self.client.force_login(
            get_user_model().objects.create_user("reload_staff", password="pw", is_staff=True)
        )

    def _post(self, action="reload"):
        return self.client.post(reverse("chat:admin_knowledge"), {"action": action}, follow=True)

    def test_reload_calls_load_dataset_command(self):
        fake_output = "CREATED     test-slug (3 chunks)\n0 created, 0 updated, 0 unchanged, 0 failed, 0 deleted"
        with patch("chat.knowledge_views.call_command", wraps=lambda cmd, *a, **kw: (
            kw["stdout"].write(fake_output) if cmd == "load_dataset" else None
        )) as cmd_mock, \
                patch("chat.knowledge_views.retrieval.invalidate"):
            response = self._post("reload")
        self.assertEqual(cmd_mock.call_args[0][0], "load_dataset")

    def test_reload_calls_invalidate(self):
        fake_output = "0 created, 0 updated, 0 unchanged, 0 failed, 0 deleted"
        with patch("chat.knowledge_views.call_command", side_effect=lambda cmd, *a, **kw: (
            kw.get("stdout", StringIO()).write(fake_output)
        )), patch("chat.knowledge_views.retrieval.invalidate") as inv:
            self._post("reload")
        self.assertTrue(inv.called)

    def test_reload_shows_success_message(self):
        fake_output = "0 created, 0 updated, 0 unchanged, 0 failed, 0 deleted"
        with patch("chat.knowledge_views.call_command", side_effect=lambda cmd, *a, **kw: (
            kw.get("stdout", StringIO()).write(fake_output)
        )), patch("chat.knowledge_views.retrieval.invalidate"):
            response = self._post("reload")
        messages_list = list(response.context["messages"])
        self.assertTrue(any("reloaded" in str(m) for m in messages_list))

    def test_reload_failure_shows_error_message(self):
        with patch("chat.knowledge_views.call_command", side_effect=Exception("disk full")), \
                patch("chat.knowledge_views.retrieval.invalidate"):
            response = self._post("reload")
        messages_list = list(response.context["messages"])
        self.assertTrue(any("failed" in str(m).lower() for m in messages_list))


@override_settings(STORAGES=STATIC)
class KnowledgeBaseClearAndReloadActionTests(TestCase):
    """US 11: The combined clear-and-reload action is atomic and invalidates correctly."""

    def setUp(self):
        self.client.force_login(
            get_user_model().objects.create_user("combo_staff", password="pw", is_staff=True)
        )
        _make_doc("old-doc")

    def _post(self):
        return self.client.post(reverse("chat:admin_knowledge"),
                                {"action": "clear_and_reload"}, follow=True)

    def test_clear_and_reload_deletes_existing_documents(self):
        self.assertEqual(Document.objects.count(), 1)
        fake_output = "0 created, 0 updated, 0 unchanged, 0 failed, 0 deleted"
        with patch("chat.knowledge_views.call_command", side_effect=lambda cmd, *a, **kw: (
            kw.get("stdout", StringIO()).write(fake_output)
        )), patch("chat.knowledge_views.retrieval.invalidate"):
            self._post()
        self.assertEqual(Document.objects.count(), 0)

    def test_clear_and_reload_shows_success_message(self):
        fake_output = "0 created, 0 updated, 0 unchanged, 0 failed, 0 deleted"
        with patch("chat.knowledge_views.call_command", side_effect=lambda cmd, *a, **kw: (
            kw.get("stdout", StringIO()).write(fake_output)
        )), patch("chat.knowledge_views.retrieval.invalidate"):
            response = self._post()
        messages_list = list(response.context["messages"])
        self.assertTrue(any("reload" in str(m).lower() for m in messages_list))

    def test_clear_and_reload_failure_shows_error_message(self):
        with patch("chat.knowledge_views.call_command", side_effect=Exception("oops")), \
                patch("chat.knowledge_views.retrieval.invalidate"):
            response = self._post()
        messages_list = list(response.context["messages"])
        self.assertTrue(any("failed" in str(m).lower() for m in messages_list))


@override_settings(STORAGES=STATIC)
class KnowledgeBaseUnknownActionTests(TestCase):
    """US 11: Unknown actions should not crash — show an error message instead."""

    def setUp(self):
        self.client.force_login(
            get_user_model().objects.create_user("unk_staff", password="pw", is_staff=True)
        )

    def test_unknown_action_returns_redirect_with_error(self):
        response = self.client.post(reverse("chat:admin_knowledge"),
                                    {"action": "nuke_everything"}, follow=True)
        messages_list = list(response.context["messages"])
        self.assertTrue(any("Unknown" in str(m) for m in messages_list))

    def test_no_action_field_is_treated_as_unknown(self):
        response = self.client.post(reverse("chat:admin_knowledge"), {}, follow=True)
        messages_list = list(response.context["messages"])
        self.assertTrue(any("Unknown" in str(m) for m in messages_list))


@override_settings(STORAGES=STATIC)
class KnowledgeBaseNavTests(TestCase):
    """US 11: Knowledge Base link appears in the admin nav for staff."""

    @classmethod
    def setUpTestData(cls):
        User = get_user_model()
        cls.staff = User.objects.create_user("nav_staff", password="pw", is_staff=True)

    def test_knowledge_base_link_in_admin_nav_for_staff(self):
        self.client.force_login(self.staff)
        for url_name in ("chat:admin_performance", "chat:admin_logs", "chat:admin_knowledge"):
            with self.subTest(page=url_name):
                response = self.client.get(reverse(url_name))
                self.assertContains(response, "Knowledge Base")
