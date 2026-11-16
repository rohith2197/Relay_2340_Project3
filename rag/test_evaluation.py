"""US 9 evaluator tests. Fabricated answers verify the harness, not chatbot accuracy."""
import copy
import io
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from django.conf import settings
from django.core.management import call_command, CommandError
from django.test import SimpleTestCase, TestCase, override_settings

from knowledge.ingestion import ingest_document
from knowledge.models import Document
from rag.evaluation import evaluate, load_cases, score_answer


QUESTIONS = Path(settings.BASE_DIR) / "data/evaluation/questions.json"


class ReferenceTests(SimpleTestCase):
    def test_cases_reference_committed_corpus_and_are_not_ingested(self):
        cases = load_cases(QUESTIONS)
        self.assertEqual(len(cases), 12)
        self.assertEqual(sum(c["unsupported"] for c in cases), 2)
        manifest = json.loads((Path(settings.BASE_DIR) / "data/dataset.json").read_text())
        documents = {d["slug"]: d for d in manifest["documents"]}
        for case in cases:
            for source in case["sources"]:
                document = documents[source["slug"]]
                self.assertEqual(source["url"], document["source_url"])
                self.assertEqual(source["file"], "data/" + document["file"])
                self.assertIn(source["evidence"], (Path(settings.BASE_DIR) / source["file"]).read_text())
        self.assertTrue(all("evaluation" not in d.get("file", "") for d in documents.values()))
        self.assertTrue({c["id"] for c in cases}.isdisjoint(documents))

    def test_invalid_fixtures_rejected(self):
        cases = load_cases(QUESTIONS)
        bad_history = copy.deepcopy(cases[0])
        bad_history["history"] = [{"role": [], "content": "Question"}]
        invalid = [[], {"cases": []}, {"cases": [cases[0], cases[0]]}, {"cases": [{}]}]
        invalid.append({"cases": [bad_history]})
        for payload in invalid:
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "bad.json"
                path.write_text(json.dumps(payload))
                with self.assertRaises(ValueError):
                    load_cases(path)


class ScoringTests(SimpleTestCase):
    def setUp(self):
        self.case = load_cases(QUESTIONS)[0]
        self.answer = {"answer": "Appeal within 60 days of the DATE of the letter.",
                       "sources": [{"title": "Appeals", "url": self.case["sources"][0]["url"]}]}

    def test_keywords_and_source_overlap_pass_but_require_review(self):
        result = score_answer(self.case, self.answer)
        self.assertTrue(result["heuristic_pass"])
        self.assertTrue(result["manual_review_required"])
        self.assertEqual(result["checks"][0]["kind"], "keyword_heuristic")

    def test_wrong_number_does_not_match_substring(self):
        self.answer["answer"] = "Appeal within 160 days of the date of the letter."
        self.assertFalse(score_answer(self.case, self.answer)["heuristic_pass"])

    def test_wrong_source_fails_attribution_check(self):
        self.answer["sources"] = [{"title": "Unrelated", "url": "https://example.org"}]
        self.assertFalse(score_answer(self.case, self.answer)["checks"][-1]["passed"])

    def test_occupancy_source_plural_receipts_matches(self):
        case = next(c for c in load_cases(QUESTIONS) if c["id"] == "occupancy-proof")
        result = score_answer(case, {
            "answer": "You can provide a utility bill or rent receipts.",
            "sources": [{"title": "Proof of occupancy", "url": case["sources"][0]["url"]}],
        })
        self.assertTrue(result["heuristic_pass"])

    def test_unsupported_case_checks_abstention(self):
        case = load_cases(QUESTIONS)[-1]
        result = score_answer(case, {"answer": "I cannot access your personal application.", "sources": []})
        self.assertTrue(result["heuristic_pass"])
        self.assertFalse(score_answer(case, {"answer": "You were approved for $5000.", "sources": []})["heuristic_pass"])

    def test_negation_can_fool_keywords_and_is_not_claimed_as_accuracy(self):
        self.answer["answer"] = "It is NOT 60 days from the date of the letter."
        result = evaluate([self.case], Mock(return_value=self.answer))
        self.assertEqual(result["summary"]["heuristic_passes"], 1)
        self.assertIn("Not measured", result["summary"]["accuracy"])
        self.assertTrue(result["results"][0]["manual_review_required"])

    def test_only_question_and_history_reach_backend(self):
        case = load_cases(QUESTIONS)[9]
        backend = Mock(return_value=self.answer)
        evaluate([case], backend)
        backend.assert_called_once_with(question=case["question"], history=case["history"])

    def test_errors_continue_and_are_counted_without_secret_text(self):
        backend = Mock(side_effect=[RuntimeError("secret-key"), self.answer])
        result = evaluate([self.case, copy.deepcopy(self.case)], backend)
        self.assertEqual(result["summary"]["errors"], 1)
        self.assertEqual(result["summary"]["answered"], 1)
        self.assertEqual(result["summary"]["heuristic_pass_rate"], 0.5)
        self.assertNotIn("secret-key", json.dumps(result))


@override_settings(RELAY_ANSWER_BACKEND="tests.mock_backend", RELAY_EMBEDDING_API_KEY="")
class EvaluationCommandTests(TestCase):
    def setUp(self):
        self.document = ingest_document(slug="isolated-reference", title="Reference", raw_text=(
            "This isolated document exists only in the Django test database. It is long enough "
            "to exercise the evaluation command without loading data or calling a provider."
        )).document

    def test_calls_configured_backend_and_outputs_report_without_ingesting_answers(self):
        out = io.StringIO()
        backend = Mock(return_value={"answer": "60 days from the date of the letter.", "sources": []})
        before = list(Document.objects.values())
        with patch("chat.services.import_string", return_value=backend) as loader:
            call_command("evaluate_rag", live=True, case_ids=["appeal-deadline"], stdout=out)
        loader.assert_called_once_with("tests.mock_backend")
        self.assertEqual(list(Document.objects.values()), before)
        report = json.loads(out.getvalue())
        self.assertEqual(report["summary"]["total"], 1)
        self.assertEqual(report["results"][0]["answer"], backend.return_value["answer"])
        self.assertEqual(report["run"]["backend"], "tests.mock_backend")
        self.assertEqual(len(report["run"]["corpus_sha256"]), 64)

    def test_live_flag_required_before_backend_call(self):
        with patch("chat.services.import_string") as loader, self.assertRaisesMessage(CommandError, "--live"):
            call_command("evaluate_rag")
        loader.assert_not_called()

    def test_unknown_case_rejected_before_backend_call(self):
        with patch("chat.services.import_string") as loader, self.assertRaisesMessage(CommandError, "Unknown case"):
            call_command("evaluate_rag", live=True, case_ids=["missing"])
        loader.assert_not_called()

    @override_settings(RELAY_ANSWER_BACKEND="chat.services.unavailable_backend")
    def test_unconfigured_backend_rejected(self):
        with self.assertRaisesMessage(CommandError, "Configure"):
            call_command("evaluate_rag", live=True)

    def test_empty_corpus_rejected(self):
        self.document.is_active = False
        self.document.save()
        with self.assertRaisesMessage(CommandError, "No active documents"):
            call_command("evaluate_rag", live=True)

    def test_backend_errors_produce_report_and_nonzero_exit(self):
        out = io.StringIO()
        with patch("chat.services.import_string", return_value=Mock(side_effect=RuntimeError("private"))):
            with self.assertRaisesMessage(CommandError, "backend errors"):
                call_command("evaluate_rag", live=True, case_ids=["appeal-deadline"], stdout=out)
        self.assertEqual(json.loads(out.getvalue())["summary"]["errors"], 1)
        self.assertNotIn("private", out.getvalue())
