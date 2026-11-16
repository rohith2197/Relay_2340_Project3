import io
import json
import urllib.error
from types import SimpleNamespace as NS
from unittest.mock import patch

from django.core.management import CommandError, call_command
from django.test import TestCase, override_settings

from knowledge.ingestion import ingest_document
from knowledge.models import Document

from . import embeddings, retrieval, vector_store
from .generation import _recent_history, build_document_blocks, parse_cited_response, rewrite_queries
from .services import DECLINED_ANSWER, NOT_FOUND_ANSWER, answer_question, answer_completed

APPEALS_TEXT = (
    "FEMA appeals: if your application for assistance was denied, you can appeal the decision. "
    "Send your appeal letter within 60 days of the date on the denial letter. "
    "Include copies of documents that show what was missing, such as proof of ownership or occupancy."
)
ELIGIBILITY_TEXT = (
    "A household can qualify for FEMA assistance through a U.S. citizen child, even if a parent is "
    "undocumented. Only the child's information is used for eligibility on the application."
)
RENTERS_TEXT = (
    "Renters without a written lease can still ask for help replacing household belongings. "
    "If your ID papers were destroyed by the water, other proof of where you lived, such as "
    "utility bills or a letter from your landlord, may be accepted."
)
WILDFIRE_TEXT = (
    "Wildfire smoke preparedness: keep N95 masks at home and know two evacuation routes. "
    "Sign up for local emergency alerts before fire season starts in your county."
)


def make_doc(slug, text, *, title=None, disaster_type="flood", phase="recovery", url=""):
    return ingest_document(slug=slug, title=title or slug.replace("-", " ").title(), raw_text=text,
                           source_url=url, disaster_type=disaster_type, phase=phase).document


class KnowledgeBaseMixin:
    """Four small documents. Vector search is off unless a test turns it on."""

    def setUp(self):
        patcher = override_settings(RELAY_EMBEDDING_API_KEY="")
        patcher.enable()
        self.addCleanup(patcher.disable)
        retrieval.invalidate()
        self.appeals = make_doc("fema-appeals", APPEALS_TEXT, title="How to appeal a FEMA decision",
                                url="https://example.org/appeals")
        self.eligibility = make_doc("citizen-child", ELIGIBILITY_TEXT, title="Qualifying through a child")
        self.renters = make_doc("renters-help", RENTERS_TEXT, title="Help for renters")
        self.wildfire = make_doc("wildfire-prep", WILDFIRE_TEXT, title="Wildfire preparedness",
                                 disaster_type="wildfire", phase="preparedness")


class TokenizeTests(TestCase):
    def test_accents_case_and_plurals_fold_together(self):
        self.assertEqual(retrieval.tokenize("Café"), retrieval.tokenize("cafe"))
        self.assertEqual(retrieval.tokenize("Appeals"), retrieval.tokenize("appeal"))
        self.assertEqual(retrieval.tokenize("phases"), retrieval.tokenize("phase"))

    def test_stopwords_removed(self):
        self.assertEqual(retrieval.tokenize("What is the FEMA appeal"), ["fema", "appeal"])


class KeywordRetrievalTests(KnowledgeBaseMixin, TestCase):
    """BM25 half of the hybrid search (also the fallback when embeddings are unavailable)."""

    def test_most_relevant_document_ranks_first(self):
        results = retrieval.retrieve("How do I appeal a denied FEMA application?")
        self.assertEqual(results[0].document_slug, "fema-appeals")
        self.assertEqual(results[0].url, "https://example.org/appeals")

    def test_renter_question_finds_renter_document(self):
        results = retrieval.retrieve("What proof of where I lived works if my papers were destroyed?")
        self.assertEqual(results[0].document_slug, "renters-help")

    def test_unrelated_question_returns_nothing(self):
        self.assertEqual(retrieval.retrieve("What is the best pizza topping?"), [])
        self.assertEqual(retrieval.retrieve("   "), [])

    def test_best_score_across_phrasings_is_kept(self):
        results = retrieval.retrieve(["zzz unrelated", "renters without written lease"])
        self.assertEqual([r.document_slug for r in results], ["renters-help"])

    def test_filters_keep_matching_and_general_documents(self):
        general = make_doc("general-alerts", WILDFIRE_TEXT.replace("Wildfire smoke", "General"),
                           disaster_type="general", phase="general")
        slugs = {r.document_slug for r in retrieval.retrieve("evacuation routes emergency alerts",
                                                             disaster_type="wildfire")}
        self.assertEqual(slugs, {"wildfire-prep", general.slug})
        flood = retrieval.retrieve("evacuation routes emergency alerts", disaster_type="flood", phase="recovery")
        self.assertEqual([r.document_slug for r in flood], [general.slug])

    def test_inactive_documents_are_excluded_and_index_refreshes(self):
        self.assertTrue(retrieval.retrieve("appeal denial letter"))
        self.appeals.is_active = False
        self.appeals.save()
        self.assertFalse(any(r.document_slug == "fema-appeals" for r in retrieval.retrieve("appeal denial letter")))

    def test_new_documents_are_found_without_manual_reload(self):
        retrieval.retrieve("appeal")  # build the index
        make_doc("mold-cleanup", "Mold cleanup after flooding: wear gloves and a mask, dry the house within "
                 "48 hours, and throw away soaked drywall and carpet that cannot be dried.")
        self.assertEqual(retrieval.retrieve("mold cleanup drywall")[0].document_slug, "mold-cleanup")

    def test_per_document_cap_and_k(self):
        long_text = "\n\n".join(f"Appeal step {i}: send the appeal letter and documents. " + "x " * 450
                                for i in range(6))
        make_doc("long-appeals", long_text)
        results = retrieval.retrieve("appeal letter documents", k=10, max_per_document=2)
        per_doc = {}
        for r in results:
            per_doc[r.document_slug] = per_doc.get(r.document_slug, 0) + 1
        self.assertLessEqual(max(per_doc.values()), 2)
        self.assertEqual(len(retrieval.retrieve("appeal letter documents", k=1)), 1)


def chunk(chunk_id, document_id, title, text, position=0, url=""):
    return retrieval.ScoredChunk(chunk_id=chunk_id, document_id=document_id, document_slug=f"d{document_id}",
                                 title=title, url=url, disaster_type="flood", phase="recovery",
                                 position=position, text=text, score=1.0)


def cite(document_index, start, end, cited_text="cited"):
    return NS(type="content_block_location", document_index=document_index, start_block_index=start,
              end_block_index=end, cited_text=cited_text, document_title=None)


def text(value, *citations):
    return NS(type="text", text=value, citations=list(citations) or None)


class GenerationParsingTests(TestCase):
    def setUp(self):
        self.groups = [
            [chunk(10, 1, "Appeals guide", "Appeal within 60 days.", 0, "https://example.org/a"),
             chunk(11, 1, "Appeals guide", "Include documents.", 1, "https://example.org/a")],
            [chunk(20, 2, "Eligibility", "A citizen child can qualify.", 0)],
        ]

    def test_document_blocks_have_citations_and_one_block_per_chunk(self):
        blocks = build_document_blocks(self.groups)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0]["citations"], {"enabled": True})
        self.assertEqual(blocks[0]["title"], "Appeals guide")
        self.assertEqual([c["text"] for c in blocks[0]["source"]["content"]],
                         ["Appeal within 60 days.", "Include documents."])

    def test_citations_become_numbered_sources_in_order_of_first_use(self):
        content = [
            NS(type="thinking", thinking=""),
            text("You may qualify through your child.", cite(1, 0, 1, "A citizen child can qualify.")),
            text(" You can appeal within 60 days.\n", cite(0, 0, 2), cite(1, 0, 1)),
            text("Ask a caseworker for help."),
        ]
        answer, sources, chunk_ids = parse_cited_response(content, self.groups)
        self.assertEqual(answer, "You may qualify through your child. [1] You can appeal within 60 days. [2][1]\n"
                                 "Ask a caseworker for help.")
        self.assertEqual([s["title"] for s in sources], ["Eligibility", "Appeals guide"])
        self.assertEqual(sources[0]["snippet"], "A citizen child can qualify.")
        self.assertEqual(sources[1]["url"], "https://example.org/a")
        self.assertEqual(chunk_ids, [20, 10, 11])

    def test_invalid_document_index_is_ignored(self):
        answer, sources, _ = parse_cited_response([text("Hi.", cite(7, 0, 1))], self.groups)
        self.assertEqual((answer, sources), ("Hi.", []))

    def test_history_is_trimmed_and_starts_with_user(self):
        history = [{"role": "assistant", "content": "Hello"}, {"role": "user", "content": "x" * 5000},
                   {"role": "assistant", "content": "Reply"}]
        trimmed = _recent_history(history)
        self.assertEqual(trimmed[0]["role"], "user")
        self.assertEqual(len(trimmed[0]["content"]), 4000)
        self.assertEqual(len(_recent_history([{"role": "user", "content": "q"}] * 20)), 6)


class FakeClient:
    """Stands in for anthropic.Anthropic: messages.create (rewrite) and beta.messages.create (answer)."""

    def __init__(self, answer_content=None, rewrite_text=None, stop_reason="end_turn", answer_error=None):
        self.answer_content = answer_content or []
        self.rewrite_text = rewrite_text or json.dumps({"standalone_question": "Can I appeal a FEMA denial?"})
        self.stop_reason = stop_reason
        self.answer_error = answer_error
        self.rewrite_calls, self.answer_calls = [], []
        self.messages = NS(create=self._rewrite)
        self.beta = NS(messages=NS(create=self._answer))

    def with_options(self, **kwargs):
        return self

    def _rewrite(self, **kwargs):
        self.rewrite_calls.append(kwargs)
        return NS(content=[NS(type="text", text=self.rewrite_text)], stop_reason="end_turn")

    def _answer(self, **kwargs):
        self.answer_calls.append(kwargs)
        if self.answer_error:
            raise self.answer_error
        return NS(content=self.answer_content, stop_reason=self.stop_reason, model="test-model")


class RewriteTests(TestCase):
    HISTORY = [{"role": "user", "content": "My FEMA application was denied"},
               {"role": "assistant", "content": "I'm sorry to hear that."}]

    def test_follow_up_adds_standalone_question(self):
        client = FakeClient(rewrite_text=json.dumps({"standalone_question": "Can I appeal a FEMA denial?"}))
        with patch("rag.generation.get_client", return_value=client):
            queries = rewrite_queries("can I appeal it?", self.HISTORY)
        self.assertEqual(queries, ["can I appeal it?", "Can I appeal a FEMA denial?"])
        self.assertIn("My FEMA application was denied", client.rewrite_calls[0]["messages"][0]["content"])

    def test_first_question_is_not_rewritten(self):
        with patch("rag.generation.get_client") as get_client:
            self.assertEqual(rewrite_queries("Can I appeal?", []), ["Can I appeal?"])
        get_client.assert_not_called()

    def test_bad_output_falls_back_to_original_question(self):
        with patch("rag.generation.get_client", return_value=FakeClient(rewrite_text="not json")), \
                self.assertLogs("rag.generation", level="WARNING"):
            self.assertEqual(rewrite_queries("Can I appeal it?", self.HISTORY), ["Can I appeal it?"])

    @override_settings(RELAY_QUERY_REWRITE=False)
    def test_can_be_disabled(self):
        with patch("rag.generation.get_client") as get_client:
            self.assertEqual(rewrite_queries("Can I appeal it?", self.HISTORY), ["Can I appeal it?"])
        get_client.assert_not_called()


class AnswerQuestionTests(KnowledgeBaseMixin, TestCase):
    def run_pipeline(self, client, question="How do I appeal a denied FEMA application?", history=(), **kwargs):
        events = []
        receiver = lambda sender, **kw: events.append(kw)  # noqa: E731
        answer_completed.connect(receiver)
        try:
            with patch("rag.generation.get_client", return_value=client):
                result = answer_question(question, list(history), **kwargs)
        finally:
            answer_completed.disconnect(receiver)
        return result, events[0]

    def test_grounded_answer_returns_cited_sources(self):
        client = FakeClient([text("Send your appeal within 60 days.", cite(0, 0, 1, "Send your appeal letter"))])
        result, event = self.run_pipeline(client)
        self.assertEqual(result["answer"], "Send your appeal within 60 days. [1]")
        self.assertEqual(result["sources"], [{"title": "How to appeal a FEMA decision",
                                              "url": "https://example.org/appeals",
                                              "snippet": "Send your appeal letter"}])
        self.assertEqual(event["outcome"], "answered")
        self.assertEqual(event["cited_sources"], 1)
        self.assertNotIn("question", event)
        request = client.answer_calls[0]
        document = request["messages"][-1]["content"][0]
        self.assertEqual(document["title"], "How to appeal a FEMA decision")
        self.assertIn("Use only the provided documents", request["system"])

    def test_no_relevant_documents_skips_the_model(self):
        client = FakeClient()
        result, event = self.run_pipeline(client, "What is the best pizza topping?")
        self.assertEqual(result, {"answer": NOT_FOUND_ANSWER, "sources": []})
        self.assertEqual(event["outcome"], "no_match")
        self.assertEqual(client.answer_calls, [])

    def test_uncited_answer_is_treated_as_ungrounded(self):
        result, event = self.run_pipeline(FakeClient([text("Appeals usually take 30 days.")]))
        self.assertEqual(result, {"answer": NOT_FOUND_ANSWER, "sources": []})
        self.assertEqual(event["outcome"], "ungrounded")

    def test_model_refusal_returns_polite_decline(self):
        result, event = self.run_pipeline(FakeClient(stop_reason="refusal"))
        self.assertEqual(result, {"answer": DECLINED_ANSWER, "sources": []})
        self.assertEqual(event["outcome"], "declined")

    def test_filters_are_applied(self):
        client = FakeClient([text("x", cite(0, 0, 1))])
        result, event = self.run_pipeline(client, disaster_type="wildfire")
        self.assertEqual(event["outcome"], "no_match")

    def test_follow_up_is_rewritten_before_retrieval(self):
        client = FakeClient([text("Yes.", cite(0, 0, 1))],
                            rewrite_text=json.dumps({"standalone_question": "How do I appeal a FEMA denial?"}))
        history = [{"role": "user", "content": "FEMA denied me"}, {"role": "assistant", "content": "Sorry."}]
        result, event = self.run_pipeline(client, "and how do I do that?", history=history)
        self.assertEqual(event["outcome"], "answered")  # "and how do I do that?" alone matches nothing
        self.assertEqual(len(client.rewrite_calls), 1)
        self.assertEqual(client.answer_calls[0]["messages"][0], history[0])

    def test_api_failure_propagates_to_the_chat_view(self):
        with self.assertRaises(RuntimeError):
            self.run_pipeline(FakeClient(answer_error=RuntimeError("timeout")))


@override_settings(RELAY_ANSWER_BACKEND="rag.services.answer_question",
                   STORAGES={"staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"}})
class ChatIntegrationTests(KnowledgeBaseMixin, TestCase):
    def test_chat_endpoint_returns_rag_answer(self):
        client = FakeClient([text("You can appeal.", cite(0, 0, 1, "you can appeal the decision"))])
        with patch("rag.generation.get_client", return_value=client):
            response = self.client.post("/api/chat/", json.dumps({"question": "Can I appeal a FEMA denial?"}),
                                        content_type="application/json")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["answer"], "You can appeal. [1]")
        self.assertEqual(body["sources"][0]["title"], "How to appeal a FEMA decision")
        self.assertEqual(body["sources"][0]["snippet"], "you can appeal the decision")


# ---------------------------------------------------------------------------
# Embeddings, vector store and hybrid search
# ---------------------------------------------------------------------------

# A tiny stand-in for an embedding model: words with the same
# meaning (synonyms) map to the same dimension.
CONCEPTS = {
    "appeal": 0, "renter": 1, "tenant": 1,
    "contract": 2, "leas": 2, "document": 3, "paper": 3, "proof": 4,
    "lost": 5, "destroyed": 5, "child": 6, "citizen": 7, "qualify": 8,
    "wildfir": 9, "evacuation": 9, "denied": 10, "deni": 10, "fema": 11, "letter": 12,
}
DIMENSIONS = 16


def fake_embed(texts):
    vectors = []
    for value in texts:
        vector = [0.0] * DIMENSIONS
        for token in retrieval.tokenize(value):
            if token in CONCEPTS:
                vector[CONCEPTS[token]] += 1.0
        if not any(vector):
            vector[DIMENSIONS - 1] = 1.0  # "nothing recognised"
        vectors.append(vector)
    return vectors


@override_settings(RELAY_EMBEDDING_API_KEY="test-key", RELAY_EMBEDDING_MODEL="test/model-a")
class EmbeddingClientTests(TestCase):
    def fake_response(self, vectors):
        data = {"data": [{"object": "embedding", "index": i, "embedding": v} for i, v in vectors]}
        return io.BytesIO(json.dumps(data).encode())

    def test_batches_in_order_with_auth_header(self):
        requests = []

        def urlopen(request, timeout):
            body = json.loads(request.data)
            requests.append((request, body))
            # Return items out of order to check they are sorted by "index".
            return self.fake_response(reversed([(i, [float(len(t))]) for i, t in enumerate(body["input"])]))

        with patch("rag.embeddings.urllib.request.urlopen", side_effect=urlopen), \
                patch("rag.embeddings.BATCH_SIZE", 2):
            vectors = embeddings.embed_texts(["a", "bb", "ccc"])
        self.assertEqual(vectors, [[1.0], [2.0], [3.0]])
        self.assertEqual(len(requests), 2)
        request, body = requests[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        self.assertEqual(body, {"model": "test/model-a", "input": ["a", "bb"], "encoding_format": "float"})

    def test_network_error_becomes_embedding_error(self):
        with patch("rag.embeddings.urllib.request.urlopen", side_effect=urllib.error.URLError("down")):
            with self.assertRaises(embeddings.EmbeddingError):
                embeddings.embed_texts(["a"])

    @override_settings(RELAY_EMBEDDING_API_KEY="")
    def test_missing_key(self):
        with self.assertRaises(embeddings.EmbeddingUnavailable):
            embeddings.embed_texts(["a"])


@override_settings(RELAY_EMBEDDING_API_KEY="test-key", RELAY_EMBEDDING_MODEL="test/model-a")
class VectorStoreTests(KnowledgeBaseMixin, TestCase):
    def setUp(self):
        super().setUp()
        patcher = override_settings(RELAY_EMBEDDING_API_KEY="test-key")
        patcher.enable()
        self.addCleanup(patcher.disable)
        self.calls = []
        embed = patch("rag.embeddings.embed_texts", side_effect=self.record)
        embed.start()
        self.addCleanup(embed.stop)

    def record(self, texts):
        self.calls.append(len(texts))
        return fake_embed(texts)

    def test_sqlite_vec_is_loaded(self):
        self.assertTrue(vector_store.is_available())

    def test_sync_embeds_new_chunks_once_and_drops_deleted_ones(self):
        self.assertEqual(vector_store.sync(), {"embedded": 4, "deleted": 0, "total": 4})
        self.assertEqual(vector_store.sync(), {"embedded": 0, "deleted": 0, "total": 4})
        self.wildfire.delete()
        make_doc("mold", "Mold cleanup after flooding: wear gloves and a mask, dry the house within 48 hours, "
                 "and throw away soaked drywall and carpet that cannot be dried in time.")
        self.assertEqual(vector_store.sync(), {"embedded": 1, "deleted": 1, "total": 4})
        self.assertEqual(self.calls, [4, 1])

    def test_changing_the_model_re_embeds_everything(self):
        vector_store.sync()
        with override_settings(RELAY_EMBEDDING_MODEL="test/model-b"):
            self.assertEqual(vector_store.sync()["embedded"], 4)

    def test_search_returns_cosine_similarity(self):
        vector_store.sync()
        [query] = fake_embed(["renter contract"])
        similarities = vector_store.search([query], limit=10)
        best = max(similarities, key=similarities.get)
        self.assertEqual(best, self.renters.chunks.get().id)
        self.assertGreater(similarities[best], 0.5)


@override_settings(RELAY_MIN_SIMILARITY=0.3)
class HybridRetrievalTests(KnowledgeBaseMixin, TestCase):
    def setUp(self):
        super().setUp()
        patcher = override_settings(RELAY_EMBEDDING_API_KEY="test-key", RELAY_EMBEDDING_MODEL="test/model-a")
        patcher.enable()
        self.addCleanup(patcher.disable)
        embed = patch("rag.embeddings.embed_texts", side_effect=fake_embed)
        self.embed = embed.start()
        self.addCleanup(embed.stop)

    def test_vector_search_finds_a_match_keywords_miss(self):
        # No shared words with the renters document, but the same meaning.
        results = retrieval.retrieve("Can a tenant with no contract receive support?")
        self.assertEqual(results[0].document_slug, "renters-help")
        self.assertIsNone(results[0].keyword_score)
        self.assertGreaterEqual(results[0].similarity, 0.3)

    def test_chunks_found_by_both_methods_rank_highest(self):
        results = retrieval.retrieve("appeal denied FEMA letter")
        self.assertEqual(results[0].document_slug, "fema-appeals")
        self.assertIsNotNone(results[0].keyword_score)
        self.assertIsNotNone(results[0].similarity)

    def test_unrelated_question_still_returns_nothing(self):
        self.assertEqual(retrieval.retrieve("What is the best pizza topping?"), [])

    def test_filters_and_inactive_documents_apply_to_vector_results(self):
        self.renters.is_active = False
        self.renters.save()
        self.assertFalse(any(r.document_slug == "renters-help"
                             for r in retrieval.retrieve("Can a tenant with no contract receive support?")))
        self.assertEqual(retrieval.retrieve("appeal denied FEMA letter", disaster_type="wildfire"), [])

    def test_embedding_failure_falls_back_to_keywords(self):
        self.embed.side_effect = embeddings.EmbeddingError("down")
        with self.assertLogs("rag.retrieval", level="WARNING"):
            results = retrieval.retrieve("appeal denied FEMA letter")
        self.assertEqual(results[0].document_slug, "fema-appeals")
        self.assertIsNone(results[0].similarity)

    def test_new_documents_are_embedded_on_the_next_question(self):
        retrieval.retrieve("appeal")
        make_doc("renters-texas", "Renters in Texas: a tenant can apply for FEMA rental assistance even "
                 "without a written lease, by showing other proof such as utility bills or letters.")
        slugs = [r.document_slug for r in retrieval.retrieve("tenant proof")]
        self.assertIn("renters-texas", slugs)


class EmbedChunksCommandTests(KnowledgeBaseMixin, TestCase):
    def test_requires_api_key(self):
        with self.assertRaises(CommandError):
            call_command("embed_chunks")

    @override_settings(RELAY_EMBEDDING_MODEL="test/model-a")
    def test_embeds_chunks(self):
        out = io.StringIO()
        with override_settings(RELAY_EMBEDDING_API_KEY="test-key"), \
                patch("rag.embeddings.embed_texts", side_effect=fake_embed):
            call_command("embed_chunks", stdout=out)
        self.assertIn("4 embedded, 0 deleted, 4 chunks", out.getvalue())


# ---------------------------------------------------------------------------
# US 8 — Dynamic prompt selection
# ---------------------------------------------------------------------------

from .prompts import (  # noqa: E402  (placed after related test classes)
    APPEAL_GUIDANCE,
    CHECKLIST_GUIDANCE,
    DEFAULT_GUIDANCE,
    DOCUMENTATION_GUIDANCE,
    ELIGIBILITY_GUIDANCE,
    build_system_prompt,
    classify_intent,
    select_prompt,
)


class IntentClassifierTests(TestCase):
    """classify_intent() should return the right intent label for representative questions."""

    # --- checklist ---
    def test_steps_triggers_checklist(self):
        self.assertEqual(classify_intent("What are the steps to apply for FEMA?"), "checklist")

    def test_how_do_i_apply_triggers_checklist(self):
        self.assertEqual(classify_intent("How do I apply for assistance?"), "checklist")

    def test_how_do_i_submit_triggers_checklist(self):
        self.assertEqual(classify_intent("How do I submit my appeal?"), "checklist")

    def test_checklist_keyword_triggers_checklist(self):
        self.assertEqual(classify_intent("Give me a checklist for applying"), "checklist")

    def test_walk_me_through_triggers_checklist(self):
        self.assertEqual(classify_intent("Can you walk me through the process?"), "checklist")

    # --- appeal (should not be shadowed by checklist when no checklist words) ---
    def test_appeal_triggers_appeal(self):
        self.assertEqual(classify_intent("I want to appeal FEMA's decision"), "appeal")

    def test_denied_triggers_appeal(self):
        self.assertEqual(classify_intent("My application was denied. What can I do?"), "appeal")

    def test_denial_triggers_appeal(self):
        self.assertEqual(classify_intent("I got a denial letter"), "appeal")

    def test_dispute_triggers_appeal(self):
        self.assertEqual(classify_intent("Can I dispute the rejection?"), "appeal")

    # --- eligibility ---
    def test_eligible_triggers_eligibility(self):
        self.assertEqual(classify_intent("Am I eligible for FEMA assistance?"), "eligibility")

    def test_qualify_triggers_eligibility(self):
        self.assertEqual(classify_intent("Do I qualify if I'm undocumented?"), "eligibility")

    def test_requirements_triggers_eligibility(self):
        self.assertEqual(classify_intent("What are the requirements for disaster aid?"), "eligibility")

    def test_undocumented_triggers_eligibility(self):
        self.assertEqual(classify_intent("Can undocumented people get help?"), "eligibility")

    def test_citizenship_triggers_eligibility(self):
        self.assertEqual(classify_intent("Does citizenship affect my application?"), "eligibility")

    def test_criteria_triggers_eligibility(self):
        self.assertEqual(classify_intent("What criteria do I need to meet?"), "eligibility")

    # --- documentation ---
    def test_documents_triggers_documentation(self):
        self.assertEqual(classify_intent("What documents do I need to provide?"), "documentation")

    def test_lost_id_triggers_documentation(self):
        self.assertEqual(classify_intent("I lost my ID in the flood"), "documentation")

    def test_missing_papers_triggers_documentation(self):
        self.assertEqual(classify_intent("My papers are missing"), "documentation")

    def test_proof_triggers_documentation(self):
        self.assertEqual(classify_intent("What counts as proof of residence?"), "documentation")

    def test_no_lease_triggers_documentation(self):
        self.assertEqual(classify_intent("I have no lease, can I still apply?"), "documentation")

    # --- general (fallback) ---
    def test_unrelated_question_falls_back_to_general(self):
        self.assertEqual(classify_intent("What is FEMA?"), "general")

    def test_background_question_falls_back_to_general(self):
        self.assertEqual(classify_intent("Tell me about the Valencia floods"), "general")

    def test_empty_question_falls_back_to_general(self):
        self.assertEqual(classify_intent(""), "general")

    def test_whitespace_only_falls_back_to_general(self):
        self.assertEqual(classify_intent("   "), "general")


class SelectPromptTests(TestCase):
    """select_prompt() returns the right guidance string and build_system_prompt() includes it."""

    def test_checklist_question_returns_checklist_guidance(self):
        prompt = select_prompt("What are the steps to apply?", [])
        self.assertEqual(prompt, CHECKLIST_GUIDANCE)

    def test_appeal_question_returns_appeal_guidance(self):
        prompt = select_prompt("My application was denied, can I appeal?", [])
        self.assertEqual(prompt, APPEAL_GUIDANCE)

    def test_eligibility_question_returns_eligibility_guidance(self):
        prompt = select_prompt("Am I eligible if I am undocumented?", [])
        self.assertEqual(prompt, ELIGIBILITY_GUIDANCE)

    def test_documentation_question_returns_documentation_guidance(self):
        prompt = select_prompt("What documents do I need?", [])
        self.assertEqual(prompt, DOCUMENTATION_GUIDANCE)

    def test_general_question_returns_default_guidance(self):
        prompt = select_prompt("Tell me about the Valencia floods", [])
        self.assertEqual(prompt, DEFAULT_GUIDANCE)

    def test_history_is_accepted_but_not_required(self):
        # history should not affect the result (classification is question-only)
        history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        self.assertEqual(
            select_prompt("What documents do I need?", history),
            select_prompt("What documents do I need?", []),
        )

    def test_grounding_rules_always_present_in_system_prompt(self):
        for question in (
            "What are the steps to apply?",
            "I was denied — can I appeal?",
            "Am I eligible?",
            "I lost my ID",
            "What is FEMA?",
        ):
            with self.subTest(question=question):
                system = build_system_prompt(question, [])
                self.assertIn("Use only the provided documents", system)
                self.assertIn("not a lawyer", system)

    def test_intent_guidance_is_appended_after_grounding_rules(self):
        system = build_system_prompt("What are the steps to apply?", [])
        grounding_end = system.index("not a lawyer")
        checklist_start = system.index("numbered action list")
        self.assertLess(grounding_end, checklist_start)

    def test_different_intents_produce_different_system_prompts(self):
        checklist = build_system_prompt("What are the steps to apply?", [])
        appeal = build_system_prompt("I was denied — can I appeal?", [])
        eligibility = build_system_prompt("Am I eligible for FEMA?", [])
        documentation = build_system_prompt("What documents do I need?", [])
        general = build_system_prompt("Tell me about FEMA", [])
        prompts = {checklist, appeal, eligibility, documentation, general}
        # All five intents produce distinct system prompts
        self.assertEqual(len(prompts), 5)

    def test_system_prompt_sent_to_claude_uses_question_intent(self):
        """End-to-end: the generated answer request should include intent-specific text."""
        from knowledge.ingestion import ingest_document  # local import to keep test isolated
        with override_settings(RELAY_EMBEDDING_API_KEY=""):
            retrieval.invalidate()
            ingest_document(slug="test-appeals-us8", title="FEMA Appeals",
                            raw_text=APPEALS_TEXT, disaster_type="flood", phase="recovery")
        client = FakeClient([text("You can appeal within 60 days.", cite(0, 0, 1, "60 days"))])
        with patch("rag.generation.get_client", return_value=client), \
                override_settings(RELAY_EMBEDDING_API_KEY=""):
            retrieval.invalidate()
            answer_question("I was denied. How do I appeal?", [])
        system_prompt = client.answer_calls[0]["system"]
        # Appeal guidance should be present
        self.assertIn("denial letter", system_prompt)
        # Grounding rules should also be present
        self.assertIn("Use only the provided documents", system_prompt)
