"""Story 3: find the chunks of the knowledge base that can answer a question.

Hybrid search, the usual production RAG setup:

* Vector search (semantic): the question is embedded and compared with every
  chunk's embedding in the sqlite-vec vector table (rag/vector_store.py). This
  matches meaning even when the question uses different words.
* BM25 keyword search: exact terms such as "FEMA", "Kerr County" or "DANA",
  which embeddings can blur. Kept in memory; the corpus is small.

The two ranked lists are merged with reciprocal rank fusion (RRF). A chunk is
only a candidate if it is relevant to at least one method: cosine similarity
>= RELAY_MIN_SIMILARITY, or sharing enough meaningful words with the query.
If embeddings are not configured or fail, search falls back to BM25 alone.

    retrieve("Can I appeal a FEMA denial?", k=6, disaster_type="flood")
"""
import logging
import math
import re
import threading
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Optional

from django.conf import settings
from django.db.models import Count, Max

from knowledge.models import Chunk, Document

from . import embeddings, vector_store

logger = logging.getLogger(__name__)

DEFAULT_K = 6
MAX_CHUNKS_PER_DOCUMENT = 3
BM25_K1 = 1.5
BM25_B = 0.75
VECTOR_CANDIDATES = 30
DEFAULT_MIN_SIMILARITY = 0.3
RRF_K = 60  # standard reciprocal-rank-fusion constant

# Common English words that carry no meaning for search.
# Accent-free because tokens are accent-folded before this check.
STOPWORDS = frozenset("""
a about after all also am an and any are as at be been before being but by can could did do does
doing for from had has have having he her here hers him his how i if in into is it its just me
more most my no not now of on or our out over own she should so some than that the their them
then there these they this those through to too under until up very was we were what when where
which while who whom why will with would you your yours
""".split())

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _fold(text):
    """Lowercase and strip accents so "café" matches "cafe"."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _stem(token):
    # Light plural folding, applied to queries and chunks alike:
    # appeals/appeal and phases/phase reduce to the same token.
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        token = token[:-1]
    if len(token) > 3 and token.endswith("e"):
        token = token[:-1]
    return token


def tokenize(text):
    return [_stem(t) for t in _TOKEN_RE.findall(_fold(text)) if len(t) > 1 and t not in STOPWORDS]


@dataclass(frozen=True)
class ScoredChunk:
    chunk_id: int
    document_id: int
    document_slug: str
    title: str
    url: str
    disaster_type: str
    phase: str
    position: int
    text: str
    score: float                         # fused (RRF) score; higher is better
    similarity: Optional[float] = None   # cosine similarity, if found by vector search
    keyword_score: Optional[float] = None  # BM25 score, if found by keyword search


@dataclass
class _Entry:
    chunk_id: int
    document_id: int
    document_slug: str
    title: str
    url: str
    disaster_type: str
    phase: str
    position: int
    text: str
    term_counts: Counter
    length: int


class _Index:
    def __init__(self, entries):
        self.entries = entries
        self.doc_freq = Counter()
        for entry in entries:
            self.doc_freq.update(entry.term_counts.keys())
        self.avg_length = (sum(e.length for e in entries) / len(entries)) if entries else 0.0

    def idf(self, term):
        n, df = len(self.entries), self.doc_freq.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def score(self, entry, query_terms):
        """Return (BM25 score, number of distinct query terms the chunk contains)."""
        total, matched = 0.0, 0
        norm = BM25_K1 * (1 - BM25_B + BM25_B * entry.length / (self.avg_length or 1))
        for term in query_terms:
            tf = entry.term_counts.get(term, 0)
            if tf:
                matched += 1
                total += self.idf(term) * tf * (BM25_K1 + 1) / (tf + norm)
        return total, matched


_lock = threading.Lock()
_cache = {"fingerprint": None, "index": None}


def _fingerprint():
    """Changes whenever chunks are rebuilt or documents are edited/toggled.

    `rebuild_chunks` uses bulk_create, which fires no signals, so the index
    checks this cheap fingerprint on every query instead of listening for saves.
    """
    docs = Document.objects.aggregate(n=Count("id"), latest=Max("updated_at"))
    chunks = Chunk.objects.aggregate(n=Count("id"), last_id=Max("id"))
    return (docs["n"], docs["latest"], chunks["n"], chunks["last_id"])


def _build_index():
    entries = []
    rows = (Chunk.objects.filter(document__is_active=True)
            .select_related("document").order_by("document_id", "position"))
    for chunk in rows:
        doc = chunk.document
        # The title is indexed with every chunk so title words help ranking.
        terms = tokenize(f"{doc.title}\n{chunk.text}")
        entries.append(_Entry(
            chunk_id=chunk.id, document_id=doc.id, document_slug=doc.slug, title=doc.title,
            url=doc.source_url, disaster_type=doc.disaster_type, phase=doc.phase,
            position=chunk.position, text=chunk.text, term_counts=Counter(terms), length=len(terms),
        ))
    return _Index(entries)


def get_index():
    return _get_index_and_fingerprint()[0]


def _get_index_and_fingerprint():
    fingerprint = _fingerprint()
    with _lock:
        if _cache["index"] is None or _cache["fingerprint"] != fingerprint:
            _cache["index"] = _build_index()
            _cache["fingerprint"] = fingerprint
        return _cache["index"], fingerprint


def invalidate():
    """Drop cached search state. Call after clearing or reloading the knowledge base (US 11)."""
    with _lock:
        _cache["index"] = None
        _cache["fingerprint"] = None
    vector_store.forget_sync()


def _min_matches(query_terms):
    # A chunk must share at least two distinct terms with the query (or the
    # only term, for one-word queries). One shared generic word like "flood"
    # is not evidence that the chunk answers the question.
    return min(2, len(query_terms))


_warned = set()


def _warn_once(key, message, *args):
    if key not in _warned:
        _warned.add(key)
        logger.warning(message, *args)


def _vector_similarities(queries, fingerprint):
    """Return {chunk_id: best cosine similarity}, or {} if vector search is unavailable."""
    if not embeddings.is_configured():
        _warn_once("no-key", "OPENROUTER_API_KEY is not set; using keyword search only.")
        return {}
    try:
        vector_store.sync_if_changed(fingerprint)
        return vector_store.search(embeddings.embed_texts(queries), limit=VECTOR_CANDIDATES)
    except vector_store.VectorStoreUnavailable:
        _warn_once("no-vec", "sqlite-vec is not available; using keyword search only.")
    except embeddings.EmbeddingError as exc:
        logger.warning("Vector search failed, using keyword search only: %s", exc)
    return {}


def _keyword_scores(index, term_sets, allowed):
    scores = {}
    for chunk_id, entry in allowed.items():
        for terms in term_sets:
            score, matched = index.score(entry, terms)
            if matched >= _min_matches(terms) and score > scores.get(chunk_id, 0.0):
                scores[chunk_id] = score
    return scores


def _passes_filters(entry, disaster_type, phase):
    if disaster_type and entry.disaster_type not in (disaster_type, Document.DisasterType.GENERAL):
        return False
    if phase and entry.phase not in (phase, Document.Phase.GENERAL):
        return False
    return True


def retrieve(queries, *, k=DEFAULT_K, disaster_type=None, phase=None,
             max_per_document=MAX_CHUNKS_PER_DOCUMENT):
    """Return up to `k` chunks ranked by relevance, best first.

    `queries` is one string or several phrasings of the same question (for
    example the original follow-up and its standalone rewrite); each chunk
    keeps its best score across them. `disaster_type` / `phase` restrict
    results to matching documents; documents tagged "general" always match
    because they apply to every situation (US 13). Inactive documents are
    never returned.
    """
    if isinstance(queries, str):
        queries = [queries]
    queries = [q.strip() for q in queries if q and q.strip()]
    term_sets = [terms for terms in (list(dict.fromkeys(tokenize(q))) for q in queries) if terms]
    if not queries or k <= 0:
        return []

    index, fingerprint = _get_index_and_fingerprint()
    allowed = {e.chunk_id: e for e in index.entries if _passes_filters(e, disaster_type, phase)}
    if not allowed:
        return []

    keyword = _keyword_scores(index, term_sets, allowed)
    min_similarity = getattr(settings, "RELAY_MIN_SIMILARITY", DEFAULT_MIN_SIMILARITY)
    similarity = {cid: sim for cid, sim in _vector_similarities(queries, fingerprint).items()
                  if cid in allowed and sim >= min_similarity}

    fused = Counter()
    for scores in (keyword, similarity):
        for rank, chunk_id in enumerate(sorted(scores, key=scores.get, reverse=True), start=1):
            fused[chunk_id] += 1.0 / (RRF_K + rank)

    ranked = sorted(fused, key=lambda cid: (-fused[cid], allowed[cid].document_id, allowed[cid].position))
    results, per_document = [], Counter()
    for chunk_id in ranked:
        entry = allowed[chunk_id]
        if per_document[entry.document_id] >= max_per_document:
            continue
        per_document[entry.document_id] += 1
        results.append(ScoredChunk(
            chunk_id=entry.chunk_id, document_id=entry.document_id, document_slug=entry.document_slug,
            title=entry.title, url=entry.url, disaster_type=entry.disaster_type, phase=entry.phase,
            position=entry.position, text=entry.text, score=round(fused[chunk_id], 6),
            similarity=round(similarity[chunk_id], 4) if chunk_id in similarity else None,
            keyword_score=round(keyword[chunk_id], 4) if chunk_id in keyword else None,
        ))
        if len(results) == k:
            break
    return results
