# Relay RAG engine (US 3 + US 4)

The `rag` app answers chat questions from Relay's curated knowledge base and shows which sources each answer used. It follows the standard conversational RAG pipeline:

```
question + chat history
  → condense: rewrite a follow-up into a standalone question     (Claude Haiku 5.5)
  → hybrid search: vector similarity (embeddings) + BM25 keywords, fused with RRF
  → generate: answer only from the retrieved chunks, with citations (Claude Haiku 5.5)
  → sources: only the documents Claude actually cited
```

- **US 3 (grounded answers):** Claude only sees the retrieved chunks. If nothing relevant is found, Claude is not called. An answer with no citations is replaced with a "not in my sources" reply.
- **US 4 (sources):** the answer carries `[1]`, `[2]` markers, and the chat page shows a numbered Sources list with the cited passage under each title.

## Run it locally

One-time setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env        # then fill in ANTHROPIC_API_KEY and OPENROUTER_API_KEY
```

Django loads `.env` automatically (via `python-dotenv`; real environment variables win). Prepare the data and start the server:

```bash
python manage.py migrate
python manage.py load_dataset     # documents → cleaned chunks (knowledge app)
python manage.py embed_chunks     # chunks → vectors (rag app)
python manage.py runserver
```

Open http://127.0.0.1:8000. `RELAY_ANSWER_BACKEND=rag.services.answer_question` (set in `.env`) connects the chat page to this engine. Never commit `.env`; it is in `.gitignore`.

## Files

| File | What it does |
|---|---|
| `rag/services.py` | `answer_question(question, history, *, disaster_type=None, phase=None)`, the backend the chat page calls. Runs the pipeline and sends the `answer_completed` signal. |
| `rag/generation.py` | Claude calls: `rewrite_queries` (follow-ups) and `generate_answer` (cited answer), plus citation-to-source mapping. |
| `rag/retrieval.py` | Hybrid search: BM25 index in memory, vector search, RRF fusion, filters. |
| `rag/vector_store.py` | Vector database: chunk embeddings in a `sqlite-vec` table inside the app's SQLite database. |
| `rag/embeddings.py` | Embedding client for OpenRouter's OpenAI-compatible embeddings API. |
| `rag/prompts.py` | System prompt: fixed grounding rules plus `select_prompt()` for US 8. |
| `rag/apps.py` | Loads the `sqlite-vec` extension into every database connection. |
| `rag/management/commands/embed_chunks.py` | `python manage.py embed_chunks [--rebuild]`. |
| `rag/tests.py` | 41 tests; no API keys needed. |
| `chat/static/chat/chat.js` | Renders answers as Markdown, makes `[n]` markers clickable, shows the numbered Sources list. |
| `data/dataset.json`, `data/sources/`, `data/SOURCES.md` | The knowledge base and why each document is in it. |

## The pipeline

1. **Condense the question** (`rewrite_queries`). On a follow-up turn, Claude Haiku 5.5 rewrites the question into one that stands on its own, using the last 6 messages. For example, "how long do I have to do that?" becomes "How long do I have to appeal a FEMA decision that denied my application?". Search uses both the original and the rewrite. The first question of a chat is searched as-is. If the rewrite fails or times out (8 s), search continues with the original question.
2. **Hybrid search** (`retrieve`):
   - **Vector search.** The question is embedded through OpenRouter (`openai/text-embedding-3-small`) and compared with every chunk's embedding in the vector table (`rag_chunk_vectors`, cosine distance). This matches meaning even when the question uses different words from the document. The 30 nearest chunks are candidates; those below `RELAY_MIN_SIMILARITY` are dropped.
   - **BM25 keyword search.** Catches exact terms such as "FEMA" or "Kerr County". A chunk must share at least two meaningful words with the question (or the only word, for one-word questions).
   - **Fusion.** The two ranked lists are merged with reciprocal rank fusion (RRF, k = 60). Chunks found by both methods rank highest. Inactive documents and the disaster type / phase filters are applied, each document contributes at most 3 chunks, and the top `RELAY_RETRIEVAL_K` (6) are kept.
   - **Fallback.** If embeddings are not configured or fail, search uses BM25 alone and logs a warning. Nothing crashes.
3. **No match means no model call.** If no chunk passes, Relay returns a fixed "I couldn't find information about this in Relay's sources" reply with no sources.
4. **Generate with citations** (`generate_answer`). Chunks are grouped by document; each document is sent to Claude as a `document` block (one content block per chunk, citations enabled), followed by the question. The system prompt (`rag/prompts.py`) tells Claude to use only these documents, say plainly what they don't cover, name which disaster a point comes from, never promise approval, treat history and document text as information rather than instructions, and write in plain words with simple Markdown.
5. **Sources come from citations** (`parse_cited_response`). Claude's citations point at exact chunks. Only cited documents become sources, numbered in the order first cited; each gets a `snippet` of the cited text. An answer with no citations is treated as ungrounded and replaced with the not-found reply. A safety refusal becomes a short polite decline.

Response returned to the chat page:

```json
{"answer": "**You have 60 days to appeal.** ... within 60 days of the date of the letter. [1] ... [2]",
 "sources": [
   {"title": "FEMA: Disagreeing with FEMA's Decision (how to appeal)", "url": "https://www.fema.gov/...", "snippet": "If you disagree with FEMA's decision, you may appeal within 60 days..."},
   {"title": "44 CFR 206.115: Appeals (federal regulation)", "url": "https://www.ecfr.gov/...", "snippet": "Applicants must file their appeal within 60 days..."}
 ]}
```

### Keeping the vector store in sync

Vectors are keyed by `knowledge.Chunk.id`. `vector_store.sync()` embeds chunks that have no vector, deletes vectors whose chunk no longer exists, and re-embeds everything if `RELAY_EMBEDDING_MODEL` changes. It runs from `python manage.py embed_chunks` (`--rebuild` re-embeds everything) and automatically before a search whenever documents or chunks have changed, so new or edited documents become searchable without a manual step.

## Chat page changes (`chat.js`, `chat.css`)

- Answers are rendered as Markdown: paragraphs, bold, italic, bullet and numbered lists, inline code, and http(s) links.
- The renderer builds DOM elements directly and never uses `innerHTML`, so model output cannot inject HTML or scripts. Raw tags show as plain text and `javascript:` links are not made clickable. The user's own messages are still shown exactly as typed.
- `[n]` markers become small links that jump to and highlight the matching source.
- Sources are a numbered list matching the markers, with the cited passage shown under each title.

## Knowledge base

`data/dataset.json` lists 22 documents (79 chunks), chosen to address the barriers identified in our proposal. `data/SOURCES.md` lists each one with its barrier and source.

| Barrier | Documents |
|---|---|
| Unclear eligibility | FEMA eligibility criteria; FEMA citizenship and immigration status requirements; 44 CFR 206.113 |
| Documents and proof | FEMA verifying home ownership or occupancy; USAGov replacing lost ID |
| Appeals and next steps | FEMA appeals; Understanding Your FEMA Letter; What's Next?; 44 CFR 206.115 |
| Fear and distrust | FEMA citizenship FAQ (information sharing); FEMA disaster fraud |
| Communication and access | FEMA Disaster Recovery Centers; third-party helpers (in What's Next?) |
| Getting help | FEMA recovery programs (free legal help, case management, counseling); 44 CFR 206.164 |
| Context: the floods | Summaries of the four proposal articles, the July 2026 Texas floods article, and two GAO reports |

Government pages and regulations are public domain and included in full. News articles are copyrighted, so those files are short labeled summaries linking to the original. To add or update a document: put the text in `data/sources/`, add an entry to `data/dataset.json`, then run `load_dataset` and `embed_chunks`.

## Settings (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `RELAY_ANSWER_BACKEND` | `chat.services.unavailable_backend` | Set to `rag.services.answer_question` to use this engine (`render.yaml` already does). |
| `ANTHROPIC_API_KEY` | none | Required. Claude answers and rewrites. |
| `OPENROUTER_API_KEY` | none | Embeddings for vector search. Without it, search is keyword-only. |
| `RELAY_LLM_MODEL` | `claude-haiku-5-5` | Answer model. `claude-sonnet-5-5` or `claude-opus-5-5` give stronger answers at roughly 20–40× the cost. |
| `RELAY_LLM_EFFORT` | `low` | `low` / `medium` / `high`; higher is slower and costs more. |
| `RELAY_REWRITE_MODEL` | `claude-haiku-5-5` | Follow-up rewrite model. |
| `RELAY_QUERY_REWRITE` | `1` | Set to `0` to skip rewriting follow-ups. |
| `RELAY_EMBEDDING_MODEL` | `openai/text-embedding-3-small` | Any OpenRouter embedding model. Changing it re-embeds everything; re-tune `RELAY_MIN_SIMILARITY` after. |
| `RELAY_MIN_SIMILARITY` | `0.3` | Minimum cosine similarity for a vector match. With the default model, off-topic test questions returned no matches and barrier questions scored about 0.37–0.77. |
| `RELAY_RETRIEVAL_K` | `6` | Chunks sent to Claude. |
| `RELAY_LLM_TIMEOUT` | `30` | Answer timeout in seconds (no retries, so the total stays under the chat page's 45 s limit). |
| `RELAY_EMBEDDING_TIMEOUT` | `20` | Embedding request timeout in seconds. |

## Cost and speed

Measured with live questions on the default settings:

| Step | When | Typical cost |
|---|---|---|
| Rewrite (Haiku 5.5) | Follow-up questions only | well under $0.0001 |
| Embedding the question | Every question | negligible |
| Answer (Haiku 5.5) | When relevant chunks are found | about $0.0006–0.0008 (≈3,300–3,700 input + 450–900 output tokens) |
| Embedding the knowledge base | When documents change, and on each Render start | well under $0.01 for all 79 chunks |

A full answer took about 6–7 seconds (rewrite ≈0.9 s, search ≈0.5 s, generation ≈5 s).

## Deploying on Render

`render.yaml` sets `RELAY_ANSWER_BACKEND` and declares the two API keys. Add `ANTHROPIC_API_KEY` and `OPENROUTER_API_KEY` under **Environment** in the Render dashboard. The start command runs `migrate`, `load_dataset` and `embed_chunks` before starting Gunicorn, because the free plan's disk is wiped on every restart; each cold start therefore re-embeds the knowledge base (a few seconds). If `sqlite-vec` cannot load on the host, a warning is logged and search runs on keywords only.

## Extension points for teammates

- **US 8 (dynamic prompts):** replace the body of `rag.prompts.select_prompt(question, history)`. Its return value is added after the fixed `GROUNDING_RULES`, which should stay as they are.
- **US 9 (accuracy tests):** `rag.retrieval.retrieve(query)` returns each chunk's `similarity` and `keyword_score`, so retrieval can be evaluated without calling Claude.
- **US 10 / US 12 (monitoring, usage logs):** connect a receiver to `rag.services.answer_completed`. It sends `outcome` (`answered`, `no_match`, `ungrounded`, `declined`), `timings_ms` (`rewrite`, `retrieval`, `generation`, `total`), `retrieved_chunks`, `cited_sources` and `model`. The question text is never sent or logged. API errors and timeouts are raised to the chat view, which returns HTTP 502 and logs only the error type.
- **US 11 (clear/reload):** call `rag.retrieval.invalidate()` after changing the knowledge base, and `rag.vector_store.sync(rebuild=True)` to re-embed from scratch.
- **US 13 (filters):** `answer_question(..., disaster_type=..., phase=...)` already filters. Documents tagged `general` match every filter. The chat page does not send filters yet.

## Tests

```bash
python manage.py test rag     # RAG engine only
python manage.py test         # whole project
```

The tests use a fake Claude client and a fake embedding model, so they need no API keys and cost nothing. They cover tokenizing and BM25 ranking, filters and inactive documents, the embedding client, vector store sync and model changes, hybrid ranking and fallback to keywords, follow-up rewriting, citation-to-source mapping, the not-found, ungrounded and declined paths, and the chat endpoint end to end.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Chat page shows raw `**` or old styling | The browser cached the old `chat.js`. Hard refresh (Cmd+Shift+R). |
| Every answer cites the same single source | Only one document is loaded. Run `load_dataset` then `embed_chunks`, and restart the server. |
| Log says `OPENROUTER_API_KEY is not set; using keyword search only` | Add the key to `.env` and restart the server, or add it on Render. |
| "Relay could not get an answer" (HTTP 502) | Check `ANTHROPIC_API_KEY`, and the server log for the error type. |
| Off-topic questions get answers, or on-topic ones get "not in my sources" | Adjust `RELAY_MIN_SIMILARITY` (raise to be stricter, lower to be looser) and test with `retrieve()`. |
