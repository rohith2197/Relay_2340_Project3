# US 9: known-question evaluation

`data/evaluation/questions.json` contains 12 cases: ten source-backed questions
(including one follow-up with fixed history) and two unsupported requests. Each
supported case records expected facts, source slug/file/URL and an evidence
excerpt from the committed corpus. Unsupported cases explain the missing evidence.
These are expectations about this corpus, not independently verified current policy.

The reference answers are **not** in `data/dataset.json` and must never be ingested.
The evaluator sends only the question and optional history to
`chat.services.get_answer`, the same configured backend boundary as the chat view.
It does not inject reference facts, alter prompts, or change retrieval.

## Offline tests (no API calls)

With requirements installed in `.venv`:

```sh
OPENROUTER_API_KEY='' ANTHROPIC_API_KEY='' .venv/bin/python manage.py test knowledge.test_scraping rag.test_evaluation
OPENROUTER_API_KEY='' ANTHROPIC_API_KEY='' .venv/bin/python manage.py test
```

Django creates and destroys an isolated test database. The evaluator tests supply
fabricated backend answers to check routing, scoring, source references, error
reporting and reference isolation. Passing these tests is **not chatbot accuracy**.

## Live evaluation (separate, opt-in)

Use a disposable local copy of the project with its own `db.sqlite3`, never a
production database. The evaluator does not ingest or clear documents. The actual
RAG backend retains its existing behavior: it can sync vectors and write usage
logs, which is why the database must be disposable. Keep existing source data and
monitoring untouched. Do not point `load_dataset` at the evaluation JSON.

From that disposable project's root, with its dependencies and credentials set:

```sh
source .venv/bin/activate
python manage.py migrate
python manage.py load_dataset --manifest data/dataset.json
python manage.py embed_chunks
RELAY_ANSWER_BACKEND=rag.services.answer_question python manage.py evaluate_rag --live > /tmp/relay-evaluation.json
```

`embed_chunks` needs `OPENROUTER_API_KEY`; generation needs `ANTHROPIC_API_KEY`.
These commands make real provider calls and may cost money. They were not run as
part of offline verification. For an explicitly keyword-only evaluation, omit
`embed_chunks` and set `OPENROUTER_API_KEY=''` on the evaluation command; generation
still calls the real model. Do not compare that run as if it used vector retrieval.

For a single case or follow-up:

```sh
RELAY_ANSWER_BACKEND=rag.services.answer_question python manage.py evaluate_rag --live --case appeal-deadline --case appeal-followup > /tmp/relay-evaluation-small.json
```

`--questions PATH` selects a different validated reference set. Without `--live`,
the command refuses to call a backend. An unconfigured backend or empty active
corpus is rejected. stdout is a JSON report containing answers, returned sources,
reference facts, checks and a summary. Errors are recorded by type (without
potentially sensitive exception messages); other cases continue, then the command
exits nonzero. Failed heuristic checks alone do not cause nonzero exit.

## Scoring and manual review

- Each expected fact has alternative phrases. A case passes that fact's
  **keyword heuristic** if at least one appears, case-insensitively with word
  boundaries and normalized whitespace. This does not understand meaning or negation.
- Supported cases also require at least one returned URL to match a reference URL
  (ignoring trailing slashes). This checks attribution overlap, not entailment.
- Unsupported cases look for abstention wording. They may cite general guidance;
  the reviewer must check they did not invent a personal decision or other missing fact.
- A heuristic pass requires all checks. The summary reports passes / all attempted
  cases, with backend errors counted separately and retained in the denominator.
  **This percentage is not response accuracy.** Every result requires manual review.

For each answer, inspect the referenced source passages and mark correct,
partially correct or incorrect: does it cover the expected facts, preserve
qualifications, avoid contradictions/unsupported claims, cite supporting passages,
and abstain when necessary? Record reasons alongside the saved report. If reporting
manual accuracy, use fully correct cases / all attempted cases and disclose errors
and partial answers separately. A keyword-matching wrong answer must not count as
correct. Fixed follow-up history tests context resolution, not an entire live dialogue.

The report includes question/corpus hashes, configured model/backend and retrieval
settings for comparison. Model settings do not prove which fallback model ran or
that vector retrieval succeeded; inspect runtime logs for fallback. Save the Git
revision and report with manual findings. Re-run after corpus/model changes.
No actual response accuracy result exists until a live run is reviewed.
