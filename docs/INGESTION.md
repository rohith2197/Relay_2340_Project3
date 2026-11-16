# US 6: local files/text and URL scraping

These are two acquisition methods: developer-supplied local text/files through
`load_dataset`, and HTTP retrieval of public pages through `ingest_url`. Different
file formats are not counted as separate methods. Both reuse the existing
cleaning, validation, document/chunk storage and retrieval pipeline.

## Local demonstration

Use a disposable local checkout/database with the project's dependencies installed.
Do not run these examples against a production knowledge base. From the project
root, prepare the local database:

```sh
python manage.py migrate
```

Method 1, existing file ingestion (the committed manifest points to text files):

```sh
python manage.py load_dataset --manifest data/dataset.json --dry-run
python manage.py load_dataset --manifest data/dataset.json
```

For inline text instead, create `/tmp/relay-us6-text.json` containing:

```json
{"documents":[{"slug":"demo-us6-text","title":"US 6 local text example","text":"This is a local ingestion demonstration document. It verifies that developer-supplied text enters the shared cleaning and chunking workflow. It is test content, not disaster recovery guidance.","disaster_type":"general","phase":"general"}]}
```

```sh
python manage.py load_dataset --manifest /tmp/relay-us6-text.json
```

Method 2, real HTTP scraping (a live source may deny automated requests or change):

```sh
python manage.py ingest_url 'https://www.fema.gov/assistance/individual/after-applying/appeals' --slug demo-us6-web-appeals --title 'FEMA appeals (URL demo)' --disaster-type general --phase recovery
```

Required arguments are the URL, `--slug` and `--title`. Disaster type and phase
default to `general`; accepted values are the existing Document choices. The
final URL after redirects is stored as source attribution. Repeat the identical
command: unchanged cleaned content keeps the existing chunk IDs; changed content
rebuilds chunks while keeping the document identity. Metadata changes report
`UPDATED`. Existing inactive documents remain inactive.

A slug already used by a non-scraped document or a different final URL is rejected.
Choose a new slug for a different source; there is deliberately no force-overwrite
flag. If a site's redirect target changes, use a new slug or coordinate an explicit
source migration. Keep scraped slugs separate from manifest slugs, because the
existing manifest loader owns its listed slugs. Run ingestion commands serially.

Inspect storage and real keyword retrieval without calling any provider:

```sh
OPENROUTER_API_KEY='' python manage.py shell -c "from knowledge.models import Document; from rag.retrieval import retrieve; print(list(Document.objects.filter(slug__startswith='demo-us6-').values('slug', 'source_type', 'source_url'))); print([(c.document_slug, c.text[:160]) for c in retrieve('FEMA appeals documents')])"
```

Keyword retrieval refreshes on the next query. To prepare vector retrieval, run
the existing separate command with `OPENROUTER_API_KEY` configured:

```sh
python manage.py embed_chunks
```

That step makes provider calls and may incur charges. It is unnecessary for the
keyword-only demonstration. Normal RAG queries can also sync missing vectors
automatically. US 6 does not change chat generation or claim response accuracy.

## Fetch limits

Only public HTTP/HTTPS destinations are allowed. Every redirect is resolved and
checked before connecting; mixed public/private DNS results are rejected. The
connection uses the checked IP directly, while HTTPS verifies the original host.
Environment proxies are not used. Private, loopback, link-local, reserved,
multicast and IPv6 transition addresses are blocked.

Limits: five redirects, 2,000,000 response bytes, ten-second socket-operation
timeout. DNS uses the operating system resolver timeout; the socket timeout is
not a total elapsed-time deadline. Supported responses are HTML, XHTML and plain
text with the declared charset (UTF-8 by default). Compressed responses, PDFs,
JavaScript-rendered pages, authentication and browser interactions are unsupported.
Existing preprocessing requires at least 100 usable characters after cleaning.
Fetch/validation errors leave document data unchanged.

## Verification

```sh
python manage.py test knowledge.test_scraping
python manage.py test
```

The focused tests mock DNS and HTTP, and use Django's isolated test database.
They exercise fetching policy, failures, redirects, ingestion, repeat/update
semantics, stored chunks and actual keyword retrieval. External embeddings are
disabled; the tests make no live requests or paid provider calls. The commands
above are a separate live demonstration, not evidence that live fetching or
chatbot answer accuracy was tested.
