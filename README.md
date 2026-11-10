# Relay — US 1 and US 2

Django chat interface for disaster recovery questions. This repository includes the web UI, a CSRF-protected question endpoint, a backend integration boundary, tests, and Render configuration.

**Status:** local foundation implemented. US 1 is complete only after the public site is deployed and checked. US 2 is complete only after the teammate's RAG answer function is connected and real questions receive relevant responses. The default service returns HTTP 503 with an honest setup message; it does not fabricate assistance advice.

## 1. Open and run on your Mac

In VS Code, choose **File → Open Folder** and select `Relay_2340_Project3`. Choose **Terminal → New Terminal**, then run:

```bash
cd /Users/harishv/CS2340_projects/Relay_2340_Project3
source .venv/bin/activate
python manage.py migrate
python manage.py runserver
```

The virtual environment and dependencies were installed during setup. For another teammate or a fresh clone, run these before the commands above:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Open http://127.0.0.1:8000. Stop the server with Control+C. In VS Code, use **Python: Select Interpreter** from the command palette and select `.venv/bin/python`.

## 2. Understand your code

- `config/settings.py`: Django configuration, environment variables, and production security.
- `config/urls.py` and `chat/urls.py`: map `/`, `/api/chat/`, and `/health/` to Python views.
- `chat/templates/chat/index.html`: chat page and form.
- `chat/static/chat/chat.css`: responsive Relay styling.
- `chat/static/chat/chat.js`: sends questions, shows answers/errors, manages the last 20 history messages in memory, and starts a new chat.
- `chat/views.py`: validates JSON and calls the answer service without logging question contents.
- `chat/services.py`: configurable boundary to the teammate's RAG implementation.
- `chat/tests.py`: request, validation, CSRF, backend integration, and failure tests.
- `render.yaml`: free Render web-service configuration.

Request flow: browser → Django `/api/chat/` → `get_answer()` → teammate's callable → JSON → browser. The browser renders all answer text as plain text, so model/user content cannot inject HTML.

No account is needed. Chat history is not saved to the database, cookies, or localStorage. Refreshing clears it. The backend receives history for follow-up context; your teammate must use that context appropriately and treat it as untrusted input.

## 3. Connect your teammate's backend

Agree on this interface now. Your teammate implements a callable such as `rag.services.answer_question`:

```python
# Interface example, not a complete retrieval implementation.
def answer_question(question, history):
    # Retrieve relevant approved documents and generate a grounded answer here.
    # Return the actual generated answer and its actual source documents.
    return {"answer": generated_answer, "sources": retrieved_sources}
```

The two names in the return example must be produced by the teammate's real implementation. A source is `{"title": "Document title", "url": "https://..."}`; URL may be omitted for a title-only source. `sources` may be an empty list if there are no supporting documents. The backend must explain missing evidence rather than invent information. Set explicit timeouts on model/network calls so the overall answer completes in under 45 seconds.

To connect it locally, stop the server and run:

```bash
export RELAY_ANSWER_BACKEND=rag.services.answer_question
python manage.py runserver
```

Use the actual Python module/function path your teammate chooses. On Render, add the same variable under **Environment**. Keep provider keys server-side in environment variables. `.env.example` is a reference; the app does not automatically load `.env` files.

API request:

```json
{"question":"What documents might I need after a flood?","history":[]}
```

API response:

```json
{"answer":"The actual source-grounded answer goes here.","sources":[{"title":"Actual source title","url":"https://example.com/source"}]}
```

These JSON strings illustrate the contract, not assistance advice. Sending success responses in automated tests uses a mocked backend; those tests do not verify RAG accuracy.

## 4. Check the app

```bash
python manage.py check
python manage.py test
python manage.py collectstatic --noinput
```

Manual checks:

1. Open the page on a desktop and a phone-width browser.
2. Type a normal question and press Enter or Send. Shift+Enter inserts a newline.
3. Before integration, verify the setup error appears and the typed question is preserved.
4. After integration, verify a real answer appears and a follow-up reaches the backend with history.
5. Verify empty questions cannot be submitted, and loading prevents duplicate submission.
6. Verify an unavailable backend shows a friendly error and lets you retry.
7. Click New chat and confirm the conversation clears.
8. Check all of these again on the public hosted URL before claiming the stories are done.

## 5. Deploy to Render's free web service

Render is the selected starter host because it runs this Django service directly and connects to your private GitHub repository. Its free web services sleep after 15 minutes of inactivity and the first request after sleep can be slow. Files on a free service are ephemeral; do not keep your team's future knowledge base or persistent database on its local disk. The current chat does not need persistent database storage. Web hosting being free does not make an external model API free.

Official references: https://render.com/docs/free and https://render.com/docs/deploy-django

1. Run the tests above.
2. Review files with `git status` and `git diff`. `.gitignore` excludes `.venv`, local databases, and secrets.
3. Commit and push to your team's **private** GitHub repository. Verify the remote with `git remote -v` first. Do not create another repository if your team already has one.
4. Sign into Render and connect the GitHub account with access to that repository.
5. Create a **Blueprint** from the repository. Render reads `render.yaml` and proposes a free Python web service. Check that the service plan says **Free** before creating it.
6. The configuration generates `DJANGO_SECRET_KEY`, disables debug, collects static assets, and runs Gunicorn. Django accepts the hostname supplied by Render in `RENDER_EXTERNAL_HOSTNAME`.
7. If using the manual **New Web Service** flow instead, use:
   - Build: `pip install -r requirements.txt && python manage.py collectstatic --noinput`
   - Start: `gunicorn config.wsgi:application --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 60`
   - Plan: Free; health path: `/health/`; environment: `DJANGO_DEBUG=0` and a randomly generated `DJANGO_SECRET_KEY`.
8. Add `RELAY_ANSWER_BACKEND` and any server-side provider keys once the real answer service is ready. Do not paste secret values into GitHub or chat.
9. Deploy and open the HTTPS URL shown in Render. Verify the chat, CSS, question request, response, and error states.
10. Share that public application URL with your TA. Keep the source repository private and grant the TA repository access as required.

If using a custom domain, set `DJANGO_ALLOWED_HOSTS` to the hostname without `https://`. For a future persistent database or uploads, have the relevant teammate configure external storage before adding those features; SQLite on Render's ephemeral filesystem is not durable.

## 6. What remains

- Teammate: implement approved-document retrieval and grounded generation (US 3) and configure the callable.
- Team/account owner: connect the private repository to Render and deploy.
- You: demonstrate the hosted app, natural-language questions, follow-ups, and error handling with the real backend.
- Other teammates retain ownership of ingestion, filtering, checklists, admin metrics, and logs. This starter does not claim to implement those stories.

No Git commit, remote push, or public deployment was performed by the assistant during initial setup.
