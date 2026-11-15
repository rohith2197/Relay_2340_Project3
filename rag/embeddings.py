"""Text embeddings through OpenRouter's OpenAI-compatible embeddings API.

    embed_texts(["How do I appeal?", "What documents do I need?"]) -> [[0.01, ...], [0.02, ...]]

Questions and passages with similar meaning get similar vectors, even when
they use different words. Change RELAY_EMBEDDING_MODEL to use another OpenRouter model; the
vector store re-embeds everything automatically when the model changes.
"""
import json
import urllib.error
import urllib.request

from django.conf import settings

BATCH_SIZE = 64


class EmbeddingUnavailable(Exception):
    """Embeddings are not configured (no API key)."""


class EmbeddingError(Exception):
    """The embeddings API call failed or returned something unusable."""


def embedding_model():
    return getattr(settings, "RELAY_EMBEDDING_MODEL", "openai/text-embedding-3-small")


def is_configured():
    return bool(getattr(settings, "RELAY_EMBEDDING_API_KEY", ""))


def _post(batch):
    body = json.dumps({"model": embedding_model(), "input": batch, "encoding_format": "float"}).encode()
    request = urllib.request.Request(
        getattr(settings, "RELAY_EMBEDDING_URL", "https://openrouter.ai/api/v1/embeddings"),
        data=body, method="POST",
        headers={"Authorization": f"Bearer {settings.RELAY_EMBEDDING_API_KEY}",
                 "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=getattr(settings, "RELAY_EMBEDDING_TIMEOUT", 20.0)) as response:
        return json.load(response)


def embed_texts(texts):
    """Return one embedding (list of floats) per input text, in input order."""
    if not is_configured():
        raise EmbeddingUnavailable("Set OPENROUTER_API_KEY to enable vector search.")
    vectors = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = [t if t.strip() else "(empty)" for t in texts[start:start + BATCH_SIZE]]
        try:
            data = _post(batch)
            items = sorted(data["data"], key=lambda item: item["index"])
            batch_vectors = [[float(x) for x in item["embedding"]] for item in items]
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError, TypeError) as exc:
            # Only the exception type: response bodies and URLs are not logged.
            raise EmbeddingError(f"Embedding request failed ({type(exc).__name__}).") from exc
        if len(batch_vectors) != len(batch) or not batch_vectors[0]:
            raise EmbeddingError("Embedding response did not match the request.")
        vectors.extend(batch_vectors)
    if len({len(v) for v in vectors}) > 1:
        raise EmbeddingError("Embedding response had inconsistent dimensions.")
    return vectors
