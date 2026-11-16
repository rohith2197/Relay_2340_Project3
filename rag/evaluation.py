"""US 9 offline heuristics, not a semantic accuracy judge."""
import json
import re
import unicodedata
from pathlib import Path


def load_cases(path):
    """Validate the whole reference set before making any backend calls."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(cases, list) or not cases:
        raise ValueError("Evaluation file must contain a nonempty cases list.")
    ids = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Each evaluation case must be an object.")
        for key in ("id", "question", "manual_review"):
            if not isinstance(case.get(key), str) or not case[key].strip():
                raise ValueError(f"Each case needs a nonempty {key}.")
        if case["id"] in ids:
            raise ValueError(f"Duplicate case id: {case['id']}")
        ids.add(case["id"])
        if not isinstance(case.get("unsupported"), bool):
            raise ValueError("unsupported must be true or false.")
        history = case.get("history", [])
        if not isinstance(history, list) or any(
            not isinstance(m, dict) or m.get("role") not in ("user", "assistant")
            or not isinstance(m.get("content"), str) or not m["content"].strip() for m in history
        ):
            raise ValueError("history must contain user/assistant messages with text.")
        facts = case.get("expected_facts")
        if not isinstance(facts, list) or not facts:
            raise ValueError("Each case needs expected_facts.")
        for fact in facts:
            if (not isinstance(fact, dict) or not isinstance(fact.get("fact"), str)
                    or not fact["fact"].strip() or not isinstance(fact.get("any_keywords"), list)
                    or not fact["any_keywords"]
                    or any(not isinstance(w, str) or not w.strip() for w in fact["any_keywords"])):
                raise ValueError("Each expected fact needs text and nonempty any_keywords.")
        sources = case.get("sources")
        if not isinstance(sources, list) or (not case["unsupported"] and not sources):
            raise ValueError("Answerable cases need source references.")
        for source in sources:
            if not isinstance(source, dict) or any(
                not isinstance(source.get(key), str) or not source[key].strip()
                for key in ("slug", "file", "url", "evidence")
            ):
                raise ValueError("Source references need slug, file, url and evidence.")
    return cases


def _normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().replace("’", "'").split())


def score_answer(case, result):
    """Count phrase checks and attribution overlap; all answers require human review."""
    answer = _normalize(result["answer"])
    checks = []
    for fact in case["expected_facts"]:
        matched = [word for word in fact["any_keywords"]
                   if re.search(r"(?<!\w)" + re.escape(_normalize(word)) + r"(?!\w)", answer)]
        checks.append({"kind": "keyword_heuristic", "expected": fact["fact"],
                       "passed": bool(matched), "matched_keywords": matched})
    if case["sources"]:
        expected_urls = {s["url"].rstrip("/") for s in case["sources"]}
        actual_urls = {s.get("url", "").rstrip("/") for s in result["sources"]}
        checks.append({"kind": "source_url_overlap", "passed": bool(expected_urls & actual_urls),
                       "expected_any_url": sorted(expected_urls)})
    return {"checks": checks, "heuristic_pass": all(c["passed"] for c in checks),
            "manual_review_required": True, "manual_review": case["manual_review"]}


def evaluate(cases, backend):
    """Call only with questions/history, never expected facts or answers."""
    results = []
    for case in cases:
        record = {"id": case["id"], "question": case["question"], "history": case.get("history", []),
                  "unsupported": case["unsupported"], "reference_sources": case["sources"],
                  "expected_facts": case["expected_facts"]}
        try:
            answer = backend(question=case["question"], history=case.get("history", []))
            record.update(answer=answer["answer"], sources=answer["sources"], **score_answer(case, answer))
        except Exception as exc:
            # Keep going so one provider failure does not hide the other cases.
            # Exception text can contain credentials or provider request details.
            record.update(error=type(exc).__name__, heuristic_pass=False, checks=[],
                          manual_review_required=True, manual_review=case["manual_review"])
        results.append(record)
    errors = sum("error" in row for row in results)
    passed = sum(row["heuristic_pass"] for row in results)
    return {"results": results, "summary": {
        "total": len(results), "answered": len(results) - errors, "errors": errors,
        "heuristic_passes": passed, "heuristic_failures": len(results) - errors - passed,
        "heuristic_pass_rate": passed / len(results) if results else None,
        "manual_review_required": len(results),
        "accuracy": "Not measured: keyword and URL checks are heuristics; review semantic correctness manually.",
    }}
