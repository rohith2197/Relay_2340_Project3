"""Story 7: clean and preprocess text before it enters the knowledge base.

Cleaning is deliberately conservative: it only removes whole lines that are
clearly page furniture (menus, cookie banners, page numbers). It never deletes
repeated content lines, because "Call 911" repeated in a checklist is real content.
"""
import hashlib
import re
import unicodedata
from html.parser import HTMLParser

MIN_CLEAN_CHARS = 100
DEFAULT_CHUNK_CHARS = 1000
DEFAULT_CHUNK_OVERLAP = 150


class PreprocessingError(ValueError):
    """The text cannot be turned into usable clean text."""


_HTML_TAG_RE = re.compile(r"<\s*/?\s*[a-zA-Z][^<>]*>")


class _HTMLTextExtractor(HTMLParser):
    _SKIP = {"script", "style", "noscript", "nav", "footer", "aside", "form", "iframe", "svg", "template"}
    _PARAGRAPH = {"p", "div", "section", "article", "blockquote", "table", "ul", "ol",
                  "h1", "h2", "h3", "h4", "h5", "h6", "header", "main", "pre"}
    _LINE = {"br", "li", "tr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip_depth += 1
        elif not self._skip_depth:
            if tag in self._PARAGRAPH or tag in self._LINE:
                self.parts.append("\n")
            if tag == "li":
                self.parts.append("- ")

    def handle_endtag(self, tag):
        if tag in self._SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif not self._skip_depth:
            if tag in self._PARAGRAPH:
                self.parts.append("\n\n")
            elif tag in self._LINE:
                self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip_depth:
            self.parts.append(re.sub(r"\s+", " ", data))


def _html_to_text(raw):
    parser = _HTMLTextExtractor()
    parser.feed(raw)
    parser.close()
    return "".join(parser.parts)


_INVISIBLE_RE = re.compile("[\u00ad\u200b\u200c\u200d\u2060\ufeff]")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0e-\x1f\x7f]")
_HYPHEN_BREAK_RE = re.compile(r"(\w)-[ \t]*\n[ \t]*(\w)")

_BOILERPLATE_PATTERNS = [
    r"advertisement|advertising|sponsored content",
    r"related (?:articles|stories|posts|coverage)",
    r"(?:read|see) (?:more|also)",
    r"continue reading",
    r"share(?: this)?(?: article| story| page)?(?: on (?:facebook|twitter|x|linkedin|whatsapp|email))?",
    r"tweet|pin it|print",
    r"follow us(?: on .{1,30})?",
    r"(?:sign up|subscribe)(?: now| today)?",
    r"(?:sign up|subscribe) (?:for|to) (?:our |the )?(?:free )?newsletters?.{0,60}",
    r"skip to (?:main )?content",
    r"(?:main )?menu|search|log ?in|sign ?in",
    r"(?:we use|this (?:site|website) uses) cookies.{0,200}",
    r"(?:accept|manage|reject)(?: all)? cookies?(?: settings| preferences)?",
    r"cookies?(?: policy| settings| notice| preferences)",
    r"privacy policy|terms (?:of (?:use|service)|and conditions)",
    r"all rights reserved.{0,100}",
    r"copyright\s*(?:©|\(c\)|\d{4}).{0,100}",
    r"©.{0,100}|\(c\) ?\d{4}.{0,100}",
    r"(?:page )?\d{1,3}(?: ?(?:of|/) ?\d{1,3})?",
    # Spanish
    r"publicidad|compartir|leer m[aá]s|lee tambi[eé]n|te puede interesar",
    r"suscr[ií]bete(?: .{1,60})?",
    r"pol[ií]tica de (?:cookies|privacidad)|aviso legal",
    r"todos los derechos reservados.{0,100}",
]
_BOILERPLATE_RE = re.compile(
    r"(?:" + "|".join(_BOILERPLATE_PATTERNS) + r")[\s.:!|>»]*", re.IGNORECASE
)


def _join_hyphen(match):
    before, after = match.group(1), match.group(2)
    return before + after if after.islower() and before.isalpha() else f"{before}-{after}"


def clean_text(raw, *, min_chars=MIN_CLEAN_CHARS):
    """Return cleaned text, or raise PreprocessingError if too little is left."""
    if not isinstance(raw, str):
        raise PreprocessingError("Document text must be a string.")

    text = raw
    if _HTML_TAG_RE.search(text):
        text = _html_to_text(text)

    for old in ("\r\n", "\r", "\x0b", "\x0c"):
        text = text.replace(old, "\n")
    text = unicodedata.normalize("NFKC", text)
    text = _INVISIBLE_RE.sub("", text)
    text = _CONTROL_RE.sub("", text)
    text = _HYPHEN_BREAK_RE.sub(_join_hyphen, text)

    lines = []
    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()
        if line and _BOILERPLATE_RE.fullmatch(line):
            continue
        if line and lines and line == lines[-1]:
            continue
        lines.append(line)

    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    if len(text) < min_chars:
        raise PreprocessingError(
            f"Only {len(text)} characters of usable text remain after cleaning "
            f"(minimum {min_chars}). Is the source empty or all page furniture?"
        )
    return text


def content_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+")


def _split_long(text, max_chars):
    pieces, current = [], ""
    for word in text.split():
        while len(word) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            pieces.append(word[:max_chars])
            word = word[max_chars:]
        candidate = f"{current} {word}" if current else word
        if len(candidate) <= max_chars:
            current = candidate
        else:
            pieces.append(current)
            current = word
    if current:
        pieces.append(current)
    return pieces


def _units(text, max_chars):
    units = []
    for paragraph in re.split(r"\n{2,}", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= max_chars:
            units.append(paragraph)
            continue
        for line in paragraph.split("\n"):
            line = line.strip()
            if len(line) <= max_chars:
                if line:
                    units.append(line)
                continue
            for sentence in _SENTENCE_RE.split(line):
                if len(sentence) <= max_chars:
                    units.append(sentence)
                else:
                    units.extend(_split_long(sentence, max_chars))
    return units


def chunk_text(text, *, max_chars=DEFAULT_CHUNK_CHARS, overlap=DEFAULT_CHUNK_OVERLAP):
    """Split cleaned text into overlapping chunks (up to max_chars + overlap long)."""
    if max_chars < 100:
        raise ValueError("max_chars must be at least 100.")
    if not 0 <= overlap < max_chars:
        raise ValueError("overlap must be between 0 and max_chars - 1.")

    chunks, current = [], ""
    for unit in _units(text, max_chars):
        candidate = f"{current}\n\n{unit}" if current else unit
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current)
            current = unit
    if current:
        chunks.append(current)

    if overlap and len(chunks) > 1:
        overlapped = [chunks[0]]
        for previous, chunk in zip(chunks, chunks[1:]):
            tail = previous[-overlap:]
            if len(previous) > overlap and " " in tail:
                tail = tail.split(" ", 1)[1]
            overlapped.append(f"{tail}\n{chunk}")
        chunks = overlapped
    return chunks