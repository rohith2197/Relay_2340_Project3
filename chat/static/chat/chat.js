const form = document.querySelector("#chat-form");
const input = document.querySelector("#question");
const messages = document.querySelector("#messages");
const status = document.querySelector("#status");
const send = document.querySelector("#send");
const clear = document.querySelector("#clear");
const suggestions = document.querySelector("#suggestions");
const welcome = messages.innerHTML;
let history = [];
let busy = false;

let messageCount = 0;

// ---------------------------------------------------------------------------
// Minimal Markdown renderer for assistant answers.
// Builds DOM nodes directly (never innerHTML), so model output can format text
// but cannot inject HTML or scripts. Supports paragraphs, line breaks,
// headings, "-"/"*" and numbered lists, blockquotes, code, **bold**, *italic*,
// http(s) links, and [n] citation markers that link to the numbered sources.
// ---------------------------------------------------------------------------

const INLINE = /(`[^`\n]+`)|(\*\*(?=\S)[\s\S]*?\S\*\*|__(?=\S)[\s\S]*?\S__)|(\*(?=[^\s*])[^*\n]*?[^\s*]\*|\*[^\s*]\*)|\[(\d{1,2})\]|\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)/g;

function safeUrl(value) {
  try {
    const url = new URL(value);
    return ["https:", "http:"].includes(url.protocol) ? url : null;
  } catch { return null; }
}

function renderInline(text, cite) {
  const fragment = document.createDocumentFragment();
  let last = 0;
  for (const match of text.matchAll(INLINE)) {
    if (match.index > last) fragment.append(text.slice(last, match.index));
    last = match.index + match[0].length;
    const [whole, code, bold, italic, citation, linkText, linkUrl] = match;
    if (code) {
      const el = document.createElement("code");
      el.textContent = code.slice(1, -1);
      fragment.append(el);
    } else if (bold) {
      const el = document.createElement("strong");
      el.append(renderInline(bold.slice(2, -2), cite));
      fragment.append(el);
    } else if (italic) {
      const el = document.createElement("em");
      el.append(renderInline(italic.slice(1, -1), cite));
      fragment.append(el);
    } else if (citation) {
      fragment.append(cite(Number(citation)) || whole);
    } else if (linkText) {
      const url = safeUrl(linkUrl);
      if (url) {
        const el = document.createElement("a");
        el.href = url.href;
        el.target = "_blank";
        el.rel = "noopener noreferrer";
        el.textContent = linkText;
        fragment.append(el);
      } else { fragment.append(whole); }
    }
  }
  if (last < text.length) fragment.append(text.slice(last));
  return fragment;
}

function renderMarkdown(text, cite) {
  const root = document.createDocumentFragment();
  const lines = text.replace(/\r\n?/g, "\n").split("\n");
  let paragraph = [];
  let list = null;

  const flushParagraph = () => {
    if (!paragraph.length) return;
    const p = document.createElement("p");
    paragraph.forEach((line, i) => {
      if (i) p.append(document.createElement("br"));
      p.append(renderInline(line, cite));
    });
    root.append(p);
    paragraph = [];
  };
  const closeList = () => { list = null; };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmed = line.trim();

    if (trimmed.startsWith("```")) {  // fenced code block
      flushParagraph(); closeList();
      const body = [];
      while (++i < lines.length && !lines[i].trim().startsWith("```")) body.push(lines[i]);
      const pre = document.createElement("pre");
      const code = document.createElement("code");
      code.textContent = body.join("\n");
      pre.append(code);
      root.append(pre);
      continue;
    }
    if (!trimmed) { flushParagraph(); closeList(); continue; }
    if (/^([-*_])\1{2,}$/.test(trimmed.replace(/\s/g, ""))) {  // horizontal rule
      flushParagraph(); closeList();
      root.append(document.createElement("hr"));
      continue;
    }
    const heading = trimmed.match(/^(#{1,6})\s+(.*)$/);
    if (heading) {
      flushParagraph(); closeList();
      const el = document.createElement("p");
      el.className = "md-heading";
      const strong = document.createElement("strong");
      strong.append(renderInline(heading[2].replace(/\s+#+$/, ""), cite));
      el.append(strong);
      root.append(el);
      continue;
    }
    const quote = trimmed.match(/^>\s?(.*)$/);
    if (quote) {
      flushParagraph(); closeList();
      const el = document.createElement("blockquote");
      el.append(renderInline(quote[1], cite));
      root.append(el);
      continue;
    }
    const bullet = line.match(/^\s*[-*+]\s+(.*)$/);
    const numbered = line.match(/^\s*(\d{1,3})[.)]\s+(.*)$/);
    if (bullet || numbered) {
      flushParagraph();
      const tag = bullet ? "UL" : "OL";
      if (!list || list.tagName !== tag) {
        list = document.createElement(tag.toLowerCase());
        if (numbered && numbered[1] !== "1") list.start = Number(numbered[1]);
        root.append(list);
      }
      const item = document.createElement("li");
      item.append(renderInline(bullet ? bullet[1] : numbered[2], cite));
      list.append(item);
      continue;
    }
    if (list && /^\s{2,}\S/.test(line)) {  // continuation of the previous list item
      list.lastElementChild.append(" ", renderInline(trimmed, cite));
      continue;
    }
    closeList();
    paragraph.push(trimmed);
  }
  flushParagraph();
  return root;
}

function addMessage(role, text, sources = []) {
  const id = `m${++messageCount}`;
  const article = document.createElement("article");
  article.className = `message ${role}`;
  const label = document.createElement("strong");
  label.textContent = role === "user" ? "You" : "Relay";
  article.append(label);

  if (role === "assistant") {
    const content = document.createElement("div");
    content.className = "message-body";
    const cite = (n) => {
      if (n < 1 || n > sources.length) return null;
      const sup = document.createElement("sup");
      sup.className = "cite";
      const link = document.createElement("a");
      link.href = `#${id}-source-${n}`;
      link.textContent = `[${n}]`;
      link.setAttribute("aria-label", `Source ${n}: ${sources[n - 1].title}`);
      sup.append(link);
      return sup;
    };
    content.append(renderMarkdown(text, cite));
    article.append(content);
  } else {
    const content = document.createElement("p");
    // The user's own text is shown exactly as typed, never interpreted.
    content.textContent = text;
    article.append(content);
  }

  if (sources.length) {
    const heading = document.createElement("p");
    heading.className = "sources-label";
    heading.textContent = "Sources";
    // Numbered to match the [n] markers in the answer.
    const list = document.createElement("ol");
    list.className = "sources";
    list.setAttribute("aria-label", "Supporting sources");
    sources.forEach((source, index) => {
      const item = document.createElement("li");
      item.id = `${id}-source-${index + 1}`;
      const url = safeUrl(source.url);
      if (url) {
        const link = document.createElement("a");
        link.textContent = source.title;
        link.href = url.href;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        item.append(link);
      } else { item.append(source.title); }
      if (source.snippet) {
        const snippet = document.createElement("span");
        snippet.className = "snippet";
        snippet.textContent = `“${source.snippet}”`;
        item.append(snippet);
      }
      list.append(item);
    });
    article.append(heading, list);
  }
  messages.append(article);
  messages.scrollTop = messages.scrollHeight;
  return article;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (busy) return;
  if (!question) { status.textContent = "Type a question first."; input.focus(); return; }
  busy = true;
  send.disabled = clear.disabled = input.disabled = true;
  messages.querySelector(".welcome")?.remove();
  suggestions.hidden = true;
  const pendingMessage = addMessage("user", question);
  status.textContent = "Getting your answer…";
  messages.setAttribute("aria-busy", "true");
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 45000);
  try {
    const response = await fetch(form.action, {
      method: "POST",
      credentials: "same-origin",
      headers: {"Content-Type": "application/json", "X-CSRFToken": form.querySelector("[name=csrfmiddlewaretoken]").value},
      body: JSON.stringify({question, history: history.slice(-20)}),
      signal: controller.signal,
    });
    let data;
    try { data = await response.json(); } catch { throw new Error("The server could not respond. Refresh the page and try again."); }
    if (!response.ok) throw new Error(data.error || "Could not get an answer. Please try again.");
    if (typeof data.answer !== "string") throw new Error("The server returned an invalid answer. Please try again.");
    addMessage("assistant", data.answer, data.sources || []);
    history.push({role: "user", content: question}, {role: "assistant", content: data.answer.slice(0, 12000)});
    history = history.slice(-20);
    input.value = "";
    status.textContent = "";
  } catch (error) {
    pendingMessage.remove();
    status.textContent = error.name === "AbortError" ? "The answer took too long. Your question is still here; please try again." : error.message;
  } finally {
    clearTimeout(timeout);
    busy = false;
    send.disabled = clear.disabled = input.disabled = false;
    messages.removeAttribute("aria-busy");
    input.focus();
  }
});
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) { event.preventDefault(); form.requestSubmit(); }
});
suggestions.addEventListener("click", (event) => {
  if (event.target.tagName === "BUTTON" && !busy) { input.value = event.target.textContent; input.focus(); }
});
clear.addEventListener("click", () => {
  if (busy) return;
  history = [];
  messages.innerHTML = welcome;
  input.value = "";
  status.textContent = "";
  suggestions.hidden = false;
  input.focus();
});
