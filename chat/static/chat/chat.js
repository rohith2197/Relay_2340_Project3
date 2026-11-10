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

function addMessage(role, text, sources = []) {
  const article = document.createElement("article");
  article.className = `message ${role}`;
  const label = document.createElement("strong");
  label.textContent = role === "user" ? "You" : "Relay";
  const content = document.createElement("p");
  // Model output and user text are never interpreted as HTML.
  content.textContent = text;
  article.append(label, content);
  if (sources.length) {
    const list = document.createElement("ul");
    list.setAttribute("aria-label", "Supporting sources");
    for (const source of sources) {
      const item = document.createElement("li");
      let url;
      try { url = new URL(source.url); } catch { /* Title-only source. */ }
      if (url && ["https:", "http:"].includes(url.protocol)) {
        const link = document.createElement("a");
        link.textContent = source.title;
        link.href = url.href;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        item.append(link);
      } else { item.textContent = source.title; }
      list.append(item);
    }
    article.append(list);
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
