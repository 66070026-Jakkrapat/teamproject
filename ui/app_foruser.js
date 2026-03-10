async function jpost(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
  return await r.json();
}

const el = (id) => document.getElementById(id);

function escapeHtml(value) {
  return String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function setStatus(text) {
  el("statusPill").textContent = text;
}

function setNotice(lines = []) {
  const node = el("notice");
  if (!lines.length) {
    node.textContent = "";
    node.dataset.visible = "false";
    return;
  }
  node.textContent = lines.join("\n");
  node.dataset.visible = "true";
}

function updateCharCount() {
  const size = el("q").value.trim().length;
  el("charCount").textContent = `${size} chars`;
}

function setQuestionPreview(text) {
  el("questionPreview").textContent = text || "No question yet";
}

function renderSources(chunks = []) {
  const root = el("sources");
  el("sourceCount").textContent = `${chunks.length} chunks`;

  if (!chunks.length) {
    root.innerHTML = '<div class="empty">Relevant chunks, URLs, scores, and retrieval modes will appear here.</div>';
    return;
  }

  root.innerHTML = chunks.map((chunk, index) => {
    const sourceType = escapeHtml(chunk.source_type || "unknown");
    const sourcePath = escapeHtml(chunk.source_path || "-");
    const sourceUrl = escapeHtml(chunk.source_url || "");
    const page = escapeHtml(chunk.page ? `Page ${chunk.page}` : "Page -");
    const retrievalMode = escapeHtml(chunk.retrieval_mode || "semantic");
    const score = Number(chunk.score || 0).toFixed(3);
    const preview = [
      escapeHtml(chunk.text_preview),
      sourceUrl ? `URL: ${sourceUrl}` : ""
    ].filter(Boolean).join("\n");

    return `
      <article class="source">
        <h3 class="source__title">Chunk ${index + 1} · ${sourceType}</h3>
        <p class="source__meta">${page} · score ${score} · mode ${retrievalMode}</p>
        <p class="source__meta">${sourcePath}</p>
        <pre class="source__preview">${preview || "No preview available"}</pre>
      </article>
    `;
  }).join("");
}

async function ask() {
  const question = el("q").value.trim();
  const top_k = parseInt(el("topk").value || "8", 10);

  if (!question) {
    setStatus("Need question");
    setNotice(["Please enter a question first.", "กรุณาพิมพ์คำถามก่อน"]);
    el("q").focus();
    return;
  }

  setQuestionPreview(question);
  el("btnAsk").disabled = true;
  el("btnAsk").textContent = "Searching...";
  el("answer").textContent = "Searching your knowledge base and drafting the answer...\n\nกำลังค้นจากคลังความรู้และสรุปคำตอบ...";
  el("routePill").textContent = "Route: processing";
  setStatus("Searching");
  setNotice([]);

  try {
    const data = await jpost("/ask", { question, top_k });
    if (data.status !== "ok") {
      el("answer").textContent = data.error || data.message || JSON.stringify(data, null, 2);
      el("routePill").textContent = "Route: error";
      setStatus("Error");
      renderSources([]);
      setNotice(["Answer generation failed.", data.message || "Unknown error"]);
      return;
    }

    el("answer").textContent = data.answer || "No answer";
    el("routePill").textContent = `Route: ${data.route || "unknown"}`;
    setStatus(data.tavily_used ? "Answered with fallback" : "Answered");

    const warnings = [...(data.warnings || [])];
    if (data.tavily_used) {
      warnings.push("Used Tavily fallback for this question.");
    }
    setNotice(warnings);

    const chunks = (data.chunks || []).map((chunk) => ({
      ...chunk,
      text_preview: chunk.text_preview || ""
    }));
    renderSources(chunks);
  } catch (err) {
    el("answer").textContent = String(err);
    el("routePill").textContent = "Route: error";
    setStatus("Connection error");
    renderSources([]);
    setNotice(["Cannot connect to backend.", "ไม่สามารถเชื่อมต่อกับ backend ได้"]);
  } finally {
    el("btnAsk").disabled = false;
    el("btnAsk").textContent = "Send";
  }
}

function bindQuickQuestions() {
  document.querySelectorAll("[data-question]").forEach((button) => {
    button.addEventListener("click", () => {
      const value = button.dataset.question || "";
      el("q").value = value;
      setQuestionPreview(value);
      updateCharCount();
      setNotice([]);
      setStatus("Ready");
      el("promptHint").textContent = "Ready to ask";
      el("q").focus();
    });
  });
}

function init() {
  el("askForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    await ask();
  });

  el("q").addEventListener("input", () => {
    const question = el("q").value.trim();
    updateCharCount();
    setQuestionPreview(question);
    if (question) {
      el("promptHint").textContent = "Ready to ask";
      setStatus("Ready");
    } else {
      el("promptHint").textContent = "Ask clearly for a better answer";
      setStatus("Ready");
    }
  });

  el("q").addEventListener("keydown", async (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault();
      await ask();
    }
  });

  bindQuickQuestions();
  updateCharCount();
  setQuestionPreview("");
}

init();
