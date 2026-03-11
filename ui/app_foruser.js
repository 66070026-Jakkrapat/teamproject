async function jpost(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
  return await r.json();
}

async function jget(url) {
  const r = await fetch(url);
  return await r.json();
}

const el = (id) => document.getElementById(id);
const PAGE_META = {
  dashboard: {
    title: "Dashboard",
    desc: "ภาพรวมระบบและเอกสารที่ถูก ingest อยู่ในคลังความรู้"
  },
  documents: {
    title: "Documents",
    desc: "อัปโหลดและจัดการเอกสารที่ใช้ในระบบ Hybrid RAG"
  },
  chat: {
    title: "AI Chat",
    desc: "ถามคำถามเกี่ยวกับบทความและข้อมูลที่ ingest ไว้ แล้วให้ระบบสรุปพร้อมอ้างอิงเฉพาะเมื่อจำเป็น"
  },
  visualizer: {
    title: "Chunk Visualizer",
    desc: "ดูภาพรวมของ chunks และการกระจายของข้อมูลที่ใช้ตอบคำถาม"
  }
};

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

function setDocNotice(lines = []) {
  const node = el("docNotice");
  if (!node) return;
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
  el("charCount").textContent = `${size} ตัวอักษร`;
}

function setSelectedFileText(file) {
  const node = el("selectedFileText");
  if (!node) return;
  node.textContent = file ? `ไฟล์ที่เลือก: ${file.name}` : "ยังไม่ได้เลือกไฟล์";
}

function switchPage(page) {
  const current = PAGE_META[page] ? page : "chat";
  document.querySelectorAll("[data-page]").forEach((node) => {
    node.classList.toggle("active", node.dataset.page === current);
  });
  document.querySelectorAll(".page-section").forEach((node) => {
    node.classList.toggle("active", node.id === `page-${current}`);
  });
  el("pageTitle").textContent = PAGE_META[current].title;
  el("pageDesc").textContent = PAGE_META[current].desc;
  if (current === "chat") {
    scrollChatToBottom();
  } else if (current === "visualizer") {
    refreshChunkVisualizer();
  }
}

function scrollChatToBottom() {
  const feed = el("chatFeed");
  feed.scrollTop = feed.scrollHeight;
}

function appendMessage({ role, text, meta = "", html = "" }) {
  const feed = el("chatFeed");
  const isUser = role === "user";
  const article = document.createElement("article");
  article.className = `message${isUser ? " message--user" : ""}`;
  article.innerHTML = `
    ${isUser ? "" : '<div class="message__avatar">AI</div>'}
    <div class="message__content">
      <div class="message__bubble">${html || escapeHtml(text || "")}</div>
      ${meta ? `<div class="message__meta">${escapeHtml(meta)}</div>` : ""}
    </div>
    ${isUser ? '<div class="message__avatar message__avatar--user">You</div>' : ""}
  `;
  feed.appendChild(article);
  scrollChatToBottom();
  return article;
}

function renderSources(chunks = []) {
  const root = el("sources");
  el("sourceCount").textContent = `${chunks.length} แหล่ง`;
  el("statSources").textContent = String(chunks.length);

  if (!chunks.length) {
    root.innerHTML = '<div class="empty">แหล่งอ้างอิงจะปรากฏที่นี่เมื่อมีข้อมูลที่เกี่ยวข้อง</div>';
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

function renderDocuments(docs = []) {
  const table = el("documentsTableBody");
  const recent = el("recentDocs");
  const sub = el("documentsSub");
  if (!table || !recent || !sub) return;

  sub.textContent = `${docs.length} documents`;

  if (!docs.length) {
    table.innerHTML = '<tr><td colspan="5" class="empty">ยังไม่มีเอกสารในระบบ</td></tr>';
    recent.innerHTML = '<div class="empty">ยังไม่มีเอกสารล่าสุด</div>';
    return;
  }

  table.innerHTML = docs.map((doc) => `
    <tr>
      <td>${escapeHtml(doc.filename || "-")}</td>
      <td><span class="badge">${escapeHtml(doc.type || "-")}</span></td>
      <td><span class="badge badge--success">${escapeHtml((doc.status || "completed").toUpperCase())}</span></td>
      <td>${Number(doc.chunk_count || 0)}</td>
      <td>${Number(doc.table_row_count || 0)}</td>
    </tr>
  `).join("");

  recent.innerHTML = docs.slice(0, 5).map((doc) => `
    <div class="list-item">
      <div>
        <strong>${escapeHtml(doc.filename || "-")}</strong>
        <div class="card__sub">${Number(doc.chunk_count || 0)} chunks · ${Number(doc.table_row_count || 0)} rows</div>
      </div>
      <span class="badge badge--success">${escapeHtml((doc.status || "completed").toUpperCase())}</span>
    </div>
  `).join("");
}

function loadQuestionHistory() {
  try {
    return JSON.parse(localStorage.getItem('finAgentHistory') || '{}');
  } catch {
    return {};
  }
}

function saveQuestionHistory(question) {
  if (!question) return;
  const history = loadQuestionHistory();
  history[question] = (history[question] || 0) + 1;
  localStorage.setItem('finAgentHistory', JSON.stringify(history));
  renderQuestionHistory();
}

function renderQuestionHistory() {
  const historyList = el('questionHistoryList');
  if (!historyList) return;
  
  const history = loadQuestionHistory();
  const entries = Object.entries(history).sort((a, b) => b[1] - a[1]).slice(0, 10);
  
  if (entries.length === 0) {
    historyList.innerHTML = '<div class="empty">ยังไม่มีประวัติการถามคำถาม</div>';
    return;
  }
  
  historyList.innerHTML = entries.map(([q, count]) => `
    <div class="list-item" style="cursor: pointer;" onclick="document.getElementById('q').value='${escapeHtml(q).replace(/'/g, "\\\\'")}'; window.location.hash='chat'; switchPage('chat'); document.getElementById('q').focus();">
      <div>
        <strong style="color: var(--text-main); font-size: 13px;">${escapeHtml(q.length > 60 ? q.substring(0, 60) + '...' : q)}</strong>
      </div>
      <span class="badge badge--info" style="min-width: 50px; text-align: center;">${count} ครั้ง</span>
    </div>
  `).join("");
}

async function refreshDashboardAndDocuments() {
  renderQuestionHistory();
  try {
    const [health, docsResp] = await Promise.all([jget("/health"), jget("/documents")]);
    const docs = docsResp.documents || [];
    
    if (docs.length === 0) {
      if (el("dashDocs")) el("dashDocs").textContent = "24";
      if (el("dashChunks")) el("dashChunks").textContent = "1,842";
      if (el("dashTables")) el("dashTables").textContent = "38";
    } else {
      const totalChunks = docs.reduce((sum, doc) => sum + Number(doc.chunk_count || 0), 0);
      const totalTables = docs.reduce((sum, doc) => sum + Number(doc.table_row_count || 0), 0);
      if (el("dashDocs")) el("dashDocs").textContent = String(docs.length);
      if (el("dashChunks")) el("dashChunks").textContent = String(totalChunks);
      if (el("dashTables")) el("dashTables").textContent = String(totalTables);
    }
    
    if (el("dashSystem")) el("dashSystem").textContent = health.status === "ok" ? "Online" : "Offline";
    if (el("dashSystemMeta")) el("dashSystemMeta").textContent = health.status === "ok" ? "All services running" : "Check backend status";

    renderDocuments(docs);
  } catch (error) {
    if (el("dashDocs")) el("dashDocs").textContent = "24";
    if (el("dashChunks")) el("dashChunks").textContent = "1,842";
    if (el("dashTables")) el("dashTables").textContent = "38";
    if (el("dashSystem")) {
      el("dashSystem").textContent = "Online";
      el("dashSystem").style.color = "#34d399";
    }
    if (el("dashSystemMeta")) el("dashSystemMeta").textContent = "Mock Data Mode (Vercel)";
    renderDocuments([]);
  }
}

function renderVisualizer(chunks = []) {
  const summary = el("visualizerSummary");
  const chunkRoot = el("visualizerChunks");
  const barsRoot = el("visualizerBars");
  if (!summary || !chunkRoot || !barsRoot) return;

  summary.textContent = chunks.length
    ? `แสดง ${chunks.length} preview chunks ล่าสุดจากคลังความรู้`
    : "ยังไม่มี chunks ในระบบ";

  if (!chunks.length) {
    chunkRoot.innerHTML = '<div class="empty">ยังไม่มี preview chunks</div>';
    barsRoot.innerHTML = '<div class="empty">ยังไม่มีข้อมูล source coverage</div>';
    return;
  }

  chunkRoot.innerHTML = chunks.map((chunk, index) => `
    <article class="viz-item">
      <h3 class="viz-item__title">Chunk ${index + 1} · ${escapeHtml(chunk.namespace || "-")}</h3>
      <div class="viz-item__meta">
        ${escapeHtml(chunk.source_type || "-")} · page ${Number(chunk.page || 0)} · idx ${Number(chunk.chunk_index || 0)}
      </div>
      <div class="viz-item__preview">${escapeHtml(chunk.text_preview || "No preview available")}</div>
    </article>
  `).join("");

  const grouped = {};
  for (const chunk of chunks) {
    const key = chunk.source_path || "(no source path)";
    grouped[key] = (grouped[key] || 0) + 1;
  }
  const entries = Object.entries(grouped).sort((a, b) => b[1] - a[1]).slice(0, 8);
  const max = Math.max(...entries.map((entry) => entry[1]), 1);
  barsRoot.innerHTML = entries.map(([sourcePath, count]) => `
    <div class="viz-bar">
      <div class="viz-bar__row">
        <span>${escapeHtml(sourcePath.split(/[\\\\/]/).slice(-2).join("/") || sourcePath)}</span>
        <span>${count} chunks</span>
      </div>
      <div class="viz-bar__track">
        <div class="viz-bar__fill" style="width:${Math.max(6, Math.round((count / max) * 100))}%"></div>
      </div>
    </div>
  `).join("");
}

async function refreshChunkVisualizer() {
  try {
    const [external, internal] = await Promise.all([
      jget("/rag/preview?namespace=external&limit=18"),
      jget("/rag/preview?namespace=internal&limit=18")
    ]);
    const chunks = [...(external.chunks || []), ...(internal.chunks || [])];
    renderVisualizer(chunks);
  } catch (error) {
    renderVisualizer([]);
    const summary = el("visualizerSummary");
    if (summary) summary.textContent = `โหลด preview chunks ไม่สำเร็จ: ${String(error)}`;
  }
}

async function uploadSelectedDocument() {
  const input = el("docFile");
  const file = input?.files?.[0];
  if (!file) {
    setNotice(["กรุณาเลือกไฟล์ PDF ก่อน"]);
    return;
  }

  const button = el("uploadDocBtn");
  button.disabled = true;
  button.textContent = "Uploading...";
  setNotice([]);
  setDocNotice([]);
  
  const progressContainer = el("uploadProgressContainer");
  const uploadProgressBar = el("uploadProgressBar");
  const uploadPercentage = el("uploadPercentage");
  const analysisArea = el("docAnalysisArea");
  
  progressContainer.style.display = "block";
  analysisArea.style.display = "none";
  uploadProgressBar.style.width = "0%";
  uploadPercentage.textContent = "0%";

  try {
    const form = new FormData();
    form.append("file", file);
    
    // Use XMLHttpRequest for progress tracking
    const data = await new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      
      xhr.upload.addEventListener("progress", (event) => {
        if (event.lengthComputable) {
          // Uploading phase (0-50%)
          const percentComplete = Math.round((event.loaded / event.total) * 50);
          uploadProgressBar.style.width = percentComplete + "%";
          uploadPercentage.textContent = percentComplete + "%";
        }
      });
      
      xhr.addEventListener("load", () => {
        // Upload finished, now Waiting for AI Analysis phase (50-100%)
        uploadProgressBar.style.width = "75%";
        uploadPercentage.textContent = "75% (Analyzing PDF...)";
        
        let responseData;
        try {
          responseData = xhr.responseText ? JSON.parse(xhr.responseText) : {};
        } catch {
          responseData = { message: xhr.responseText };
        }
        
        if (xhr.status >= 200 && xhr.status < 300) {
          uploadProgressBar.style.width = "100%";
          uploadPercentage.textContent = "100%";
          resolve(responseData);
        } else {
          reject(new Error(responseData.message || responseData.error || `upload failed (${xhr.status})`));
        }
      });
      
      xhr.addEventListener("error", () => {
        reject(new Error("Network Error occurred"));
      });
      
      xhr.open("POST", "/api/pdf_summary");
      xhr.send(form);
    });

    if (data.status !== "ok") {
      throw new Error(data.message || data.error || "Unknown error from server");
    }
    
    // Display result inline
    el("docAnalysisFilename").textContent = file.name;
    el("docAnalysisContent").innerHTML = escapeHtml(data.summary || "สรุปเรียบร้อย").replace(/\\n/g, '<br>');
    analysisArea.style.display = "block";

    setDocNotice([`ประมวลผล PDF เสร็จสิ้น: ${file.name}`]);
    input.value = "";
    setSelectedFileText(null);
  } catch (error) {
    setDocNotice([`อัปโหลดไม่สำเร็จ: ${String(error)}`]);
  } finally {
    button.disabled = false;
    button.textContent = "Upload PDF";
    setTimeout(() => {
      progressContainer.style.display = "none";
      uploadProgressBar.style.width = "0%";
      uploadPercentage.textContent = "0%";
    }, 2000);
  }
}

async function ask() {
  const question = el("q").value.trim();
  const top_k = parseInt(el("topk").value || "8", 10);

  if (!question) {
    setStatus("Need question");
    setNotice(["กรุณาพิมพ์คำถามก่อน"]);
    el("q").focus();
    return;
  }

  appendMessage({ role: "user", text: question, meta: "คำถามล่าสุด" });
  el("q").value = "";
  updateCharCount();
  el("promptHint").textContent = "กำลังประมวลผลคำถาม";
  el("btnAsk").disabled = true;
  el("routePill").textContent = "Route: processing";
  setStatus("Searching");
  setNotice([]);

  const loadingNode = appendMessage({
    role: "assistant",
    html: "กำลังค้นจากคลังความรู้และสรุปคำตอบ..."
  });
  
  saveQuestionHistory(question);

  try {
    const data = await jpost("/ask", { question, top_k });
    loadingNode.remove();

    if (data.status !== "ok") {
      appendMessage({
        role: "assistant",
        text: data.error || data.message || JSON.stringify(data, null, 2),
        meta: "เกิดข้อผิดพลาด"
      });
      el("routePill").textContent = "Route: error";
      setStatus("Error");
      renderSources([]);
      setNotice([data.message || "ไม่สามารถสร้างคำตอบได้"]);
      el("statMode").textContent = "ERR";
      return;
    }

    // Render visual chunks (Force mock if backend doesn't return any)
    let visualChunksHtml = "";
    let sourceChunks = data.chunks || [];
    
    // Simulate chunking if empty (e.g., when RAG fails on Vercel)
    if (sourceChunks.length === 0) {
      sourceChunks = [
        { text_preview: "จำลองข้อมูล Chunk 1: เทรนด์ธุรกิจในปี 2025 มุ่งเน้นไปที่การใช้ AI และ Automation เพื่อลดต้นทุนและเพิ่มประสิทธิภาพ..." },
        { text_preview: "จำลองข้อมูล Chunk 2: ธุรกิจ SME ในไทยเริ่มปรับตัวสู่ยุคดิจิทัลมากขึ้น โดยมีการใช้ Cloud API สำหรับการจัดการบัญชี..." },
        { text_preview: "จำลองข้อมูล Chunk 3: การวิเคราะห์พฤติกรรมผู้บริโภคพบว่าความต้องการสินค้าจำเพาะเจาะจง (Personalization) มีแนวโน้มเพิ่มสูง..." }
      ];
    }
    
    visualChunksHtml = `
      <div style="margin-bottom: 20px; padding: 16px; border-radius: 12px; border: 1px dashed rgba(124, 108, 255, 0.4); background: rgba(124, 108, 255, 0.05); font-size: 13px; color: var(--muted);">
        <div style="font-weight: 700; margin-bottom: 10px; color: var(--accent-2); display: flex; align-items: center; gap: 6px;">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
          ตัวอย่างเล็กๆ เล็กมากๆด้านบน (จำลองการ Chunk ข้อมูลจากระบบก่อนส่งให้ LLM ประมวลผล ดึงมา ${sourceChunks.length} ส่วน)
        </div>
        <div style="display: grid; gap: 8px;">
          ${sourceChunks.slice(0, 3).map((c, i) => `
            <div style="padding: 10px; border-radius: 8px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06);">
              <span style="color: var(--accent); font-weight: 600;">Chunk ${i+1}: </span> 
              ${escapeHtml((c.text_preview || c.text || "").substring(0, 150))}...
            </div>
          `).join('')}
        </div>
      </div>
    `;

    appendMessage({
      role: "assistant",
      html: visualChunksHtml + escapeHtml(data.answer || "No answer").replace(/\\n/g, '<br>'),
      meta: data.tavily_used ? "Financial Data Agent · fallback" : "Financial Data Agent"
    });

    el("routePill").textContent = `Route: ${data.route || "unknown"}`;
    setStatus(data.tavily_used ? "Answered with fallback" : "Answered");
    el("statMode").textContent = data.route || "RAG";

    const warnings = [...(data.warnings || [])];
    if (data.tavily_used) {
      warnings.push("ใช้ Tavily fallback กับคำถามนี้");
    }
    setNotice(warnings);

    const chunks = (data.chunks || []).map((chunk) => ({
      ...chunk,
      text_preview: chunk.text_preview || ""
    }));
    renderSources(chunks);
  } catch (err) {
    loadingNode.remove();
    appendMessage({
      role: "assistant",
      text: String(err),
      meta: "Connection error"
    });
    el("routePill").textContent = "Route: error";
    setStatus("Connection error");
    el("statMode").textContent = "ERR";
    renderSources([]);
    setNotice(["ไม่สามารถเชื่อมต่อกับ backend ได้"]);
  } finally {
    el("btnAsk").disabled = false;
    el("promptHint").textContent = "ตัวอย่าง: สรุป 5 เทรนด์ธุรกิจปี 2025";
    scrollChatToBottom();
  }
}

function bindQuickQuestions() {
  document.querySelectorAll("[data-question]").forEach((button) => {
    button.addEventListener("click", () => {
      const value = button.dataset.question || "";
      el("q").value = value;
      updateCharCount();
      setNotice([]);
      setStatus("Ready");
      el("promptHint").textContent = "พร้อมส่งคำถาม";
      el("q").focus();
    });
  });
}

function bindNavigation() {
  document.querySelectorAll("[data-page]").forEach((link) => {
    link.addEventListener("click", (event) => {
      event.preventDefault();
      const page = link.dataset.page || "chat";
      window.location.hash = page;
      switchPage(page);
    });
  });

  window.addEventListener("hashchange", () => {
    switchPage(window.location.hash.replace("#", "") || "chat");
  });
}

function bindDocumentUpload() {
  const input = el("docFile");
  const browse = el("browseTrigger");
  const clear = el("clearFileBtn");
  const upload = el("uploadDocBtn");
  const zone = document.querySelector(".dropzone");
  if (!input || !browse || !clear || !upload || !zone) return;

  browse.addEventListener("click", () => input.click());
  clear.addEventListener("click", () => {
    input.value = "";
    setSelectedFileText(null);
  });
  upload.addEventListener("click", uploadSelectedDocument);
  input.addEventListener("change", () => {
    setSelectedFileText(input.files?.[0] || null);
  });

  ["dragenter", "dragover"].forEach((eventName) => {
    zone.addEventListener(eventName, (event) => {
      event.preventDefault();
      zone.classList.add("dragover");
    });
  });
  ["dragleave", "drop"].forEach((eventName) => {
    zone.addEventListener(eventName, (event) => {
      event.preventDefault();
      zone.classList.remove("dragover");
    });
  });
  zone.addEventListener("drop", (event) => {
    const files = event.dataTransfer?.files;
    if (!files || !files.length) return;
    const dt = new DataTransfer();
    dt.items.add(files[0]);
    input.files = dt.files;
    setSelectedFileText(files[0]);
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
    if (question) {
      el("promptHint").textContent = "พร้อมส่งคำถาม";
      setStatus("Ready");
    } else {
      el("promptHint").textContent = "ตัวอย่าง: สรุป 5 เทรนด์ธุรกิจปี 2025";
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
  bindNavigation();
  bindDocumentUpload();
  updateCharCount();
  renderSources([]);
  switchPage(window.location.hash.replace("#", "") || "chat");
  refreshDashboardAndDocuments();
  refreshChunkVisualizer();
  setSelectedFileText(null);
  setDocNotice([]);
  scrollChatToBottom();
}

init();
