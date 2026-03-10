async function jget(url) {
  const r = await fetch(url);
  return await r.json();
}

async function jpost(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return await r.json();
}

const el = (id) => document.getElementById(id);
let activePollTimer = null;

function pretty(data) {
  return JSON.stringify(data, null, 2);
}

function setBadge(id, text, mode) {
  const node = el(id);
  if (!node) return;
  node.textContent = text;
  node.className = `badge badge--${mode || "idle"}`;
}

function setJobId(jobId) {
  if (jobId) {
    el("jobId").value = jobId;
  }
}

function detectMode(data) {
  const execution = data?.execution || "";
  if (execution === "remote_worker") return "remote";
  if (execution === "worker_local" || execution === "local") return "local";
  return "idle";
}

function stopPolling() {
  if (activePollTimer) {
    window.clearTimeout(activePollTimer);
    activePollTimer = null;
  }
}

function shouldContinuePolling(statusData) {
  const stage = String(statusData?.stage || "").toLowerCase();
  return !["ready", "error", "done", "completed"].includes(stage);
}

async function startPipeline() {
  const keyword = el("kw").value.trim();
  const amount = parseInt(el("amt").value || "3", 10);
  const fixed_sources = (el("fixedSources").value || "")
    .split(/\r?\n/)
    .map((x) => x.trim())
    .filter(Boolean);
  if (!keyword) return;

  el("pipeOut").textContent = "starting...";
  setBadge("pipeMode", "execution: starting", "idle");
  try {
    const data = await jpost("/pipeline/external/scrape", { keyword, amount, fixed_sources });
    el("pipeOut").textContent = pretty(data);
    setBadge("pipeMode", `execution: ${data.execution || data.mode || "local"}`, detectMode(data));
    if (data.job_id) {
      setJobId(data.job_id);
      startAutoPoll(data.job_id);
    }
  } catch (error) {
    el("pipeOut").textContent = String(error);
    setBadge("pipeMode", "execution: failed", "idle");
  }
}

async function uploadPdf() {
  const fileInput = el("pdfFile");
  const file = fileInput.files && fileInput.files[0];
  const entityHint = el("entityHint").value.trim() || "internal_doc";
  if (!file) return;

  el("uploadOut").textContent = "uploading...";
  setBadge("uploadMode", "execution: uploading", "idle");
  try {
    const form = new FormData();
    form.append("file", file);
    const url = `/pipeline/internal/upload_pdf?entity_hint=${encodeURIComponent(entityHint)}`;
    const r = await fetch(url, { method: "POST", body: form });
    const data = await r.json();
    el("uploadOut").textContent = pretty(data);
    setBadge("uploadMode", `execution: ${data.execution || "local"}`, detectMode(data));
    if (data.job_id) {
      setJobId(data.job_id);
      startAutoPoll(data.job_id);
    }
  } catch (error) {
    el("uploadOut").textContent = String(error);
    setBadge("uploadMode", "execution: failed", "idle");
  }
}

async function ask() {
  const question = el("q").value.trim();
  const top_k = parseInt(el("topk").value || "8", 10);
  if (!question) return;

  el("answer").textContent = "thinking...";
  try {
    const data = await jpost("/ask", { question, top_k });
    if (data.status !== "ok") {
      el("answer").textContent = pretty(data);
      return;
    }
    el("answer").textContent = data.answer || "";
    el("chunks").textContent = pretty(data.chunks || []);
    el("tavily").textContent = pretty(data.tavily_results || []);
  } catch (error) {
    el("answer").textContent = String(error);
  }
}

async function preview() {
  const namespace = el("ns").value;
  const source_type = el("stype").value.trim();
  const limit = parseInt(el("limit").value || "20", 10);

  el("preview").textContent = "loading...";
  try {
    const data = await jget(
      `/rag/preview?namespace=${encodeURIComponent(namespace)}&source_type=${encodeURIComponent(source_type)}&limit=${limit}`
    );
    el("preview").textContent = pretty(data);
  } catch (error) {
    el("preview").textContent = String(error);
  }
}

async function checkWorkerHealth() {
  el("workerOut").textContent = "checking worker...";
  setBadge("trackMode", "tracking: worker health", "idle");
  try {
    const data = await jget("/worker/status");
    el("workerOut").textContent = pretty(data);
    setBadge("trackMode", `tracking: ${data.reachable ? "worker reachable" : "worker unavailable"}`, data.reachable ? "remote" : "idle");
  } catch (error) {
    el("workerOut").textContent = `Worker health failed\n${String(error)}`;
    setBadge("trackMode", "tracking: failed", "idle");
  }
}

async function trackJob(jobIdOverride = "") {
  const jobId = jobIdOverride || el("jobId").value.trim();
  if (!jobId) return;

  el("workerOut").textContent = "loading status...";
  try {
    const [status, logs] = await Promise.all([
      jget(`/jobs/${encodeURIComponent(jobId)}/status`),
      jget(`/jobs/${encodeURIComponent(jobId)}/logs?tail=80`),
    ]);
    el("workerOut").textContent = pretty({ status, logs });
    const stage = status?.stage || status?.status || "unknown";
    const mode = String(jobId).startsWith("remote:") ? "remote" : "local";
    setBadge("trackMode", `tracking: ${stage}`, mode);
    return status;
  } catch (error) {
    el("workerOut").textContent = `Track job failed\n${String(error)}`;
    setBadge("trackMode", "tracking: failed", "idle");
    return null;
  }
}

function startAutoPoll(jobId) {
  stopPolling();
  const tick = async () => {
    const status = await trackJob(jobId);
    if (status && shouldContinuePolling(status)) {
      activePollTimer = window.setTimeout(tick, 3000);
    } else {
      activePollTimer = null;
    }
  };
  tick();
}

el("btnStart").addEventListener("click", startPipeline);
el("btnUploadPdf").addEventListener("click", uploadPdf);
el("btnAsk").addEventListener("click", ask);
el("btnPrev").addEventListener("click", preview);
el("btnWorkerHealth").addEventListener("click", checkWorkerHealth);
el("btnTrackJob").addEventListener("click", () => {
  const jobId = el("jobId").value.trim();
  if (!jobId) return;
  startAutoPoll(jobId);
});
