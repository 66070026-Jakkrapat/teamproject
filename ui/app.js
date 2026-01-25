const API = ""; // same origin

const el = (id) => document.getElementById(id);
const pretty = (obj) => JSON.stringify(obj, null, 2);

let chart = null;

async function getJSON(url) {
  const r = await fetch(url);
  return await r.json();
}
async function postJSON(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
  return await r.json();
}

function renderDashboard(dash) {
  const highlights = dash?.highlights || [];
  const metrics = dash?.metrics || [];
  const tables = dash?.tables || [];

  // highlights
  el("dashHighlights").innerHTML = highlights.length
    ? `<ul>${highlights.map(h => `<li>${h}</li>`).join("")}</ul>`
    : `<div class="muted">No highlights</div>`;

  // chart (use first metric if available)
  const ctx = el("chart").getContext("2d");
  if (chart) chart.destroy();

  if (metrics.length > 0 && metrics[0].points?.length) {
    const m = metrics[0];
    const labels = m.points.map(p => p.x);
    const values = m.points.map(p => p.y);
    chart = new Chart(ctx, {
      type: "line",
      data: {
        labels,
        datasets: [{
          label: `${m.name} (${m.unit || ""})`,
          data: values
        }]
      },
      options: {
        responsive: true,
        plugins: { legend: { display: true } }
      }
    });
  } else {
    chart = new Chart(ctx, {
      type: "bar",
      data: { labels: ["No data"], datasets: [{ label: "Metrics", data: [0] }] },
      options: { responsive: true }
    });
  }

  // tables
  let html = "";
  for (const t of tables) {
    const cols = t.columns || [];
    const rows = t.rows || [];
    html += `<h4>${t.name || "Table"}</h4>`;
    html += `<table><thead><tr>${cols.map(c => `<th>${c}</th>`).join("")}</tr></thead><tbody>`;
    html += rows.map(r => `<tr>${(r || []).map(v => `<td>${v}</td>`).join("")}</tr>`).join("");
    html += `</tbody></table>`;
  }
  el("dashTables").innerHTML = html || `<div class="muted">No tables</div>`;
}

el("btnHealth").onclick = async () => {
  el("healthStatus").textContent = "checking...";
  const data = await getJSON("/health");
  el("healthStatus").textContent = data?.ollama?.reachable ? "Ollama OK" : "Ollama NOT OK";
  el("scrapeOut").textContent = pretty(data);
};

el("btnScrape").onclick = async () => {
  const kw = el("kw").value.trim();
  const amt = parseInt(el("amt").value || "5", 10);
  if (!kw) return alert("ใส่ keyword ก่อน");

  el("scrapeOut").textContent = "scraping...";
  const data = await getJSON(`/scrape?keyword=${encodeURIComponent(kw)}&amount=${amt}`);
  el("scrapeOut").textContent = pretty(data);

  if (data?.ocr_job_id) {
    el("jobId").value = data.ocr_job_id;
    el("btnPostprocess").disabled = false;
  }
};

el("btnOcrStatus").onclick = async () => {
  const jobId = el("jobId").value.trim();
  if (!jobId) return alert("ใส่ OCR job id ก่อน");

  el("ocrOut").textContent = "checking...";
  const data = await getJSON(`/ocr/status/${encodeURIComponent(jobId)}`);
  el("ocrOut").textContent = pretty(data);
};

el("btnPostprocess").onclick = async () => {
  const jobId = el("jobId").value.trim();
  if (!jobId) return alert("ใส่ OCR job id ก่อน");

  el("ocrOut").textContent = "postprocessing (vision+report+index)...";
  const data = await postJSON(`/postprocess/${encodeURIComponent(jobId)}`, {});
  el("ocrOut").textContent = pretty(data);
};

el("btnUpload").onclick = async () => {
  const f = el("pdfFile").files[0];
  if (!f) return alert("เลือกไฟล์ PDF ก่อน");

  el("uploadOut").textContent = "uploading...";
  const fd = new FormData();
  fd.append("file", f);

  const r = await fetch(`/upload`, { method: "POST", body: fd });
  const data = await r.json();
  el("uploadOut").textContent = pretty(data);
};

el("btnAsk").onclick = async () => {
  const question = el("question").value.trim();
  const topk = parseInt(el("topk").value || "10", 10);
  if (!question) return alert("ใส่คำถามก่อน");

  el("askOut").textContent = "asking...";
  const data = await postJSON(`/ask`, { question, top_k: topk });
  el("askOut").textContent = pretty(data);

  if (data?.dashboard) renderDashboard(data.dashboard);
};
