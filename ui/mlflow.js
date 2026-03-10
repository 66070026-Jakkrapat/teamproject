async function fetchJson(url) {
  const response = await fetch(url);
  return await response.json();
}

function escapeHtml(value) {
  return String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function renderCards(cards = []) {
  const root = document.getElementById("metricCards");
  root.innerHTML = cards.map((card) => `
    <article class="metric-card">
      <div class="metric-card__label">${escapeHtml(card.label || "-")}</div>
      <div class="metric-card__value">${escapeHtml(card.value || "--")}</div>
    </article>
  `).join("");
}

function renderLines(series = []) {
  const root = document.getElementById("linePanels");
  if (!series.length) {
    root.innerHTML = '<div class="empty">ยังไม่มี line metrics สำหรับแสดงผล</div>';
    return;
  }

  root.innerHTML = `<div class="chart-stack">${series.map((item) => {
    const max = Math.max(...item.values.map((point) => Number(point.y || 0)), 1);
    return `
      <section class="chart-card">
        <h3 class="chart-card__title">${escapeHtml(item.label || "-")}</h3>
        <div class="line-chart">
          ${item.values.map((point) => {
            const height = Math.max(8, Math.round((Number(point.y || 0) / max) * 140));
            const title = `${point.x || ""}: ${Number(point.y || 0).toFixed(3)}`;
            return `<div class="line-chart__bar" title="${escapeHtml(title)}" style="height:${height}px"></div>`;
          }).join("")}
        </div>
      </section>
    `;
  }).join("")}</div>`;
}

function renderBars(metrics = []) {
  const root = document.getElementById("barPanel");
  if (!metrics.length) {
    root.innerHTML = '<div class="empty">ยังไม่มี bar metrics สำหรับแสดงผล</div>';
    return;
  }

  const max = Math.max(...metrics.map((item) => Number(item.value || 0)), 1);
  root.innerHTML = `<div class="bars">${metrics.map((item) => {
    const height = Math.max(12, Math.round((Number(item.value || 0) / max) * 210));
    return `
      <div class="bar">
        <div class="bar__fill" style="height:${height}px"></div>
        <div class="bar__label">${escapeHtml(item.label || "-")}<br>${Number(item.value || 0).toFixed(2)}</div>
      </div>
    `;
  }).join("")}</div>`;
}

function renderRuns(runs = []) {
  const root = document.getElementById("runList");
  if (!runs.length) {
    root.innerHTML = '<div class="empty">ยังไม่มี recent runs</div>';
    return;
  }

  root.innerHTML = `<div class="run-list">${runs.map((run) => `
    <article class="run">
      <div class="run__title">${escapeHtml(run.run_name || run.run_id || "-")}</div>
      <div class="run__meta">
        experiment: ${escapeHtml(run.experiment || "-")}<br>
        status: ${escapeHtml(run.status || "-")}<br>
        metrics: ${Object.keys(run.metrics || {}).length}<br>
        run id: ${escapeHtml((run.run_id || "").slice(0, 12))}
      </div>
    </article>
  `).join("")}</div>`;
}

async function init() {
  const trackingUri = document.getElementById("trackingUri");
  const mlflowState = document.getElementById("mlflowState");

  try {
    const data = await fetchJson("/mlflow/summary");
    trackingUri.textContent = `Tracking URI: ${data.tracking_uri || "not configured"}`;
    mlflowState.textContent = data.enabled ? "State: connected" : "State: disabled";

    if (!data.enabled) {
      renderCards([
        { label: "Accuracy", value: "--" },
        { label: "Recall", value: "--" },
        { label: "Faithfulness", value: "--" },
        { label: "Relevance", value: "--" }
      ]);
      renderLines([]);
      renderBars([]);
      renderRuns([]);
      return;
    }

    renderCards(data.cards || []);
    renderLines(data.line_series || []);
    renderBars(data.bar_metrics || []);
    renderRuns(data.runs || []);
  } catch (error) {
    trackingUri.textContent = "Tracking URI: error";
    mlflowState.textContent = `State: ${String(error)}`;
    renderLines([]);
    renderBars([]);
    renderRuns([]);
  }
}

init();
