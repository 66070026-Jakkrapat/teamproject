/**
 * Chunk Visualizer Component
 * Visual display of semantic chunks with color-coded boundaries,
 * similarity heatmap, and interactive details.
 */

function renderChunkVisualizer(container) {
    container.innerHTML = `
        <div class="page-header">
            <h1>Chunk Visualizer</h1>
            <p>ดูผลลัพธ์การแบ่ง Chunk แบบ Semantic + LLM — เห็นขอบเขตและ similarity ระหว่าง chunks</p>
        </div>

        <div class="card" style="margin-bottom:24px">
            <div class="card-header">
                <div class="card-title">Select Document</div>
            </div>
            <div class="doc-selector">
                <select id="vis-doc-select" class="input-field" onchange="loadChunkVisualization()">
                    <option value="">— Select a document —</option>
                </select>
                <button class="btn btn-secondary btn-sm" onclick="refreshDocList()">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg>
                    Refresh
                </button>
            </div>
        </div>

        <!-- Visualization Area -->
        <div id="chunk-vis-area">
            <div class="empty-state">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                    <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
                </svg>
                <h3>Select a document to visualize</h3>
                <p>See how Semantic + LLM chunking divides your document with color-coded boundaries and similarity scores.</p>
            </div>
        </div>
    `;

    loadVisDocList();
}

async function loadVisDocList() {
    const select = document.getElementById('vis-doc-select');
    try {
        const data = await api.get('/documents');
        const docs = (data.documents || []).filter(d => d.status === 'completed' && d.chunk_count > 0);

        select.innerHTML = '<option value="">— Select a document —</option>';
        docs.forEach(d => {
            const opt = document.createElement('option');
            opt.value = d.id;
            opt.textContent = `${d.filename} (${d.chunk_count} chunks)`;
            select.appendChild(opt);
        });

        // Auto-select if coming from documents page
        const savedId = sessionStorage.getItem('selectedDocId');
        if (savedId) {
            select.value = savedId;
            sessionStorage.removeItem('selectedDocId');
            loadChunkVisualization();
        }
    } catch (e) {
        showToast('Failed to load document list', 'error');
    }
}

function refreshDocList() {
    loadVisDocList();
}

async function loadChunkVisualization() {
    const docId = document.getElementById('vis-doc-select').value;
    const area = document.getElementById('chunk-vis-area');

    if (!docId) {
        area.innerHTML = `
            <div class="empty-state">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                    <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
                </svg>
                <h3>Select a document to visualize</h3>
                <p>See how Semantic + LLM chunking divides your document.</p>
            </div>
        `;
        return;
    }

    area.innerHTML = `
        <div class="loader" style="padding:32px;justify-content:center">
            <span class="loader-spinner"></span>
            <span>Loading chunk data...</span>
        </div>
    `;

    try {
        const data = await api.get(`/chunks/${docId}`);
        renderChunks(area, data);
    } catch (e) {
        area.innerHTML = `
            <div class="empty-state">
                <p style="color:var(--accent-danger)">Failed to load chunks: ${e.message}</p>
            </div>
        `;
    }
}

function renderChunks(area, data) {
    const { filename, total_chunks, chunks } = data;

    // Generate colors for chunks using HSL for visual distinction
    const getChunkColor = (index, total) => {
        const hue = (index / total) * 300 + 220; // Range from blue to purple
        return `hsl(${hue % 360}, 70%, 55%)`;
    };

    // Similarity to color
    const simToColor = (sim) => {
        if (sim === null || sim === undefined) return 'var(--text-muted)';
        // High similarity = green, low = red
        const hue = sim * 120; // 0=red, 120=green
        return `hsl(${hue}, 80%, 50%)`;
    };

    const simToWidth = (sim) => {
        if (sim === null || sim === undefined) return 0;
        return Math.round(sim * 100);
    };

    let html = `
        <div class="card" style="margin-bottom:20px">
            <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px">
                <div>
                    <span style="font-weight:600;font-size:1rem">${filename}</span>
                    <span class="badge badge-primary" style="margin-left:8px">${total_chunks} chunks</span>
                </div>
                <div style="display:flex;align-items:center;gap:16px;font-size:0.78rem;color:var(--text-muted)">
                    <span>🟢 High Similarity</span>
                    <span>🟡 Medium</span>
                    <span>🔴 Low (Split Point)</span>
                </div>
            </div>
        </div>

        <div class="chunk-vis-container">
    `;

    chunks.forEach((chunk, i) => {
        const color = getChunkColor(i, chunks.length);

        html += `
            <div class="chunk-block" style="border-left-color:${color}" onclick="toggleChunk(this)">
                <div class="chunk-header">
                    <span class="chunk-index">Chunk #${chunk.chunk_index}</span>
                    <span class="chunk-tokens">${chunk.token_count || '—'} tokens</span>
                </div>
                <div class="chunk-text">${escapeHtmlVis(chunk.chunk_text)}</div>
                ${chunk.summary ? `
                    <div class="chunk-summary">
                        <strong style="font-size:0.75rem;display:block;margin-bottom:4px">📝 LLM Summary</strong>
                        ${escapeHtmlVis(chunk.summary)}
                    </div>
                ` : ''}
            </div>
        `;

        // Add boundary bar between chunks (except after last)
        if (i < chunks.length - 1) {
            const sim = chunk.similarity_to_next;
            const simText = sim !== null && sim !== undefined ? (sim * 100).toFixed(1) + '%' : '—';
            const barColor = simToColor(sim);
            const barWidth = simToWidth(sim);

            html += `
                <div class="chunk-boundary">
                    <div class="boundary-bar">
                        <div class="boundary-fill" style="width:${barWidth}%;background:${barColor}"></div>
                    </div>
                    <div class="boundary-label" style="color:${barColor}">similarity: ${simText}</div>
                </div>
            `;
        }
    });

    html += '</div>';
    area.innerHTML = html;
}

function toggleChunk(el) {
    const textEl = el.querySelector('.chunk-text');
    if (textEl) {
        textEl.classList.toggle('expanded');
    }
}

function escapeHtmlVis(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
