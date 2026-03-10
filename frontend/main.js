/**
 * Financial Data Agent — Main Application
 * SPA Router, API Client, and Page Orchestration
 */

// ============================================================
// API Client
// ============================================================
const API_BASE = '/api';

const api = {
    async get(path) {
        const res = await fetch(`${API_BASE}${path}`);
        if (!res.ok) throw new Error(`API Error: ${res.status}`);
        return res.json();
    },

    async post(path, body) {
        const res = await fetch(`${API_BASE}${path}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        if (!res.ok) throw new Error(`API Error: ${res.status}`);
        return res.json();
    },

    async upload(path, file) {
        const form = new FormData();
        form.append('file', file);
        const res = await fetch(`${API_BASE}${path}`, {
            method: 'POST',
            body: form,
        });
        if (!res.ok) throw new Error(`API Error: ${res.status}`);
        return res.json();
    },

    async delete(path) {
        const res = await fetch(`${API_BASE}${path}`, { method: 'DELETE' });
        if (!res.ok) throw new Error(`API Error: ${res.status}`);
        return res.json();
    },
};

// ============================================================
// Toast Notifications
// ============================================================
const toastContainer = document.createElement('div');
toastContainer.className = 'toast-container';
document.body.appendChild(toastContainer);

function showToast(message, type = 'info', duration = 4000) {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;

    const icons = {
        success: '✓',
        error: '✗',
        warning: '⚠',
        info: 'ℹ',
    };

    toast.innerHTML = `
        <span style="font-size:1.1rem">${icons[type] || icons.info}</span>
        <span style="font-size:0.88rem;color:var(--text-secondary)">${message}</span>
    `;
    toastContainer.appendChild(toast);

    setTimeout(() => {
        toast.style.animation = 'slideIn 0.3s ease-out reverse forwards';
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

// ============================================================
// SPA Router
// ============================================================
const pages = {
    dashboard: renderDashboard,
    scraping: renderScraping,
    documents: renderDocuments,
    chat: renderChat,
    visualizer: renderChunkVisualizer,
};

function navigate(page) {
    // Update nav active state
    document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
    const navEl = document.getElementById(`nav-${page}`);
    if (navEl) navEl.classList.add('active');

    // Render page
    const main = document.getElementById('main-content');
    main.style.opacity = '0';
    main.style.transform = 'translateY(8px)';

    setTimeout(() => {
        if (pages[page]) {
            pages[page](main);
        } else {
            pages.dashboard(main);
        }
        main.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
        main.style.opacity = '1';
        main.style.transform = 'translateY(0)';
    }, 150);
}

// Hash-based routing
function handleRoute() {
    const hash = window.location.hash.slice(1) || 'dashboard';
    navigate(hash);
}

window.addEventListener('hashchange', handleRoute);

// ============================================================
// Dashboard Page
// ============================================================
async function renderDashboard(container) {
    container.innerHTML = `
        <div class="page-header">
            <h1>Dashboard</h1>
            <p>ภาพรวมระบบ Intelligent Financial Data Agent</p>
        </div>

        <div class="stats-grid" id="dashboard-stats">
            <div class="stat-card">
                <div class="stat-label">Documents</div>
                <div class="stat-value" id="stat-docs">—</div>
                <div class="stat-change positive">Total uploaded</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Chunks</div>
                <div class="stat-value" id="stat-chunks">—</div>
                <div class="stat-change positive">Semantic + LLM</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Table Rows</div>
                <div class="stat-value" id="stat-tables">—</div>
                <div class="stat-change positive">Structured Data</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">System</div>
                <div class="stat-value" style="font-size:1.2rem;color:var(--accent-success)">Online</div>
                <div class="stat-change positive">All services running</div>
            </div>
        </div>

        <div class="grid-2">
            <div class="card">
                <div class="card-header">
                    <div>
                        <div class="card-title">Recent Documents</div>
                        <div class="card-subtitle">Last uploaded files</div>
                    </div>
                    <a href="#documents" class="btn btn-secondary btn-sm">View All</a>
                </div>
                <div id="recent-docs-list">
                    <div class="skeleton" style="height:120px;width:100%"></div>
                </div>
            </div>
            <div class="card">
                <div class="card-header">
                    <div>
                        <div class="card-title">Pipeline Overview</div>
                        <div class="card-subtitle">Data processing pipeline</div>
                    </div>
                </div>
                <div style="padding:16px 0;color:var(--text-secondary);font-size:0.88rem;line-height:2">
                    <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
                        <span class="badge badge-info">1</span> Web Scraping / Upload
                    </div>
                    <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
                        <span class="badge badge-primary">2</span> Typhoon OCR (Layout + Tables)
                    </div>
                    <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
                        <span class="badge badge-success">3</span> Thai Text Cleaning (PyThaiNLP)
                    </div>
                    <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
                        <span class="badge badge-warning">4</span> Semantic + LLM Chunking
                    </div>
                    <div style="display:flex;align-items:center;gap:10px">
                        <span class="badge badge-primary">5</span> Hybrid RAG (Vector + SQL)
                    </div>
                </div>
            </div>
        </div>
    `;

    // Load stats
    try {
        const data = await api.get('/documents');
        const docs = data.documents || [];
        document.getElementById('stat-docs').textContent = docs.length;

        let totalChunks = 0;
        let totalTables = 0;
        docs.forEach(d => {
            totalChunks += d.chunk_count || 0;
            totalTables += d.table_row_count || 0;
        });
        document.getElementById('stat-chunks').textContent = totalChunks;
        document.getElementById('stat-tables').textContent = totalTables;

        // Recent docs
        const recentList = document.getElementById('recent-docs-list');
        if (docs.length === 0) {
            recentList.innerHTML = `
                <div class="empty-state" style="padding:24px">
                    <p>No documents yet. Upload or scrape to get started.</p>
                </div>
            `;
        } else {
            recentList.innerHTML = docs.slice(0, 5).map(d => `
                <div style="display:flex;align-items:center;justify-content:space-between;padding:10px 0;border-bottom:1px solid var(--border-primary)">
                    <div>
                        <div style="font-size:0.88rem;font-weight:500">${d.filename}</div>
                        <div style="font-size:0.75rem;color:var(--text-muted)">${d.chunk_count} chunks · ${d.table_row_count} rows</div>
                    </div>
                    <span class="badge ${d.status === 'completed' ? 'badge-success' : d.status === 'failed' ? 'badge-danger' : 'badge-warning'}">${d.status}</span>
                </div>
            `).join('');
        }
    } catch (e) {
        document.getElementById('stat-docs').textContent = '—';
        document.getElementById('stat-chunks').textContent = '—';
        document.getElementById('stat-tables').textContent = '—';
    }
}

// ============================================================
// Initialize
// ============================================================
document.addEventListener('DOMContentLoaded', () => {
    handleRoute();

    // System health check
    api.get('/health')
        .then(() => {
            document.querySelector('.status-dot').style.background = 'var(--accent-success)';
        })
        .catch(() => {
            document.querySelector('.status-dot').style.background = 'var(--accent-danger)';
            document.querySelector('#system-status span').textContent = 'System Offline';
        });
});
