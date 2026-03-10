/**
 * Documents Component
 * Upload drag-and-drop zone + document list with processing status
 */

function renderDocuments(container) {
    container.innerHTML = `
        <div class="page-header">
            <h1>Documents</h1>
            <p>อัปโหลดและจัดการเอกสารการเงิน (PDF / Image)</p>
        </div>

        <!-- Upload Zone -->
        <div class="card" style="margin-bottom:24px">
            <div class="card-header">
                <div class="card-title">Upload Document</div>
            </div>
            <div class="upload-zone" id="upload-zone">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                    <polyline points="17 8 12 3 7 8"/>
                    <line x1="12" y1="3" x2="12" y2="15"/>
                </svg>
                <p>Drag & drop files here or <strong style="color:var(--accent-primary-light)">browse</strong></p>
                <p class="upload-hint">Supports PDF, PNG, JPG, WEBP, TIFF</p>
                <input type="file" id="file-input" accept=".pdf,.png,.jpg,.jpeg,.webp,.tiff" multiple>
            </div>

            <!-- Upload progress -->
            <div id="upload-progress" style="margin-top:16px;display:none">
                <div class="loader">
                    <span class="loader-spinner"></span>
                    <span id="upload-status-text">Processing document...</span>
                </div>
            </div>
        </div>

        <!-- Document List -->
        <div class="card">
            <div class="card-header">
                <div>
                    <div class="card-title">All Documents</div>
                    <div class="card-subtitle" id="doc-count-label">Loading...</div>
                </div>
                <button class="btn btn-secondary btn-sm" onclick="loadDocumentList()">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg>
                    Refresh
                </button>
            </div>
            <div id="doc-list">
                <div class="skeleton" style="height:200px;width:100%"></div>
            </div>
        </div>
    `;

    // Setup drag & drop
    const zone = document.getElementById('upload-zone');
    const input = document.getElementById('file-input');

    zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.classList.add('dragover');
    });

    zone.addEventListener('dragleave', () => {
        zone.classList.remove('dragover');
    });

    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) uploadFiles(files);
    });

    input.addEventListener('change', (e) => {
        if (e.target.files.length > 0) uploadFiles(e.target.files);
    });

    loadDocumentList();
}

async function uploadFiles(files) {
    const progress = document.getElementById('upload-progress');
    const statusText = document.getElementById('upload-status-text');
    progress.style.display = 'block';

    for (let i = 0; i < files.length; i++) {
        statusText.textContent = `Processing ${files[i].name} (${i + 1}/${files.length})...`;

        try {
            const result = await api.upload('/documents/upload', files[i]);
            showToast(
                `${files[i].name}: ${result.chunks_created} chunks, ${result.tables_extracted} tables`,
                'success'
            );
        } catch (e) {
            showToast(`Failed to process ${files[i].name}: ${e.message}`, 'error');
        }
    }

    progress.style.display = 'none';
    loadDocumentList();
}

async function loadDocumentList() {
    const listEl = document.getElementById('doc-list');
    const countLabel = document.getElementById('doc-count-label');

    try {
        const data = await api.get('/documents');
        const docs = data.documents || [];
        countLabel.textContent = `${docs.length} documents`;

        if (docs.length === 0) {
            listEl.innerHTML = `
                <div class="empty-state">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                        <polyline points="14 2 14 8 20 8"/>
                    </svg>
                    <h3>No documents yet</h3>
                    <p>Upload a PDF or image to get started with OCR extraction and chunking.</p>
                </div>
            `;
            return;
        }

        listEl.innerHTML = `
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Filename</th>
                        <th>Type</th>
                        <th>Status</th>
                        <th>Chunks</th>
                        <th>Tables</th>
                        <th>Date</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
                    ${docs.map(doc => `
                        <tr>
                            <td style="font-weight:500">${doc.filename}</td>
                            <td><span class="badge badge-primary">${doc.doc_type?.toUpperCase() || '—'}</span></td>
                            <td>
                                <span class="badge ${doc.status === 'completed' ? 'badge-success' : doc.status === 'failed' ? 'badge-danger' : 'badge-warning'}">
                                    ${doc.status}
                                </span>
                            </td>
                            <td>${doc.chunk_count}</td>
                            <td>${doc.table_row_count}</td>
                            <td style="font-size:0.8rem;color:var(--text-muted)">
                                ${doc.created_at ? new Date(doc.created_at).toLocaleDateString('th-TH') : '—'}
                            </td>
                            <td style="display:flex;gap:6px">
                                <button class="btn btn-secondary btn-sm" onclick="viewDocChunks(${doc.id})">Chunks</button>
                                <button class="btn btn-danger btn-sm" onclick="deleteDoc(${doc.id})">Delete</button>
                            </td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        `;
    } catch (e) {
        listEl.innerHTML = `
            <div class="empty-state" style="padding:24px">
                <p style="color:var(--accent-danger)">Failed to load documents: ${e.message}</p>
            </div>
        `;
    }
}

function viewDocChunks(docId) {
    window.location.hash = `visualizer`;
    // Store selected doc ID for visualizer
    sessionStorage.setItem('selectedDocId', docId);
}

async function deleteDoc(docId) {
    if (!confirm('Delete this document and all associated data?')) return;

    try {
        await api.delete(`/documents/${docId}`);
        showToast('Document deleted', 'success');
        loadDocumentList();
    } catch (e) {
        showToast(`Delete failed: ${e.message}`, 'error');
    }
}
