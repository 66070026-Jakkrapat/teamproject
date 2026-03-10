/**
 * Web Scraping Component
 * Simple keyword input + max Google results slider
 */

function renderScraping(container) {
    container.innerHTML = `
        <div class="page-header">
            <h1>Web Scraping</h1>
            <p>ใส่ keyword แล้วระบบจะค้นหาผ่าน Google และดึงเอกสาร PDF/รูปภาพจากเว็บที่เกี่ยวข้องแบบอัตโนมัติ</p>
        </div>

        <!-- Simple keyword scraping card -->
        <div class="card" style="margin-bottom:24px">
            <div class="card-header">
                <div>
                    <div class="card-title">🔍 Scrape by Keyword (Google Search)</div>
                    <div class="card-subtitle">ใส่ keyword และกำหนดจำนวนเว็บผลลัพธ์จาก Google ที่ต้องการดึงข้อมูล</div>
                </div>
            </div>

            <!-- Keyword input -->
            <div class="input-group">
                <label for="scrape-keyword">Keyword</label>
                <input type="text" id="scrape-keyword" class="input-field" placeholder="เช่น เทรนด์ธุรกิจ SME 2025, งบการเงิน PTT">
            </div>

            <!-- Max sites slider -->
            <div class="input-group">
                <label>จำนวนเว็บจาก Google ที่จะ scrape: <strong id="max-sites-label">3</strong></label>
                <input type="range" id="max-sites-slider" min="1" max="10" value="3" class="input-field"
                    style="padding:8px 0;cursor:pointer;accent-color:var(--accent-primary)"
                    oninput="document.getElementById('max-sites-label').textContent = this.value">
                <small class="text-muted" style="display:block;margin-top:4px;font-size:0.75rem">
                    ระบบจะค้นหาใน Google และไล่กดเข้าไปทีละเว็บ (สูงสุดไม่เกินตัวเลขที่เลือก)
                </small>
            </div>

            <!-- Max files per site -->
            <div class="input-group" style="margin-top:20px">
                <label>จำนวนไฟล์เอกสารสูงสุดต่อเว็บ: <strong id="max-files-label">10</strong></label>
                <input type="range" id="max-files-slider" min="1" max="30" value="10" class="input-field"
                    style="padding:8px 0;cursor:pointer;accent-color:var(--accent-primary)"
                    oninput="document.getElementById('max-files-label').textContent = this.value">
            </div>

            <button class="btn btn-primary" id="btn-scrape" onclick="startScraping()" style="width:100%;justify-content:center;padding:14px;margin-top:16px">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
                ค้นหาใน Google และเริ่ม Scraping
            </button>
        </div>

        <!-- Progress -->
        <div class="card" id="scrape-progress-card" style="display:none;margin-bottom:24px">
            <div class="card-header">
                <div class="card-title">⏳ Scraping Progress</div>
            </div>
            <div id="scrape-progress">
                <div class="loader">
                    <span class="loader-spinner"></span>
                    <span id="scrape-progress-text">กำลังเริ่มต้นค้นหาใน Google...</span>
                </div>
            </div>
        </div>

        <!-- Results -->
        <div class="card" id="scrape-results-card" style="display:none">
            <div class="card-header">
                <div class="card-title">📦 Scraping Results</div>
            </div>
            <div id="scrape-results"></div>
        </div>
    `;
}

async function startScraping() {
    const keyword = document.getElementById('scrape-keyword').value.trim();
    if (!keyword) {
        showToast('กรุณาใส่ keyword', 'warning');
        return;
    }

    const maxSites = parseInt(document.getElementById('max-sites-slider').value);
    const maxFiles = parseInt(document.getElementById('max-files-slider').value);

    const btn = document.getElementById('btn-scrape');
    btn.disabled = true;
    btn.innerHTML = '<span class="loader-spinner"></span> กำลังค้นหาและดึงข้อมูล...';

    // Show progress
    const progressCard = document.getElementById('scrape-progress-card');
    const progressText = document.getElementById('scrape-progress-text');
    progressCard.style.display = 'block';
    progressText.textContent = `กำลังค้นหา keyword "${keyword}" ใน Google และ scrape Top ${maxSites} เว็บ...`;

    try {
        const response = await fetch('/api/scrape/keyword', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                keyword: keyword,
                max_sites: maxSites,
                max_files_per_site: maxFiles
            })
        });

        if (!response.ok) {
            throw new Error(`API Error: ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = "";
        let finalResult = null;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            let lines = buffer.split("\n");
            buffer = lines.pop(); // Keep the last incomplete line in buffer
            
            for (let line of lines) {
                if (!line.trim()) continue;
                try {
                    const msg = JSON.parse(line);
                    if (msg.status === "done") {
                        finalResult = msg.result;
                    } else if (msg.message) {
                        // Update progress UI in real-time!
                        progressText.innerHTML = msg.message;
                    }
                } catch (err) {
                    console.error("Parse error on stream chunk:", line, err);
                }
            }
        }

        progressCard.style.display = 'none';
        
        if (finalResult) {
            displayScrapeResults(finalResult);
            showToast(`Scraping เสร็จ! พบ ${finalResult.total_files || 0} ไฟล์จาก ${finalResult.urls_scraped} เว็บ`, 'success');
        } else {
            showToast(`Scraping เสร็จสิ้นแต่ไม่พบผลลัพธ์ที่ถูกต้อง`, 'warning');
        }

    } catch (e) {
        progressCard.style.display = 'none';
        showToast(`Scraping ล้มเหลว: ${e.message}`, 'error');
    } finally {
        btn.disabled = false;
        btn.innerHTML = `
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
            ค้นหาใน Google และเริ่ม Scraping
        `;
    }
}

function displayScrapeResults(result) {
    const card = document.getElementById('scrape-results-card');
    const container = document.getElementById('scrape-results');
    card.style.display = 'block';

    // Per-site results
    const siteResults = (result.results || []).map((r, i) => {
        const statusColor = r.success ? 'var(--accent-success)' : 'var(--accent-danger)';
        const statusIcon = r.success ? '✅' : '❌';
        const filesCount = r.files?.length || 0;
        const imagesCount = r.images?.length || 0;
        const linksCount = r.links_found?.length || 0;
        const contentLen = r.content_length || r.page_text?.length || 0;
        const url = r.source_url || r.url || result.urls?.[i] || '—';
        const title = r.search_title || r.title || '';
        const textPreview = (r.page_text || '').substring(0, 300);

        return `
            <div style="padding:16px;background:var(--bg-tertiary);border-radius:var(--radius-md);border-left:3px solid ${statusColor}">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
                    <div>
                        <div style="font-size:0.9rem;font-weight:600">${statusIcon} ${title || (url.length > 55 ? url.slice(0, 55) + '...' : url)}</div>
                        <a href="${url}" target="_blank" style="font-size:0.75rem;color:var(--text-muted);text-decoration:none">${url.length > 70 ? url.slice(0, 70) + '...' : url}</a>
                    </div>
                </div>
                <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px">
                    <span class="badge badge-primary">📄 ${filesCount} files</span>
                    <span class="badge badge-info">🖼️ ${imagesCount} images</span>
                    <span class="badge badge-success">📝 ${contentLen.toLocaleString()} chars</span>
                    <span class="badge" style="background:var(--bg-secondary);color:var(--text-secondary)">🔗 ${linksCount} links</span>
                </div>
                ${textPreview ? `
                    <details style="margin-top:8px">
                        <summary style="font-size:0.78rem;color:var(--text-secondary);cursor:pointer">ดูตัวอย่างเนื้อหา</summary>
                        <div style="margin-top:8px;font-size:0.78rem;color:var(--text-muted);background:var(--bg-secondary);padding:12px;border-radius:var(--radius-sm);white-space:pre-wrap;max-height:200px;overflow-y:auto">${textPreview}${contentLen > 300 ? '...' : ''}</div>
                    </details>
                ` : ''}
                ${r.error ? `<div style="font-size:0.78rem;color:var(--accent-danger);margin-top:4px">Error: ${r.error}</div>` : ''}
            </div>
        `;
    }).join('');

    container.innerHTML = `
        <div style="display:flex;gap:12px;margin-bottom:16px;flex-wrap:wrap">
            <span class="badge badge-success" style="font-size:0.85rem;padding:6px 14px">
                🔍 keyword: "${result.keyword}"
            </span>
            <span class="badge badge-info" style="font-size:0.85rem;padding:6px 14px">
                🌐 ${result.urls_scraped} เว็บ
            </span>
            <span class="badge badge-primary" style="font-size:0.85rem;padding:6px 14px">
                📄 ${result.total_files || 0} ไฟล์
            </span>
            <span class="badge" style="font-size:0.85rem;padding:6px 14px;background:var(--accent-primary);color:white">
                🖼️ ${result.total_images || 0} รูปภาพ
            </span>
        </div>

        ${result.output_folder ? `
            <div style="font-size:0.78rem;color:var(--text-muted);margin-bottom:16px;padding:8px 12px;background:var(--bg-tertiary);border-radius:var(--radius-sm)">
                📁 ข้อมูลถูกบันทึกที่: <code style="color:var(--accent-primary)">${result.output_folder}</code>
            </div>
        ` : ''}

        <div style="display:flex;flex-direction:column;gap:10px;margin-bottom:20px">
            ${siteResults}
        </div>

        ${result.files && result.files.length > 0 ? `
            <div style="margin-top:16px">
                <div style="font-size:0.9rem;font-weight:600;margin-bottom:10px">Downloaded Files</div>
                <table class="data-table">
                    <thead>
                        <tr><th>File</th><th>Type</th></tr>
                    </thead>
                    <tbody>
                        ${result.files.map(f => {
                            const name = f.split(/[/\\]/).pop();
                            const ext = name.split('.').pop().toUpperCase();
                            return `<tr>
                                <td style="font-size:0.85rem">${name}</td>
                                <td><span class="badge badge-primary">${ext}</span></td>
                            </tr>`;
                        }).join('')}
                    </tbody>
                </table>
            </div>
        ` : ''}
    `;
}

