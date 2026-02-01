# AI Agent: Web Scraping + OCR + Dual RAG (pgvector) + Agent (Tavily fallback) + Evaluation

โปรเจกต์นี้เป็นระบบ **AI Agent** ที่ทำงานแบบ end-to-end:

- **External (ระบบไปเก็บเอง):** ค้นเว็บ → scrape เนื้อหา/รูป/ไฟล์ PDF → caption รูป (BLIP) + แปลไทย (Argos) → OCR PDF (PyMuPDF + PaddleOCR fallback) → **Ingest เข้า RAG (Postgres + pgvector)**
- **Internal (ผู้ใช้ upload):** upload PDF → OCR → ingest เข้า RAG (namespace=internal)
- **Ask Agent:** ระบบตอบจาก RAG ก่อน (structured/semantic) ถ้าไม่พอค่อย **fallback ไป Tavily** (ใช้เฉพาะตอนถาม ไม่ใช้ในขั้น scraping)

---

## โครงสร้างโปรเจกต์ (สรุป)

```
full_project/
  .env
  .gitignore
  README.md
  requirements.txt
  test.py

  ui/
    index.html
    styles.css
    app.js
    workflow.html
    workflow.js
    swagger_custom.js

  backend/
    settings.py
    main.py
    utils/
    scraping/
    vision/
    ocr/
    rag/
    agent/
    evaluation/
    report_utils/
```

---

## 0) เตรียม .env

สร้างไฟล์ `.env` ที่ root ของโปรเจกต์ (ข้าง README) โดยใช้ตัวอย่างนี้:

```env
# ---- (optional) facebook login ----
FB_EMAIL=
FB_PASS=

# ---- API ----
API_HOST=0.0.0.0
API_PORT=8000

# ---- Paths ----
OUTPUT_BASE_DIR=./scraped_outputs
UPLOAD_DIR=./tmp_uploads

# ---- Vector DB (pgvector) ----
DATABASE_URL=postgresql+asyncpg://postgres:2026@127.0.0.1:5436/ragdb

# ---- Ollama ----
OLLAMA_HOST=http://localhost:11434
EMBED_MODEL=nomic-embed-text:latest
EMBED_DIMS=768
OLLAMA_LLM_MODEL=llama3.1:8b

# ---- Image Caption ----
BLIP_MODEL=Salesforce/blip-image-captioning-base

# ---- Argos translate (en->th) ----
ARGOS_SRC_LANG=en
ARGOS_TGT_LANG=th

# ---- Web Search fallback (only when /ask and rag not enough) ----
TAVILY_API_KEY=

# ---- Optional ----
DISABLE_MODEL_SOURCE_CHECK=True
```

> หมายเหตุ:
> - `TAVILY_API_KEY` **ไม่ต้องใส่** ถ้ายังไม่ใช้ fallback (ถามแล้ว RAG ไม่พอ ระบบจะบอกว่า Tavily ใช้ไม่ได้)
> - `DATABASE_URL` ต้องตรงกับ docker-compose (port 5436 และ database = `ragdb`)

---

## 1) ติดตั้ง Dependencies

### 1.1 Python packages หลัก
```bash
pip install -r requirements.txt
```

### 1.2 Playwright (สำคัญสำหรับ scraping)
```bash
python -m playwright install
```

---

## 2) ติดตั้ง OCR (PaddleOCR + PaddlePaddle)

โปรเจกต์ใช้:
- PyMuPDF สำหรับดึง text layer จาก PDF (เร็ว/แม่น)
- ถ้า text layer น้อย → fallback เป็น **PaddleOCR**

### 2.1 ติดตั้ง PaddleOCR
```bash
pip install paddleocr
```

### 2.2 ติดตั้ง PaddlePaddle (CPU)
> แนะนำวิธีนี้ใน Windows/CPU

```bash
python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
```

### 2.3 (ถ้ามี GPU) PaddlePaddle GPU
```bash
# ตัวอย่าง cu118
python -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
```

---

## 3) ติดตั้ง BLIP Caption (Transformers + Torch CPU)

โปรเจกต์ใช้ BLIP captioner จาก Transformers  
ถ้าเครื่องไม่มี GPU ก็ลง Torch CPU ได้

```bash
pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -U transformers
```

---

## 4) ติดตั้ง Argos Translate (แปล en -> th)

```bash
pip install -U argostranslate
```

> หมายเหตุ: Argos บางเครื่องต้องโหลด language pack ครั้งแรก  
> โค้ดจะพยายาม translate ถ้าติดตั้งไลบรารีสำเร็จ

---

## 5) (Optional) Facebook scraping requirements

ถ้าจะ scrape Facebook (login + reuse session) ให้แน่ใจว่า:

```bash
pip install -U DrissionPage pydub SpeechRecognition
```

และติดตั้ง FFmpeg (Windows):
```bash
winget install Gyan.FFmpeg
```

> หมายเหตุ: Facebook มักมี checkpoint/2FA  
> ถ้า login ไม่ผ่าน ให้รันแบบ `headless=False` ครั้งแรกเพื่อให้กรอกเอง

---

## 6) ติดตั้ง + รัน Ollama

### 6.1 เช็คว่า Ollama ทำงาน
```bash
ollama --version
```

### 6.2 pull models ที่ต้องใช้
Embedding:
```bash
ollama pull nomic-embed-text
```

LLM ตอบคำถาม (เลือก 1):
```bash
ollama pull llama3.1:8b
# หรือรุ่นอื่นที่คุณมี
```

> ให้แน่ใจว่า `.env` ตั้ง:
> - `EMBED_MODEL=nomic-embed-text:latest`
> - `OLLAMA_LLM_MODEL=llama3.1:8b`

---

## 7) รัน Postgres + pgvector (Docker)

### 7.1 Start docker
```bash
docker compose up -d
```

### 7.2 เช็คว่า container ขึ้น
```bash
docker ps
```

> โปรเจกต์คาดว่า Postgres เปิดที่ `127.0.0.1:5436`  
> และมี database ชื่อ `ragdb`  
> ถ้า docker-compose ใช้ชื่อ DB อื่น ให้แก้ `.env` ให้ตรง

---

## 8) รัน API (FastAPI)

```bash
python -m backend.main
```

จะขึ้นข้อความประมาณ:
- Swagger: `http://localhost:8000/docs`
- UI: `http://localhost:8000/ui`
- Workflow UI: `http://localhost:8000/ui/workflow`

---

## 9) เปิดหน้าใช้งาน

- **Swagger:** `http://localhost:8000/docs`
- **UI:** `http://localhost:8000/ui`
- **Workflow UI:** `http://localhost:8000/ui/workflow`

---

## 10) วิธีใช้งาน (External Pipeline)

### 10.1 เริ่ม scrape + OCR + ingest
ในหน้า UI (External Pipeline):
1) ใส่ keyword เช่น `SME ไทย รายงานประจำปี`
2) ใส่ amount เช่น `3`
3) กด **Start Pipeline**
4) ระบบจะได้ `job_id` สำหรับติดตาม

หรือเรียกผ่าน curl:

```bash
curl -X POST http://localhost:8000/pipeline/external/scrape ^
  -H "Content-Type: application/json" ^
  -d "{\"keyword\":\"SME ไทย รายงานประจำปี\", \"amount\":3}"
```

### 10.2 ดูสถานะ pipeline
ไปหน้า: `http://localhost:8000/ui/workflow`  
ใส่ `job_id` แล้วกด Watch

สถานะจะวิ่งตาม step เช่น:
- web_scraping
- data_collecting
- captioning
- ocr
- ingest_external
- ready (สำเร็จ) หรือ error

---

## 11) วิธีใช้งาน (Internal Upload PDF)

### 11.1 Upload PDF แล้ว ingest เข้า internal namespace
ใช้ Swagger:
- endpoint: `POST /pipeline/internal/upload_pdf`
- อัปไฟล์ PDF + ใส่ entity_hint (optional)

เมื่อเสร็จ จะ ingest เข้า RAG โดย namespace = `internal`

---

## 12) ถามคำถาม (Agent /ask)

ไปที่ UI (Ask Agent) หรือเรียกผ่าน API:

```bash
curl -X POST http://localhost:8000/ask ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"บริษัทมีแนวทางด้านความยั่งยืนอย่างไร\", \"top_k\":8}"
```

ระบบจะ:
1) route structured vs semantic
2) ดึงข้อมูลจาก RAG (internal ก่อนหรือ external ก่อนตาม config)
3) ถ้าไม่พอ → fallback Tavily (เฉพาะตอนถามเท่านั้น)

> ถ้า `TAVILY_API_KEY` ไม่ได้ใส่ ระบบจะตอบว่า Tavily ใช้งานไม่ได้ (เป็น expected behavior)

---

## 13) Preview สิ่งที่อยู่ใน RAG (Debug)

ดู chunk ที่ถูกเก็บจริง:

```bash
curl "http://localhost:8000/rag/preview?namespace=external&source_type=&limit=20"
```

หรือดูใน UI section “RAG Preview”

---

## 14) Reset RAG DB (ล้างข้อมูลทั้งหมด)

```bash
curl -X POST http://localhost:8000/rag/reset
```

> ใช้กรณีเปลี่ยน `EMBED_DIMS` หรือ schema แล้วข้อมูลเก่าไม่ตรง

---

## 15) Evaluation (precision/recall@k)

รัน eval dataset ตัวอย่าง:

```bash
python test.py eval --dataset backend/evaluation/eval_dataset.example.json --namespace external --k 5
```

---

## 16) Troubleshooting

### 16.1 ต่อ DB ไม่ได้ / database ไม่พบ
- เช็ค docker `ports` และ `POSTGRES_DB`
- เช็ค `.env` ว่า `DATABASE_URL` ตรง
- ถ้า DB ชื่อ `postgres` แต่ `.env` ใช้ `ragdb` → เปลี่ยนให้ตรงกัน

### 16.2 embedding dimension mismatch
อาการ: insert/query พัง
- ตรวจ `EMBED_DIMS` ใน `.env`
- ต้องสอดคล้องกับ embed model
- แก้แล้วให้ `/rag/reset` (หรือล้างตาราง)

### 16.3 BLIP/torch ช้า
- CPU จะช้าเป็นปกติ
- ลดจำนวนรูป (`max_images`) หรือปรับ threshold keep_for_rag
- หรือเปลี่ยนไป caption model ที่เบากว่า (ภายหลังค่อย optimize)

### 16.4 Playwright error
- ต้อง `python -m playwright install` ก่อน
- ถ้า Windows policy/antivirus block ให้ลอง run as admin

### 16.5 OCR ช้า/กินเครื่อง
- PDF ใหญ่ + dpi=300 จะหนัก
- ลด OCR_DPI ใน `backend/ocr/ocr_pipeline.py` (เช่น 200) ถ้าต้องการเร็วขึ้น

---

## 17) Concept Workflow (ภาพรวม)

**External:**
Web Scraping → collect text/images/pdfs → Image caption (BLIP+Argos) → OCR PDFs (PyMuPDF + PaddleOCR fallback) → Ingest external (semantic + structured) → Ask

**Internal:**
Upload PDF → OCR → Ingest internal → Ask

**Ask:**
Structured RAG (facts) หรือ Semantic RAG (summary) → ถ้าไม่พอ → Tavily fallback

---

## Architecture Diagram (Mermaid)

```mermaid
flowchart LR
  U[User] -->|Ask| API[FastAPI Agent API]
  API --> R[Router: structured vs semantic]

  subgraph ExternalPipeline[External Pipeline (Auto)]
    G[Web Scraper] --> W[Web Text + Images + PDFs]
    W --> C[Image Caption BLIP + Argos TH]
    W --> O[OCR PDFs: PyMuPDF + PaddleOCR]
    C --> IE[Ingest External RAG]
    O --> IE
    W --> IE
  end

  subgraph InternalPipeline[Internal Pipeline (User Upload)]
    UP[Upload PDF] --> IO[OCR + Clean]
    IO --> II[Ingest Internal RAG]
  end

  R -->|semantic| SR[(Semantic RAG pgvector\nrag_chunks)]
  R -->|structured| FR[(Structured RAG SQL\nrag_facts)]

  SR --> A[Answer Synthesizer LLM]
  FR --> A

  A -->|if insufficient| T[Tavily Search]
  T --> A

  A --> OUT[Answer + Retrieved Chunks]
  API --> UI[UI Workflow Stepper + Logs]
```
