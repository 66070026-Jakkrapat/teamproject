workflow:

✅ Scrape web → เก็บ text/pdf/image

✅ OCR PDF + OCR รูป

✅ Vision understanding (caption/labels/relevance/keep_for_rag)

✅ Ingest/Chunk/Clean → Embed → Index เข้า Chroma (Vector DB)

✅ Agentic RAG (ReAct) ด้วย LangGraph: Router → Retrieve → Grader → (Fallback Web Search) → Synthesizer

✅ Output 2 แบบ: Summary + Dashboard JSON

✅ FastAPI endpoints ครบ: /scrape, /upload, /index, /ask, /dashboard/{job_id}, /health

✅ Preflight logs เช็คทุกอย่าง (Playwright/OCR/Ollama/Vision/Embed/Tavily/Chroma)

AI Web Scraping + OCR Pipeline (FastAPI)

โปรเจคนี้เป็นระบบ Web Scraping + OCR + (Image Caption/Relevance) เพื่อดึงข้อมูลจากเว็บ (ข้อความ, ไฟล์, รูปภาพ) แล้วเก็บเป็นโครงสร้างที่พร้อมเอาไปทำ RAG / AI Agent / Dashboard ต่อได้


**Features**


1) Web Scraping (Playwright)
- ค้นเว็บจาก keyword (Google Search ผ่าน Playwright)
- เข้าเว็บทีละลิงก์ แล้วดึงข้อความหลัก (main content)
- ดาวน์โหลดรูปภาพ + ไฟล์ (เช่น PDF, Excel, Docx, PPTX ฯลฯ)

2) AI Image Caption + Relevance (Ollama Vision)
- หลังดาวน์โหลดรูป จะเรียก Ollama Vision model (เช่น MiniCPM-V 2.6) เพื่อทำ caption
- ประเมินความเกี่ยวข้องรูปกับเนื้อหาเว็บด้วย TF-IDF + cosine similarity
- เก็บ metadata เป็น images_meta.jsonl

3) OCR Background Job (Queue + Worker)
- OCR PDFs:
  - ถ้า PDF มี text layer เยอะ -> ใช้ text layer (เร็ว)
  - ถ้า text layer น้อย/เป็นสแกน -> render เป็นรูป แล้ว OCR ด้วย PaddleOCR
  - extract รูปใน PDF (กราฟ/ตาราง) แล้ว OCR เพิ่ม
- OCR รูปจากเว็บ:
  - OCR รูป (jpg/png/webp) ด้วย PaddleOCR (ไทยหลัก + fallback อังกฤษ)
- เช็คสถานะ OCR ได้ที่ /ocr/status/{job_id}

**Project Structure**


- web_scraping.py   : FastAPI + scraper + download images/files + caption/relevance + OCR queue
- ocr_pipeline.py   : OCR pipeline สำหรับ PDF + รูป (PaddleOCR + PyMuPDF)
- .env              : ตั้งค่า Ollama host/model (ไม่ควร commit)


**Output Folder Structure (ตัวอย่าง)**


(ค่า default ปัจจุบัน สร้างใน working directory)
data_<keyword>_<timestamp>/
- final_data.csv
- <site_folder>/
  - content.txt
  - images/
  - images_meta.jsonl
  - files/
- ocr_results/
  - <pdf_name>/docs.jsonl
  - web_images/images.jsonl


**Requirements**


Python packages:
fastapi, uvicorn, playwright, beautifulsoup4, requests, pandas, pillow, python-dotenv,
scikit-learn, pymupdf, numpy, paddleocr

ติดตั้ง:
pip install fastapi uvicorn playwright beautifulsoup4 requests pandas pillow python-dotenv scikit-learn pymupdf numpy paddleocr

ติดตั้ง browser ของ Playwright:
playwright install


**Environment Variables (.env)**


ตัวอย่าง .env:
OLLAMA_HOST=http://localhost:11434
OLLAMA_VISION_MODEL=minicpm-v:2.6


**Run Server**


python web_scraping.py

หรือ:
uvicorn web_scraping:app --host 0.0.0.0 --port 8000 --reload

Swagger:
http://localhost:8000/docs


**API**


1) GET /
health check

2) GET /scrape?keyword=...&amount=5
trigger scrape

3) GET /ocr/status/{job_id}
เช็คสถานะ OCR background
