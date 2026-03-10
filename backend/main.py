"""FastAPI main application with lifespan, CORS, and static file serving."""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.database import init_db

from backend.routes.documents import router as documents_router
from backend.routes.scraping import router as scraping_router
from backend.routes.query import router as query_router
from backend.routes.chunks import router as chunks_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup."""
    await init_db()
    yield


app = FastAPI(
    title="Intelligent Financial Data Agent",
    description="Thai financial document extraction & hybrid RAG system",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Routes
app.include_router(documents_router, prefix="/api/documents", tags=["Documents"])
app.include_router(scraping_router, prefix="/api/scrape", tags=["Scraping"])
app.include_router(query_router, prefix="/api/query", tags=["Query"])
app.include_router(chunks_router, prefix="/api/chunks", tags=["Chunks"])


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "service": "Financial Data Agent"}


# Serve frontend static files
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")

if os.path.isdir(FRONTEND_DIR):
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="frontend_assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        """Serve the SPA index.html for any non-API route."""
        file_path = os.path.join(FRONTEND_DIR, full_path)
        if full_path and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


if __name__ == "__main__":
    import uvicorn
    from backend.config import get_settings
    
    settings = get_settings()
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)


