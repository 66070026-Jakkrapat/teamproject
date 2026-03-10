"""Scraping API routes."""

import json
from typing import Optional, List
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.services.scraper import scrape_url, scrape_by_keyword

router = APIRouter()


class ScrapeURLRequest(BaseModel):
    url: str
    download_pdfs: bool = True

    download_images: bool = True
    max_files: int = 20


class ScrapeKeywordRequest(BaseModel):
    keyword: str
    max_sites: int = 3
    max_files_per_site: int = 10


@router.post("/url")
async def scrape_single_url(request: ScrapeURLRequest):
    """Scrape a single URL for PDFs and images."""
    result = await scrape_url(
        url=request.url,
        download_pdfs=request.download_pdfs,
        download_images=request.download_images,
        max_files=request.max_files,
    )
    return result


@router.post("/keyword")
async def scrape_by_keyword_endpoint(request: ScrapeKeywordRequest):
    """Search by keyword on Google and scrape top N results (Streams progress)."""
    async def event_generator():
        async for chunk in scrape_by_keyword(
            keyword=request.keyword,
            max_sites=request.max_sites,
            max_files_per_site=request.max_files_per_site,
        ):
            yield json.dumps(chunk, ensure_ascii=False) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")

