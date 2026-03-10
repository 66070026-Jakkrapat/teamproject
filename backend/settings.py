# backend/settings.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH)

@dataclass
class Settings:
    # API
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))

    # Output dirs
    OUTPUT_BASE_DIR: str = os.getenv("OUTPUT_BASE_DIR", str(PROJECT_ROOT / "scraped_outputs"))
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", str(PROJECT_ROOT / "tmp_uploads"))

    # DB / RAG
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://postgres:2026@127.0.0.1:5436/ragdb"
    )
    EMBED_MODEL: str = os.getenv("EMBED_MODEL", "nomic-embed-text:latest")
    EMBED_DIMS: int = int(os.getenv("EMBED_DIMS", "768"))
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    OLLAMA_LLM_MODEL: str = os.getenv("OLLAMA_LLM_MODEL", "llama3.1:8b")

    # Vision caption
    BLIP_MODEL: str = os.getenv("BLIP_MODEL", "Salesforce/blip-image-captioning-base")
    ARGOS_SRC_LANG: str = os.getenv("ARGOS_SRC_LANG", "en")
    ARGOS_TGT_LANG: str = os.getenv("ARGOS_TGT_LANG", "th")

    # Browser (Playwright/Drission)
    BROWSER_HEADLESS: bool = os.getenv("BROWSER_HEADLESS", "false").lower() in ("1", "true", "yes")
    BROWSER_SLOWMO_MS: int = int(os.getenv("BROWSER_SLOWMO_MS", "0"))

    # Facebook
    FB_EMAIL: str = os.getenv("FB_EMAIL", "")
    FB_PASS: str = os.getenv("FB_PASS", "")
    FB_STORAGE_STATE_PATH: str = os.getenv(
        "FB_STORAGE_STATE_PATH",
        str(PROJECT_ROOT / "backend" / "scraping" / "storage_state_facebook.json")
    )

    # Tavily
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    TAVILY_MAX_RESULTS: int = int(os.getenv("TAVILY_MAX_RESULTS", "5"))

    # Pipeline
    WAIT_TIMEOUT_SEC: int = int(os.getenv("WAIT_TIMEOUT_SEC", "1800"))
    WAIT_POLL_SEC: float = float(os.getenv("WAIT_POLL_SEC", "1.5"))
    DISABLE_MODEL_SOURCE_CHECK: bool = os.getenv("DISABLE_MODEL_SOURCE_CHECK", "true").lower() in ("1", "true", "yes")

    # Agent
    ROUTE_PREFER_INTERNAL: bool = os.getenv("ROUTE_PREFER_INTERNAL", "true").lower() in ("1", "true", "yes")
    RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "8"))

    # MLflow
    MLFLOW_ENABLED: bool = os.getenv("MLFLOW_ENABLED", "false").lower() in ("1", "true", "yes")
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "")
    MLFLOW_EXPERIMENT: str = os.getenv("MLFLOW_EXPERIMENT", "thai-business-insight-ai")
    MLFLOW_PROMPT_EXPERIMENT: str = os.getenv("MLFLOW_PROMPT_EXPERIMENT", "thai-business-insight-prompts")
    MLFLOW_PIPELINE_NAME: str = os.getenv("MLFLOW_PIPELINE_NAME", "thai-business-insight-pipeline")
    MLFLOW_ARTIFACT_LOCATION: str = os.getenv("MLFLOW_ARTIFACT_LOCATION", "")
    MLFLOW_TAG_ENV: str = os.getenv("MLFLOW_TAG_ENV", "local")
    MLFLOW_REGISTER_PIPELINE: bool = os.getenv("MLFLOW_REGISTER_PIPELINE", "false").lower() in ("1", "true", "yes")

    # Hybrid worker deployment
    HYBRID_WORKER_ENABLED: bool = os.getenv("HYBRID_WORKER_ENABLED", "false").lower() in ("1", "true", "yes")
    WORKER_BASE_URL: str = os.getenv("WORKER_BASE_URL", "").rstrip("/")
    WORKER_SHARED_SECRET: str = os.getenv("WORKER_SHARED_SECRET", "")
    WORKER_TIMEOUT_SEC: int = int(os.getenv("WORKER_TIMEOUT_SEC", "180"))

settings = Settings()
