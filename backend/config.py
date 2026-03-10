"""Application configuration loaded from .env file."""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:2026@localhost:5436/ragdb"
    DATABASE_URL_SYNC: str = "postgresql://postgres:2026@localhost:5436/ragdb"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # Ollama
    OLLAMA_HOST: str = "http://localhost:11434"
    EMBED_MODEL: str = "nomic-embed-text:latest"
    OLLAMA_LLM_MODEL: str = "llama3.1:8b"

    # Typhoon OCR
    TYPHOON_API_KEY: str = ""
    TYPHOON_OCR_ENDPOINT: str = "https://api.opentyphoon.ai/v1/chat/completions"
    TYPHOON_OCR_MODEL: str = "typhoon-ocr"

    # App
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
