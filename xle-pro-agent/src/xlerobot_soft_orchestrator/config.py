"""Configuration and environment loading."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for local/dev and Docker execution."""

    llm_provider: str = "vertex"
    llm_model: str = ""

    vertex_project_id: str = ""
    vertex_location: str = "us-central1"
    vertex_model: str = "gemini-2.5-flash"
    google_application_credentials: str = ""

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    langsmith_tracing: bool = True
    langsmith_api_key: str = ""
    langsmith_project: str = "xlerobot-orchestrator"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    max_iterations: int = 20


def get_settings() -> Settings:
    """Load environment variables and return validated settings."""
    load_dotenv()
    settings = Settings()
    if settings.google_application_credentials:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.google_application_credentials
    return settings


def configure_langsmith(settings: Settings) -> None:
    """Set LangSmith tracing env vars if enabled."""
    if not settings.langsmith_tracing:
        return
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    if settings.langsmith_api_key:
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
