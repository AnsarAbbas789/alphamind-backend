"""
AlphaMind AI Signal Engine — config.py
=====================================================================
تمام settings ایک جگہ سے آتی ہیں — کہیں بھی کوڈ میں key ہارڈ کوڈ
نہیں ہوگی۔ اگر کوئی LLM key خالی ہو تو وہ provider خودکار طور پر
ensemble سے باہر رہ جاتا ہے (کریش نہیں ہوتا) — دیکھیں llm/registry.py۔
=====================================================================
"""

from functools import lru_cache
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM provider keys (any may be blank) ---
    GEMINI_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    CEREBRAS_API_KEY: Optional[str] = None
    MISTRAL_API_KEY: Optional[str] = None
    OPENROUTER_API_KEY: Optional[str] = None

    # --- Supabase (JWT verification) ---
    SUPABASE_URL: Optional[str] = None
    SUPABASE_JWT_SECRET: Optional[str] = None

    # --- Exchange data ---
    DEFAULT_EXCHANGE: str = "bybit"

    # --- Server ---
    ALLOWED_ORIGINS: str = "http://localhost:5500"
    ENVIRONMENT: str = "production"
    LOG_LEVEL: str = "INFO"

    @property
    def allowed_origins_list(self) -> List[str]:
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]

    @property
    def is_dev(self) -> bool:
        return self.ENVIRONMENT.lower() in ("dev", "development", "local")


@lru_cache
def get_settings() -> Settings:
    return Settings()