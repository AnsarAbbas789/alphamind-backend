"""
AlphaMind — llm/registry.py
=====================================================================
جو key .env میں خالی/غائب ہو، وہ provider خودکار طور پر skip ہو جاتا
ہے — سسٹم صرف موجود keys کے ساتھ چلتا رہتا ہے، کبھی crash نہیں ہوتا۔
اسی لیے آپ ابھی 2 keys کے ساتھ بھی چلا سکتے ہیں، بعد میں 3 اور keys
ڈالتے ہی وہ خودکار ensemble میں شامل ہو جائیں گی — کوڈ میں کوئی
تبدیلی درکار نہیں۔
=====================================================================
"""

from __future__ import annotations

from app.config import Settings
from app.llm.base import BaseLLMProvider
from app.llm.providers import (
    CerebrasProvider,
    GeminiProvider,
    GroqProvider,
    MistralProvider,
    OpenRouterProvider,
)


def build_active_providers(settings: Settings) -> list[BaseLLMProvider]:
    candidates = [
        (settings.GEMINI_API_KEY, GeminiProvider),
        (settings.GROQ_API_KEY, GroqProvider),
        (settings.CEREBRAS_API_KEY, CerebrasProvider),
        (settings.MISTRAL_API_KEY, MistralProvider),
        (settings.OPENROUTER_API_KEY, OpenRouterProvider),
    ]
    active: list[BaseLLMProvider] = []
    for key, provider_cls in candidates:
        if key and key.strip() and "PASTE" not in key.upper():
            active.append(provider_cls(api_key=key.strip()))
    return active