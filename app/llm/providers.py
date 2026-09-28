"""
AlphaMind — llm/providers.py
=====================================================================
پانچوں LLM providers کے adapters۔ ہر ایک صرف اپنی API کی خاص شکل
(request/response format) سنبھالتا ہے — باقی سب منطق base.py میں
مشترک ہے۔ نیا provider شامل کرنا ہو تو صرف یہاں ایک نئی کلاس بڑھائیں۔
=====================================================================
"""

from __future__ import annotations

import httpx

from app.llm.base import BaseLLMProvider, RetryableError, TIMEOUT_SECONDS, extract_json

SYSTEM_INSTRUCTION = (
    "You are a professional crypto trading analyst. You will be given pre-computed "
    "technical indicators and a mechanically-derived bias score. Your job is to "
    "review them and respond with ONLY a JSON object: "
    '{"signal": "LONG" or "SHORT" or "NEUTRAL", "confidence": 0-100, "reasoning": "2 sentences"}. '
    "You MUST follow the mandatory bias-score rule given in the prompt — you may only "
    "deviate if the raw candle data shows an extremely obvious reversal pattern, and if "
    "you do, explain why in reasoning. Respond with JSON only, no markdown, no extra text."
)


# ---------------------------------------------------------------------
# 1) Google Gemini
# ---------------------------------------------------------------------
class GeminiProvider(BaseLLMProvider):
    name = "gemini"
    URL = URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent"

    async def _call(self, client: httpx.AsyncClient, prompt: str) -> dict:
        try:
            resp = await client.post(
                self.URL,
                headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
                json={
                    "system_instruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.2},
                },
                timeout=TIMEOUT_SECONDS,
            )
        except httpx.TimeoutException as exc:
            raise RetryableError("gemini_timeout") from exc
        except httpx.TransportError as exc:
            raise RetryableError("gemini_network_error") from exc

        if resp.status_code >= 500:
            raise RetryableError(f"gemini_{resp.status_code}")
        if resp.status_code in (401, 403):
            raise ValueError("gemini_invalid_key")
        if resp.status_code == 429:
            raise ValueError("gemini_rate_limited")
        resp.raise_for_status()

        d = resp.json()
        text = d["candidates"][0]["content"]["parts"][0]["text"]
        return extract_json(text)


# ---------------------------------------------------------------------
# 2) Groq
# ---------------------------------------------------------------------
class GroqProvider(BaseLLMProvider):
    name = "groq"
    URL = "https://api.groq.com/openai/v1/chat/completions"
    MODEL = "llama-3.3-70b-versatile"

    async def _call(self, client: httpx.AsyncClient, prompt: str) -> dict:
        return await _openai_compatible_call(
            client, self.URL, self.api_key, self.MODEL, prompt, provider_label="groq"
        )


# ---------------------------------------------------------------------
# 3) Cerebras
# ---------------------------------------------------------------------
class CerebrasProvider(BaseLLMProvider):
    name = "cerebras"
    URL = "https://api.cerebras.ai/v1/chat/completions"
    MODEL = "llama-3.3-70b"

    async def _call(self, client: httpx.AsyncClient, prompt: str) -> dict:
        return await _openai_compatible_call(
            client, self.URL, self.api_key, self.MODEL, prompt, provider_label="cerebras"
        )


# ---------------------------------------------------------------------
# 4) Mistral
# ---------------------------------------------------------------------
class MistralProvider(BaseLLMProvider):
    name = "mistral"
    URL = "https://api.mistral.ai/v1/chat/completions"
    MODEL = "mistral-large-latest"

    async def _call(self, client: httpx.AsyncClient, prompt: str) -> dict:
        return await _openai_compatible_call(
            client, self.URL, self.api_key, self.MODEL, prompt, provider_label="mistral"
        )


# ---------------------------------------------------------------------
# 5) OpenRouter (DeepSeek R1 free tier — strong numeric reasoning)
# ---------------------------------------------------------------------
class OpenRouterProvider(BaseLLMProvider):
    name = "openrouter"
    URL = "https://openrouter.ai/api/v1/chat/completions"
    MODEL = MODEL = "openrouter/free"

    async def _call(self, client: httpx.AsyncClient, prompt: str) -> dict:
        return await _openai_compatible_call(
            client, self.URL, self.api_key, self.MODEL, prompt, provider_label="openrouter",
            extra_headers={"HTTP-Referer": "https://alphamind.app", "X-Title": "AlphaMind"},
        )


# ---------------------------------------------------------------------
# Shared helper — Groq / Cerebras / Mistral / OpenRouter سب
# OpenAI-compatible /chat/completions شکل استعمال کرتے ہیں
# ---------------------------------------------------------------------
async def _openai_compatible_call(
    client: httpx.AsyncClient,
    url: str,
    api_key: str,
    model: str,
    prompt: str,
    provider_label: str,
    extra_headers: dict | None = None,
) -> dict:
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    if extra_headers:
        headers.update(extra_headers)

    try:
        resp = await client.post(
            url,
            headers=headers,
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_INSTRUCTION},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
                "response_format": {"type": "json_object"},
            },
            timeout=TIMEOUT_SECONDS,
        )
    except httpx.TimeoutException as exc:
        raise RetryableError(f"{provider_label}_timeout") from exc
    except httpx.TransportError as exc:
        raise RetryableError(f"{provider_label}_network_error") from exc

    if resp.status_code >= 500:
        raise RetryableError(f"{provider_label}_{resp.status_code}")
    if resp.status_code in (401, 403):
        raise ValueError(f"{provider_label}_invalid_key")
    if resp.status_code == 429:
        raise ValueError(f"{provider_label}_rate_limited")
    resp.raise_for_status()

    d = resp.json()
    text = d["choices"][0]["message"]["content"]
    return extract_json(text)