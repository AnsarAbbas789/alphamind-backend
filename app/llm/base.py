"""
AlphaMind — llm/base.py
=====================================================================
تمام LLM providers اسی ایک base کلاس سے بنتے ہیں۔ فائدہ: کسی ایک
provider کی timeout/retry/error-handling الگ سے نہیں لکھنی پڑتی، اور
اگر ایک provider ناکام ہو تو باقی سب چلتے رہتے ہیں — "کبھی رسوا نہ
کرنے والا انجن" کا یہی مطلب ہے: ایک کی ناکامی پورے سسٹم کو نہیں گراتی۔
=====================================================================
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

TIMEOUT_SECONDS = 18.0


@dataclass
class LLMVote:
    provider: str
    signal: Optional[str] = None          # "LONG" | "SHORT" | "NEUTRAL"
    confidence: Optional[int] = None
    reasoning: Optional[str] = None
    latency_ms: Optional[int] = None
    error: Optional[str] = None
    raw: Optional[dict] = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.signal in ("LONG", "SHORT", "NEUTRAL")


class RetryableError(Exception):
    """صرف عارضی خرابیوں (network/5xx/timeout) پر دوبارہ کوشش کی جائے —
    غلط key (401/403) پر دوبارہ کوشش بیکار ہے، فوراً ناکام قرار دیں۔"""


def extract_json(text: str) -> dict:
    """AI کبھی کبھار JSON کے ارد گرد markdown fences یا اضافی الفاظ
    لکھ دیتا ہے — یہاں سختی سے صاف کر کے صرف JSON نکالا جاتا ہے۔"""
    cleaned = re.sub(r"```json|```", "", text).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError("no_json_found_in_response")
    return json.loads(match.group(0))


class BaseLLMProvider:
    name: str = "base"

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def _call(self, client: httpx.AsyncClient, prompt: str) -> dict:
        raise NotImplementedError

    @retry(
        reraise=True,
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception_type(RetryableError),
    )
    async def _call_with_retry(self, client: httpx.AsyncClient, prompt: str) -> dict:
        return await self._call(client, prompt)

    async def get_vote(self, client: httpx.AsyncClient, prompt: str) -> LLMVote:
        started = time.monotonic()
        try:
            data = await self._call_with_retry(client, prompt)
            latency = int((time.monotonic() - started) * 1000)

            signal = str(data.get("signal", "")).upper().strip()
            if signal not in ("LONG", "SHORT", "NEUTRAL"):
                return LLMVote(provider=self.name, error="invalid_signal_value", latency_ms=latency)

            confidence = data.get("confidence")
            try:
                confidence = max(0, min(100, int(confidence)))
            except (TypeError, ValueError):
                confidence = 50

            return LLMVote(
                provider=self.name,
                signal=signal,
                confidence=confidence,
                reasoning=str(data.get("reasoning", ""))[:600],
                latency_ms=latency,
                raw=data,
            )
        except Exception as exc:  # noqa: BLE001 — ہر ممکنہ ناکامی یہاں پکڑی جائے
            latency = int((time.monotonic() - started) * 1000)
            return LLMVote(provider=self.name, error=str(exc)[:200], latency_ms=latency)