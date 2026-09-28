"""
AlphaMind — core/news_fetcher.py
=====================================================================
CryptoCompare کی مفت News API (کوئی key درکار نہیں) سے ہر coin کے لیے
تازہ خبریں لاتا ہے۔ یہ خبریں AI prompt میں صرف EXTRA CONTEXT کے طور پر
شامل ہوں گی — سمت کا فیصلہ ہمیشہ rule_engine.py کے bias score سے ہوتا
ہے، خبروں سے کبھی نہیں۔ اگر یہ فنکشن ناکام ہو تو خالی نتیجہ دے، کبھی
exception نہ اٹھائے — سگنل انجن کو کبھی نہیں روکنا۔

Cache: 5 منٹ فی symbol (scanner کے SCAN_INTERVAL_SECONDS سے ملتا ہوا) —
اس سے ہر /api/signal کال پر بلاوجہ بار بار news API کال نہیں ہوتی۔
=====================================================================
"""

from __future__ import annotations

import time

import httpx

TIMEOUT_SECONDS = 6.0  # جان بوجھ کر مختصر — خبریں صرف اضافی معلومات ہیں،
                        # سگنل کی رفتار کو ان کی وجہ سے سست نہیں ہونا چاہیے
CACHE_TTL_SECONDS = 300  # 5 منٹ
MAX_HEADLINES = 5

# {symbol: (fetched_at_epoch, [headline, ...])}
_cache: dict[str, tuple[float, list[str]]] = {}

# BTCUSDT -> "bitcoin", ETHUSDT -> "ethereum" وغیرہ — صرف مطابقت کے
# لیے، تاکہ عمومی crypto خبروں میں سے متعلقہ coin چنا جا سکے
_COIN_KEYWORDS = {
    "BTCUSDT": ["bitcoin", "btc"],
    "ETHUSDT": ["ethereum", "eth"],
    "BNBUSDT": ["bnb", "binance coin"],
    "SOLUSDT": ["solana", "sol"],
    "XRPUSDT": ["xrp", "ripple"],
    "ADAUSDT": ["cardano", "ada"],
    "DOGEUSDT": ["dogecoin", "doge"],
}


async def _fetch_raw_news() -> list[dict]:
    """CryptoCompare سے عمومی crypto خبریں لاتا ہے (سب coins کے لیے مشترک)۔"""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://min-api.cryptocompare.com/data/v2/news/?lang=EN",
            timeout=TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("Data", []) or []


def _filter_for_symbol(articles: list[dict], symbol: str) -> list[str]:
    """صرف اس symbol سے متعلقہ خبروں کے عنوانات چنتا ہے (title میں
    coin کا نام یا مختصر شکل ملے تو)۔ کچھ نہ ملے تو عمومی top خبریں
    (پہلی 3) fallback کے طور پر دیتا ہے — بالکل خالی سے بہتر ہے۔"""
    keywords = _COIN_KEYWORDS.get(symbol.upper(), [])
    matched = []

    for article in articles:
        title = (article.get("title") or "").lower()
        if any(kw in title for kw in keywords):
            matched.append(article.get("title", "").strip())
        if len(matched) >= MAX_HEADLINES:
            break

    if not matched:
        # کوائن-مخصوص خبر نہ ملے تو عمومی مارکیٹ خبریں دیں (پھر بھی مفید سیاق)
        matched = [a.get("title", "").strip() for a in articles[:3] if a.get("title")]

    return [h for h in matched if h][:MAX_HEADLINES]


async def get_news_headlines(symbol: str) -> list[str]:
    """
    ایک symbol کے لیے تازہ ترین متعلقہ خبروں کے عنوانات (زیادہ سے زیادہ 5)۔
    ناکامی کی صورت میں خالی فہرست — کبھی exception نہیں اٹھاتا۔
    """
    symbol = symbol.upper()
    now = time.time()

    cached = _cache.get(symbol)
    if cached and (now - cached[0]) < CACHE_TTL_SECONDS:
        return cached[1]

    try:
        raw = await _fetch_raw_news()
        headlines = _filter_for_symbol(raw, symbol)
    except Exception:  # noqa: BLE001 — خبریں اختیاری ہیں، ناکامی کبھی سگنل نہ روکے
        headlines = cached[1] if cached else []  # پرانا cache ہو تو وہی دے دیں، ورنہ خالی

    _cache[symbol] = (now, headlines)
    return headlines


def format_news_summary(headlines: list[str]) -> str:
    """AI prompt میں شامل کرنے کے لیے ایک مختصر متن بناتا ہے۔"""
    if not headlines:
        return "No recent news available."
    return " | ".join(headlines)