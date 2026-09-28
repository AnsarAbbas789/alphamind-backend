"""
AlphaMind — core/candle_fetcher.py  (v2 — adds fixed date-range fetching)
=====================================================================
نیا اضافہ: fetch_candles_range() — یہ فنکشن ایک مقرر شروع اور اختتامی
تاریخ کے درمیان تمام candles لاتا ہے (pagination کے ذریعے، کیونکہ
Bybit/Binance ایک درخواست میں زیادہ سے زیادہ 1000 candles دیتے ہیں)۔
اس سے backtest ہمیشہ ایک جیسا، قابلِ اعتماد (reproducible) نتیجہ دیتا
ہے — چاہے آج چلائیں یا ایک ہفتے بعد، وہی مقرر تاریخوں کا نتیجہ آئے گا۔

Live سگنل کے لیے پرانا fetch_candles() (تازہ ترین N candles) بدستور
استعمال ہوگا — یہ نیا فنکشن صرف backtesting کے لیے ہے۔
=====================================================================
"""

from __future__ import annotations

from datetime import datetime, timezone

import httpx

TIMEOUT_SECONDS = 12.0

_INTERVAL_MAPS = {
    "bybit":   {"15m": "15", "1H": "60", "4H": "240", "1D": "D", "1W": "W"},
    "binance": {"15m": "15m", "1H": "1h", "4H": "4h", "1D": "1d", "1W": "1w"},
    "okx":     {"15m": "15m", "1H": "1H", "4H": "4H", "1D": "1Dutc", "1W": "1Wutc"},
    "bingx":   {"15m": "15m", "1H": "1h", "4H": "4h", "1D": "1d", "1W": "1w"},
    "kucoin":  {"15m": "15min", "1H": "1hour", "4H": "4hour", "1D": "1day", "1W": "1week"},
}

_INTERVAL_SECONDS = {"15m": 900, "1H": 3600, "4H": 14400, "1D": 86400, "1W": 604800}

EXCHANGE_PRIORITY = ["bybit", "binance", "okx", "bingx", "kucoin"]


async def _fetch_bybit(client: httpx.AsyncClient, symbol: str, interval: str, limit: int) -> list[dict]:
    resp = await client.get(
        "https://api.bybit.com/v5/market/kline",
        params={"category": "spot", "symbol": symbol, "interval": interval, "limit": limit},
        timeout=TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    rows = resp.json().get("result", {}).get("list", [])
    rows = list(reversed(rows))
    return [
        {"time": int(r[0]) // 1000, "open": float(r[1]), "high": float(r[2]),
         "low": float(r[3]), "close": float(r[4]), "volume": float(r[5])}
        for r in rows
    ]


async def _fetch_binance(client: httpx.AsyncClient, symbol: str, interval: str, limit: int) -> list[dict]:
    resp = await client.get(
        "https://api.binance.com/api/v3/klines",
        params={"symbol": symbol, "interval": interval, "limit": limit},
        timeout=TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    rows = resp.json()
    return [
        {"time": int(r[0]) // 1000, "open": float(r[1]), "high": float(r[2]),
         "low": float(r[3]), "close": float(r[4]), "volume": float(r[5])}
        for r in rows
    ]


async def _fetch_okx(client: httpx.AsyncClient, symbol: str, interval: str, limit: int) -> list[dict]:
    inst_id = symbol.replace("USDT", "-USDT")
    resp = await client.get(
        "https://www.okx.com/api/v5/market/candles",
        params={"instId": inst_id, "bar": interval, "limit": limit},
        timeout=TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    rows = list(reversed(resp.json().get("data", [])))
    return [
        {"time": int(r[0]) // 1000, "open": float(r[1]), "high": float(r[2]),
         "low": float(r[3]), "close": float(r[4]), "volume": float(r[5])}
        for r in rows
    ]


async def _fetch_bingx(client: httpx.AsyncClient, symbol: str, interval: str, limit: int) -> list[dict]:
    resp = await client.get(
        "https://open-api.bingx.com/openApi/spot/v2/market/kline",
        params={"symbol": symbol, "interval": interval, "limit": limit},
        timeout=TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    rows = resp.json().get("data", [])
    return [
        {"time": int(r["openTime"]) // 1000, "open": float(r["open"]), "high": float(r["high"]),
         "low": float(r["low"]), "close": float(r["close"]), "volume": float(r["volume"])}
        for r in rows
    ]


async def _fetch_kucoin(client: httpx.AsyncClient, symbol: str, interval: str, limit: int) -> list[dict]:
    import time as _time
    inst = symbol.replace("USDT", "-USDT")
    seconds_map = {"15min": 900, "1hour": 3600, "4hour": 14400, "1day": 86400, "1week": 604800}
    now = int(_time.time())
    start = now - limit * seconds_map.get(interval, 3600)
    resp = await client.get(
        "https://api.kucoin.com/api/v1/market/candles",
        params={"symbol": inst, "type": interval, "startAt": start, "endAt": now},
        timeout=TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    rows = list(reversed(resp.json().get("data", [])))
    return [
        {"time": int(r[0]), "open": float(r[1]), "close": float(r[2]),
         "high": float(r[3]), "low": float(r[4]), "volume": float(r[5])}
        for r in rows
    ]


_FETCHERS = {
    "bybit": _fetch_bybit,
    "binance": _fetch_binance,
    "okx": _fetch_okx,
    "bingx": _fetch_bingx,
    "kucoin": _fetch_kucoin,
}


async def fetch_candles(symbol: str, timeframe: str, limit: int = 300, preferred: str | None = None) -> tuple[list[dict], str]:
    """Live سگنل کے لیے — 'ابھی سے پیچھے N candles'۔"""
    order = list(EXCHANGE_PRIORITY)
    if preferred and preferred in order:
        order.remove(preferred)
        order.insert(0, preferred)

    last_error: Exception | None = None
    async with httpx.AsyncClient() as client:
        for exchange in order:
            interval = _INTERVAL_MAPS[exchange].get(timeframe)
            if not interval:
                continue
            try:
                candles = await _FETCHERS[exchange](client, symbol, interval, limit)
                if candles and len(candles) >= 20:
                    return candles, exchange
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue

    raise RuntimeError(f"all_exchanges_failed: {last_error}")


# ═══════════════════════════════════════════════════════════════════
# نیا — Fixed Date-Range Fetching (صرف backtesting کے لیے)
# ═══════════════════════════════════════════════════════════════════

def _date_to_ms(date_str: str) -> int:
    """'YYYY-MM-DD' کو UTC milliseconds timestamp میں بدلتا ہے۔"""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


async def _fetch_bybit_range(client: httpx.AsyncClient, symbol: str, interval: str,
                              start_ms: int, end_ms: int) -> list[dict]:
    """Bybit سے ایک مقرر تاریخی حد کے تمام candles pagination کے ذریعے لاتا ہے۔

    اہم دریافت: Bybit کی kline API ہمیشہ 'end' کے قریب ترین candles واپس دیتی
    ہے، 'start' سے آگے بڑھ کر نہیں — چاہے 'start' کچھ بھی ہو۔ اس لیے pagination
    کو 'end' سے 'start' کی طرف پیچھے کی سمت میں چلانا ضروری ہے (ہر صفحہ پرانی
    طرف بڑھے)، ورنہ صرف تازہ ترین 1000 candles ملتے ہیں، پورا دورانیہ نہیں۔"""
    all_candles: list[dict] = []
    seen_times: set[int] = set()
    cursor_end = end_ms

    for _ in range(500):  # 2+ سال کے 1H ڈیٹا کے لیے کافی صفحات (تقریباً 18 چاہئیں)
        if cursor_end <= start_ms:
            break
        resp = await client.get(
            "https://api.bybit.com/v5/market/kline",
            params={"category": "spot", "symbol": symbol, "interval": interval,
                     "start": start_ms, "end": cursor_end, "limit": 1000},
            timeout=TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        rows = resp.json().get("result", {}).get("list", [])
        if not rows:
            break

        rows = sorted(rows, key=lambda r: int(r[0]))  # ہمیشہ خود ascending ترتیب دیں
        batch_min_time = int(rows[0][0])

        new_rows_added = 0
        for r in rows:
            t = int(r[0]) // 1000
            if t in seen_times:
                continue
            seen_times.add(t)
            all_candles.append({
                "time": t, "open": float(r[1]), "high": float(r[2]),
                "low": float(r[3]), "close": float(r[4]), "volume": float(r[5]),
            })
            new_rows_added += 1

        if new_rows_added == 0:
            break

        next_cursor_end = batch_min_time - 1  # اگلا صفحہ اس سے پرانی طرف مانگیں
        if next_cursor_end >= cursor_end:  # آگے نہ بڑھے تو لامحدود loop سے بچیں
            break
        cursor_end = next_cursor_end

    all_candles.sort(key=lambda c: c["time"])
    return all_candles


async def _fetch_binance_range(client: httpx.AsyncClient, symbol: str, interval: str,
                                start_ms: int, end_ms: int) -> list[dict]:
    """Binance سے ایک مقرر تاریخی حد کے candles pagination کے ذریعے لاتا ہے (Bybit fallback)۔"""
    all_candles: list[dict] = []
    cursor = start_ms

    for _ in range(200):
        if cursor >= end_ms:
            break
        resp = await client.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": symbol, "interval": interval,
                     "startTime": cursor, "endTime": end_ms, "limit": 1000},
            timeout=TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            break

        rows = sorted(rows, key=lambda r: int(r[0]))  # اسی حفاظتی اصول کے مطابق

        for r in rows:
            all_candles.append({
                "time": int(r[0]) // 1000, "open": float(r[1]), "high": float(r[2]),
                "low": float(r[3]), "close": float(r[4]), "volume": float(r[5]),
            })

        last_time_ms = int(rows[-1][0])
        new_cursor = last_time_ms + 1
        if new_cursor <= cursor:
            break
        cursor = new_cursor

    all_candles.sort(key=lambda c: c["time"])
    return all_candles


_RANGE_FETCHERS = {
    "bybit": _fetch_bybit_range,
    "binance": _fetch_binance_range,
}


async def fetch_candles_range(
    symbol: str, timeframe: str, start_date: str, end_date: str, preferred: str | None = None,
) -> tuple[list[dict], str]:
    """
    Backtesting کے لیے — ایک مقرر تاریخی حد ('YYYY-MM-DD' سے 'YYYY-MM-DD') کے
    تمام candles لاتا ہے۔ نتیجہ ہمیشہ ایک جیسا رہتا ہے (reproducible) کیونکہ
    تاریخیں مقرر ہیں، 'ابھی سے پیچھے N candles' نہیں۔
    """
    start_ms = _date_to_ms(start_date)
    end_ms = _date_to_ms(end_date)
    interval_key = timeframe
    min_expected = 250  # اس سے کم آئے تو backtest بے معنی ہوگا — خاموشی سے نظرانداز نہ ہو

    order = ["bybit", "binance"]
    if preferred and preferred in order:
        order.remove(preferred)
        order.insert(0, preferred)

    last_error: Exception | None = None
    async with httpx.AsyncClient() as client:
        for exchange in order:
            interval = _INTERVAL_MAPS[exchange].get(interval_key)
            if not interval:
                continue
            try:
                candles = await _RANGE_FETCHERS[exchange](client, symbol, interval, start_ms, end_ms)
                if candles and len(candles) >= min_expected:
                    return candles, exchange
                last_error = RuntimeError(
                    f"only {len(candles)} candles returned (need >= {min_expected}) from {exchange}"
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue

    raise RuntimeError(f"all_exchanges_failed_range: {last_error}")