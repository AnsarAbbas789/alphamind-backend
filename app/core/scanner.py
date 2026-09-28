"""
AlphaMind — core/scanner.py
=====================================================================
Live Scanner کا "دماغ"۔ یہ backend میں ہمیشہ پس منظر میں چلتا رہتا
ہے (ہر SCAN_INTERVAL_SECONDS بعد) اور watchlist کی ہر coin کو 4
حالتوں میں سے ایک میں رکھتا ہے:

  WATCHING  — کچھ خاص نہیں، عام حالت
  BUILDING  — کچھ شرائط پوری ہو رہی ہیں، رجحان بن رہا ہے
  HOT       — تقریباً تمام ریاضی شرائط پوری، AI تصدیق باقی
  SIGNAL    — مکمل تصدیق شدہ trade تیار (AI ensemble نے بھی ہاں کہا)

اہم اصول (لاگت بچانے کے لیے): WATCHING/BUILDING/HOT کی تشخیص صرف
ریاضی (indicators) سے ہوتی ہے — کوئی LLM کال نہیں۔ صرف جب کوئی coin
HOT حالت میں پہنچے تو ایک بار compose_signal() (جو LLM ensemble
استعمال کرتا ہے) کال ہوتا ہے۔

⚠️ 6 ستمبر 2026 اضافہ — Daily LLM Cost Cap: اگر کبھی مارکیٹ میں
اچانک بہت ہلچل ہو اور کئی coins بیک وقت HOT بن جائیں، تو بغیر حد کے
LLM بل اچانک بڑھ سکتا ہے۔ اب ایک روزانہ حد (MAX_LLM_CALLS_PER_DAY)
مقرر ہے — حد پوری ہونے پر باقی دن کے لیے نئی HOT coins صرف ریاضی
کی بنیاد پر "HOT (AI check skipped — daily cap reached)" دکھائی
جائیں گی، AI کال نہیں ہوگی۔ اگلے دن (UTC کیلنڈر) شمار خودکار صفر
ہو جاتا ہے۔
=====================================================================
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone

from app.config import get_settings
from app.core import rule_engine
from app.core.candle_fetcher import fetch_candles
from app.core.signal_composer import compose_signal
from app.core.watchlist import VALIDATED_WATCHLIST_SYMBOLS
from app.indicators.mtf_confirmation import evaluate_mtf_confirmation
from app.indicators.technical import compute_all, to_dataframe
from app.llm.registry import build_active_providers

SCAN_INTERVAL_SECONDS = 300  # 5 منٹ — تجویز کردہ متوازن قدر
MIN_ADX = 20.0  # signal_composer.py کے MIN_ADX سے ملتی ہونی چاہیے

# ── Daily LLM cost cap ──────────────────────────────────────────
MAX_LLM_CALLS_PER_DAY = 30  # آپ اپنی مرضی سے یہ نمبر بدل سکتے ہیں

# ── نتیجہ یہاں محفوظ رہتا ہے (in-memory) ─────────────────────────
_scan_results: dict[str, dict] = {}
_llm_call_count_today = 0
_llm_cap_date: str | None = None  # "YYYY-MM-DD" (UTC) — کس دن کا شمار ہے


def _reset_daily_cap_if_new_day() -> None:
    global _llm_call_count_today, _llm_cap_date
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if _llm_cap_date != today:
        _llm_cap_date = today
        _llm_call_count_today = 0


async def scan_symbol(symbol: str, exchange: str | None = None) -> dict:
    """صرف ریاضی سے ایک coin کی حالت معلوم کرتا ہے — کوئی LLM کال نہیں۔"""
    try:
        candles_15m, _ = await fetch_candles(symbol, "15m", limit=200, preferred=exchange)
        candles_1h, _ = await fetch_candles(symbol, "1H", limit=300, preferred=exchange)
        candles_4h, _ = await fetch_candles(symbol, "4H", limit=200, preferred=exchange)
        candles_1d, _ = await fetch_candles(symbol, "1D", limit=250, preferred=exchange)
    except RuntimeError as exc:
        return {"symbol": symbol, "state": "ERROR", "error": str(exc)[:150], "updated_at": time.time()}

    df = to_dataframe(candles_1h)
    stats = compute_all(df)

    trend_ok, _ = rule_engine.trend_strength_ok(stats, min_adx=MIN_ADX)
    bias = rule_engine.compute_bias(stats)

    mtf = evaluate_mtf_confirmation(
        {"15m": candles_15m, "1H": candles_1h, "4H": candles_4h, "1D": candles_1d}
    )

    # bias.direction، entry-timing veto لگنے پر NEUTRAL دے دیتا ہے —
    # سکینر کے لیے ہمیں "خالص ریاضی کا رجحان" چاہیے، veto سے پہلے
    if bias.net_score >= 4:
        raw_direction = "LONG"
    elif bias.net_score <= -4:
        raw_direction = "SHORT"
    else:
        raw_direction = None

    mtf_aligned = False
    if raw_direction:
        bias_dir_simple = "BULLISH" if raw_direction == "LONG" else "BEARISH"
        mtf_aligned = mtf["confirmed"] and bias_dir_simple == mtf["aligned_direction"]

    checklist = {
        "trend_strength": trend_ok,
        "bias_score": raw_direction is not None,
        "mtf_aligned": mtf_aligned,
        "volume_confirmed": (stats.get("volume_ratio") or 0) >= 1.1,
        "entry_timing": bool(raw_direction) and bias.entry_timing_ok,
    }
    passed = sum(1 for v in checklist.values() if v)
    total = len(checklist)

    if not trend_ok or raw_direction is None:
        state = "WATCHING"
    elif passed >= total:  # سب شرائط پوری — AI تصدیق کے لیے تیار
        state = "HOT"
    elif passed >= 3:
        state = "BUILDING"
    else:
        state = "WATCHING"

    return {
        "symbol": symbol,
        "state": state,
        "direction_lean": raw_direction,
        "checklist": checklist,
        "checklist_passed": passed,
        "checklist_total": total,
        "price": stats["price"],
        "adx": stats["adx"],
        "rsi": stats["rsi"],
        "entry_timing_note": bias.entry_timing_note,
        "updated_at": time.time(),
        "_candles": {"15m": candles_15m, "1H": candles_1h, "4H": candles_4h, "1D": candles_1d},
    }


async def run_scan_cycle() -> None:
    """ایک مکمل چکر — پوری watchlist چیک کرے، HOT کوائنز پر AI تصدیق کرے
    (روزانہ حد کے اندر رہتے ہوئے)۔"""
    global _llm_call_count_today

    _reset_daily_cap_if_new_day()

    settings = get_settings()
    providers = build_active_providers(settings)

    print(f"\n[SCANNER] Starting scan cycle — {len(VALIDATED_WATCHLIST_SYMBOLS)} symbols "
          f"(LLM calls today: {_llm_call_count_today}/{MAX_LLM_CALLS_PER_DAY})...")

    for symbol in VALIDATED_WATCHLIST_SYMBOLS:
        result = await scan_symbol(symbol)
        candles = result.pop("_candles", None)

        cap_reached = _llm_call_count_today >= MAX_LLM_CALLS_PER_DAY

        if result["state"] == "HOT" and candles and len(providers) >= 2:
            if cap_reached:
                result["ai_check_skipped"] = "daily_llm_cap_reached"
            else:
                # صرف HOT حالت میں — مہنگا LLM ensemble صرف یہیں کال ہوتا ہے
                try:
                    _llm_call_count_today += 1
                    composed = await compose_signal(
                        symbol=symbol,
                        market_type="spot",
                        candles_by_timeframe=candles,
                        providers=providers,
                    )
                    if not composed.blocked_reason:
                        result["state"] = "SIGNAL"
                        result["full_signal"] = {
                            "signal": composed.signal,
                            "confidence": composed.confidence,
                            "entry_price": composed.entry_price,
                            "stop_loss": composed.stop_loss,
                            "take_profit_1": composed.take_profit_1,
                            "reasoning_en": composed.reasoning_en,
                        }
                    else:
                        result["blocked_reason"] = composed.blocked_reason
                except Exception as exc:  # ایک coin کی ناکامی پورے scanner کو نہ روکے
                    result["ai_check_error"] = str(exc)[:150]

        _scan_results[symbol] = result

        # ── صاف، پڑھنے کے قابل status line ──────────────────────
        state = result["state"]
        lean = result.get("direction_lean") or "-"
        passed = result.get("checklist_passed", 0)
        total = result.get("checklist_total", 0)
        extra = ""
        if result.get("ai_check_skipped"):
            extra = "  (AI check skipped — daily cap reached)"
        print(f"  [{symbol:<10}] state={state:<9} lean={lean:<5} checklist={passed}/{total}{extra}")

    print("[SCANNER] Scan cycle complete.\n")


async def scanner_background_loop() -> None:
    """یہ فنکشن سرور شروع ہوتے ہی ہمیشہ کے لیے چلتا رہتا ہے۔ کوئی بھی
    غیرمتوقع ایرر پورے scanner کو نہیں روکتا۔"""
    while True:
        try:
            await run_scan_cycle()
        except Exception as exc:
            print(f"[SCANNER] Cycle failed: {exc}")  # اگلے چکر میں دوبارہ کوشش ہوگی
        await asyncio.sleep(SCAN_INTERVAL_SECONDS)


def get_scan_results() -> dict:
    """موجودہ نتائج کی کاپی — frontend اسی کو پڑھے گا۔"""
    return {k: v for k, v in _scan_results.items()}


def get_llm_cap_status() -> dict:
    """Frontend یا debugging کے لیے — آج کتنی LLM calls ہو چکیں۔"""
    _reset_daily_cap_if_new_day()
    return {
        "calls_today": _llm_call_count_today,
        "daily_cap": MAX_LLM_CALLS_PER_DAY,
        "date": _llm_cap_date,
    }