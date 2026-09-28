"""
AlphaMind — routes/signal.py
=====================================================================
یہی وہ endpoint ہے جو آپ کے موجودہ ai-signal-fixed.js میں
CONFIG.AI_SIGNAL_ENDPOINT پر کال ہوتا ہے۔

⚠️ اپڈیٹس:
  1. Coin فہرست watchlist.py سے آتی ہے (DEFAULT_BACKTEST_SYMBOLS اب
     VALIDATED_WATCHLIST_SYMBOLS کا دوسرا نام ہے)۔
  2. Backtest endpoints میں asyncio.wait_for(timeout=90) شامل ہے —
     احتیاطی حفاظت، تاکہ سرور کبھی ہمیشہ کے لیے نہ لٹکے۔
  3. (نیا) response میں "news_summary_en" field شامل کیا گیا ہے —
     signal_composer.py اب news_fetcher.py سے تازہ خبریں لا کر
     result.news_summary_en میں رکھتا ہے؛ یہاں وہ frontend کو بھیجی
     جاتی ہے (blocked اور success دونوں حالتوں میں)۔
=====================================================================
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.config import get_settings
from app.core.auth import AuthenticatedUser, verify_supabase_token
from app.core.backtester import run_backtest
from app.core.candle_fetcher import fetch_candles, fetch_candles_range
from app.core.signal_composer import compose_signal
from app.core.watchlist import VALIDATED_WATCHLIST_SYMBOLS as DEFAULT_BACKTEST_SYMBOLS
from app.llm.registry import build_active_providers

router = APIRouter()

BACKTEST_TIMEOUT_SECONDS = 90


class SignalRequest(BaseModel):
    symbol: str
    marketType: str = "spot"
    exchange: str | None = None  # dashboard پر منتخب کردہ ایکسچینج (اگر ہو)
    prompt: str | None = None    # پرانی JS بھیجتی تھی — اب نظرانداز ہوتا ہے


@router.post("/api/signal")
async def generate_signal(body: SignalRequest, user: AuthenticatedUser = Depends(verify_supabase_token)):
    settings = get_settings()
    symbol = body.symbol.upper().strip()
    market_type = body.marketType.lower().strip()

    if market_type not in ("spot", "futures"):
        raise HTTPException(400, detail="invalid_market_type")

    try:
        candles_15m, ex15 = await fetch_candles(symbol, "15m", limit=200, preferred=body.exchange)
        candles_1h, ex1h = await fetch_candles(symbol, "1H", limit=300, preferred=body.exchange)
        candles_4h, ex4h = await fetch_candles(symbol, "4H", limit=200, preferred=body.exchange)
        candles_1d, ex1d = await fetch_candles(symbol, "1D", limit=250, preferred=body.exchange)
    except RuntimeError as exc:
        raise HTTPException(502, detail=f"candle_fetch_failed: {exc}")

    providers = build_active_providers(settings)
    if len(providers) < 2:
        raise HTTPException(
            503,
            detail="not_enough_llm_providers_configured — need at least 2 API keys in .env",
        )

    result = await compose_signal(
        symbol=symbol,
        market_type=market_type,
        candles_by_timeframe={"15m": candles_15m, "1H": candles_1h, "4H": candles_4h, "1D": candles_1d},
        providers=providers,
    )
    result.exchange_used = ex1h

    if result.blocked_reason:
        return {
            "coin_analyzed": symbol,
            "signal": "NEUTRAL",
            "confidence": 0,
            "strength": "Weak",
            "risk_level": "High",
            "entry_price": 0,
            "stop_loss": 0,
            "take_profit_1": 0,
            "take_profit_2": 0,
            "take_profit_3": 0,
            "take_profit_4": 0,
            "reasoning_en": result.reasoning_en,
            "reasoning_ur": "",
            "invalidation_note_en": "",
            "invalidation_note_ur": "",
            "news_summary_en": result.news_summary_en,
            "indicator_breakdown": {
                "rsi_analysis_en": f"RSI: {result.indicator_breakdown.get('rsi')}",
                "macd_analysis_en": f"MACD histogram: {result.indicator_breakdown.get('macd_histogram')}",
                "ema_analysis_en": f"EMA stack: {result.indicator_breakdown.get('ema_stack')}",
            },
            "blocked_reason": result.blocked_reason,
            "exchange_used": result.exchange_used,
        }

    response = {
        "coin_analyzed": result.coin_analyzed,
        "signal": result.signal,
        "confidence": result.confidence,
        "strength": result.strength,
        "risk_level": result.risk_level,
        "entry_price": result.entry_price,
        "stop_loss": result.stop_loss,
        "take_profit_1": result.take_profit_1,
        "take_profit_2": result.take_profit_2,
        "take_profit_3": result.take_profit_3,
        "take_profit_4": result.take_profit_4,
        "reasoning_en": result.reasoning_en,
        "reasoning_ur": "",
        "invalidation_note_en": result.invalidation_note_en,
        "invalidation_note_ur": "",
        "news_summary_en": result.news_summary_en,
        "indicator_breakdown": {
            "rsi_analysis_en": f"RSI(14)={result.indicator_breakdown.get('rsi')}",
            "macd_analysis_en": f"MACD histogram={result.indicator_breakdown.get('macd_histogram')}",
            "ema_analysis_en": f"EMA stack={result.indicator_breakdown.get('ema_stack')}",
        },
        "risk_reward_ratio": result.risk_reward_ratio,
        "mtf_confirmation": result.mtf_confirmation,
        "llm_agreement_ratio": result.llm_agreement_ratio,
        "llm_votes_summary": [
            {"provider": v.provider, "signal": v.signal, "confidence": v.confidence, "ok": v.ok}
            for v in result.llm_votes
        ],
        "exchange_used": result.exchange_used,
    }

    if market_type == "futures" and result.leverage_suggestion:
        response["leverage_suggestion"] = result.leverage_suggestion

    return response


# مقرر تاریخیں — ہمیشہ یہی دورانیہ ٹیسٹ ہوگا، نتیجہ کبھی نہیں بدلے گا
DEFAULT_START_DATE = "2023-06-01"
DEFAULT_END_DATE = "2025-06-01"


class BacktestRequest(BaseModel):
    symbol: str
    start_date: str = DEFAULT_START_DATE
    end_date: str = DEFAULT_END_DATE


async def _fetch_range_set(symbol: str, start_date: str, end_date: str):
    """ایک coin کے تینوں timeframes کے candles ایک ساتھ لائیں (timeout کے اندر)۔"""
    candles_1h, _ = await fetch_candles_range(symbol, "1H", start_date, end_date)
    candles_4h, _ = await fetch_candles_range(symbol, "4H", start_date, end_date)
    candles_1d, _ = await fetch_candles_range(symbol, "1D", start_date, end_date)
    return candles_1h, candles_4h, candles_1d


@router.post("/api/backtest")
async def backtest(body: BacktestRequest, user: AuthenticatedUser = Depends(verify_supabase_token)):
    """Win-rate کی تصدیق — مقرر تاریخوں پر۔ asyncio.wait_for سے محفوظ —
    انٹرنیٹ سست ہو تب بھی سرور {BACKTEST_TIMEOUT_SECONDS} سیکنڈ بعد صاف
    ٹائم آؤٹ ایرر دے گا، ہمیشہ کے لیے نہیں لٹکے گا۔"""
    symbol = body.symbol.upper().strip()
    try:
        candles_1h, candles_4h, candles_1d = await asyncio.wait_for(
            _fetch_range_set(symbol, body.start_date, body.end_date),
            timeout=BACKTEST_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        raise HTTPException(504, detail=f"backtest_timeout_after_{BACKTEST_TIMEOUT_SECONDS}s")
    except RuntimeError as exc:
        raise HTTPException(502, detail=f"candle_fetch_failed: {exc}")

    report = run_backtest(symbol, candles_1h, candles_4h, candles_1d)
    return {
        "symbol": report.symbol,
        "period": f"{body.start_date} to {body.end_date}",
        "total_signals": report.total_signals,
        "wins": report.wins,
        "losses": report.losses,
        "open_at_end": report.open_at_end,
        "win_rate_pct": report.win_rate_pct,
        "avg_bars_held": report.avg_bars_held,
        "expectancy_r": report.expectancy_r,
        "total_r_multiple": report.total_r_multiple,
    }


class BatchBacktestRequest(BaseModel):
    symbols: list[str] | None = None
    start_date: str = DEFAULT_START_DATE
    end_date: str = DEFAULT_END_DATE


@router.post("/api/backtest/batch")
async def backtest_batch(body: BatchBacktestRequest, user: AuthenticatedUser = Depends(verify_supabase_token)):
    """ایک ہی کال میں کئی coins — ہر coin کو الگ سے timeout حفاظت ملتی
    ہے تاکہ ایک coin کا سست ہونا باقیوں کو نہ روکے۔ اگر symbols نہ دیں
    تو watchlist.py کی تصدیق شدہ فہرست خودکار استعمال ہوگی۔"""
    symbols = body.symbols or DEFAULT_BACKTEST_SYMBOLS
    per_symbol_results = []

    for symbol in symbols:
        symbol = symbol.upper().strip()
        try:
            candles_1h, candles_4h, candles_1d = await asyncio.wait_for(
                _fetch_range_set(symbol, body.start_date, body.end_date),
                timeout=BACKTEST_TIMEOUT_SECONDS,
            )
            report = run_backtest(symbol, candles_1h, candles_4h, candles_1d)
            per_symbol_results.append({
                "symbol": symbol,
                "total_signals": report.total_signals,
                "wins": report.wins,
                "losses": report.losses,
                "win_rate_pct": report.win_rate_pct,
                "avg_bars_held": report.avg_bars_held,
                "expectancy_r": report.expectancy_r,
                "total_r_multiple": report.total_r_multiple,
                "error": None,
            })
        except asyncio.TimeoutError:
            per_symbol_results.append({
                "symbol": symbol, "total_signals": 0, "wins": 0, "losses": 0,
                "win_rate_pct": 0.0, "avg_bars_held": 0.0,
                "expectancy_r": 0.0, "total_r_multiple": 0.0,
                "error": f"timeout_after_{BACKTEST_TIMEOUT_SECONDS}s",
            })
        except Exception as exc:
            per_symbol_results.append({
                "symbol": symbol, "total_signals": 0, "wins": 0, "losses": 0,
                "win_rate_pct": 0.0, "avg_bars_held": 0.0,
                "expectancy_r": 0.0, "total_r_multiple": 0.0,
                "error": str(exc)[:150],
            })

    total_signals = sum(r["total_signals"] for r in per_symbol_results)
    total_wins = sum(r["wins"] for r in per_symbol_results)
    total_losses = sum(r["losses"] for r in per_symbol_results)
    total_r = round(sum(r["total_r_multiple"] for r in per_symbol_results), 3)
    closed = total_wins + total_losses

    return {
        "period": f"{body.start_date} to {body.end_date}",
        "per_symbol": per_symbol_results,
        "aggregate": {
            "symbols_tested": len(symbols),
            "total_signals": total_signals,
            "total_wins": total_wins,
            "total_losses": total_losses,
            "overall_win_rate_pct": round((total_wins / closed) * 100, 2) if closed else 0.0,
            "total_r_multiple": total_r,
            "expectancy_r": round(total_r / closed, 3) if closed else 0.0,
        },
    }


class DebugCandlesRequest(BaseModel):
    symbol: str
    timeframe: str = "1H"
    start_date: str = DEFAULT_START_DATE
    end_date: str = DEFAULT_END_DATE


@router.post("/api/debug/candles")
async def debug_candles(body: DebugCandlesRequest, user: AuthenticatedUser = Depends(verify_supabase_token)):
    """تشخیصی endpoint — کتنے candles آ رہے ہیں، کوئی gap تو نہیں۔"""
    symbol = body.symbol.upper().strip()
    try:
        candles, exchange = await fetch_candles_range(symbol, body.timeframe, body.start_date, body.end_date)
    except RuntimeError as exc:
        return {"error": str(exc)}

    if not candles:
        return {"error": "no_candles_returned", "count": 0}

    times = [c["time"] for c in candles]
    gaps = []
    expected_gap = {"15m": 900, "1H": 3600, "4H": 14400, "1D": 86400}.get(body.timeframe, 3600)
    for i in range(1, len(times)):
        diff = times[i] - times[i - 1]
        if diff > expected_gap * 1.5:
            gaps.append({"after_index": i - 1, "gap_hours": round(diff / 3600, 1)})

    return {
        "symbol": symbol,
        "exchange_used": exchange,
        "total_candles": len(candles),
        "first_candle_time": times[0],
        "last_candle_time": times[-1],
        "duplicate_timestamps": len(times) - len(set(times)),
        "gaps_found": len(gaps),
        "gap_details_sample": gaps[:5],
    }