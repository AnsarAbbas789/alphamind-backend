"""
AlphaMind — indicators/mtf_confirmation.py
=====================================================================
Multi-Timeframe Confirmation — یہ win-rate بڑھانے کا سب سے بڑا ذریعہ ہے۔

اصول: چار timeframes (15m, 1H, 4H, 1D) پر آزادانہ طور پر bias نکالیں۔
اگر کم از کم 3 ایک ہی سمت پر متفق نہ ہوں تو سگنل بالکل نہ دیں (NEUTRAL) —
چاہے 1H اکیلا بہت مضبوط کیوں نہ لگے۔

⚠️ 2 ستمبر 2026 پرفارمنس فکس: نیچے timeframe_bias_series() شامل کیا
گیا ہے — یہ _timeframe_bias() جیسا ہی حساب پوری تاریخ کے لیے ایک ساتھ
(vectorized) کرتا ہے۔ صرف backtester.py استعمال کرتا ہے تاکہ ہزاروں
بار دوبارہ حساب نہ لگے۔ evaluate_mtf_confirmation() (لائیو انجن)
بالکل ویسا ہی رہا، کوئی تبدیلی نہیں۔
=====================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.indicators.technical import compute_all, ema, macd, to_dataframe


def _timeframe_bias(df) -> str:
    """ایک ہی timeframe کے لیے سادہ EMA + MACD histogram کی بنیاد پر bias۔
    صرف LIVE انجن استعمال کرتا ہے (ایک وقت میں ایک بار کال، رفتار مسئلہ نہیں)۔"""
    if len(df) < 55:
        return "INSUFFICIENT_DATA"

    stats = compute_all(df)
    score = 0

    if stats["ema_stack"] == "BULLISH":
        score += 1
    elif stats["ema_stack"] == "BEARISH":
        score -= 1

    if stats["macd_histogram"] is not None:
        score += 1 if stats["macd_histogram"] > 0 else -1

    if stats["price_vs_ema21_pct"] is not None:
        if stats["price_vs_ema21_pct"] > 0.1:
            score += 1
        elif stats["price_vs_ema21_pct"] < -0.1:
            score -= 1

    if score >= 2:
        return "BULLISH"
    if score <= -2:
        return "BEARISH"
    return "NEUTRAL"


def timeframe_bias_series(df: pd.DataFrame) -> pd.Series:
    """
    نیا (پرفارمنس فکس) — _timeframe_bias() کی vectorized شکل: پوری
    تاریخ کے ہر row کے لیے ایک ساتھ BULLISH/BEARISH/NEUTRAL نکالتی
    ہے (صرف ایک بار حساب)۔ منطق _timeframe_bias() جیسی بالکل ہے۔
    صرف backtester.py استعمال کرتا ہے۔
    """
    if len(df) < 55:
        return pd.Series(["INSUFFICIENT_DATA"] * len(df), index=df.index)

    closes = df["close"]
    ema9 = ema(closes, 9)
    ema21 = ema(closes, 21)
    ema50 = ema(closes, 50)
    macd_hist = macd(closes)["histogram"]
    price_vs_ema21_pct = ((closes - ema21) / ema21.replace(0, np.nan)) * 100

    stack_score = np.select(
        [(ema9 > ema21) & (ema21 > ema50), (ema9 < ema21) & (ema21 < ema50)],
        [1, -1], default=0,
    )
    macd_score = np.select([macd_hist > 0, macd_hist < 0], [1, -1], default=0)
    price_score = np.select(
        [price_vs_ema21_pct > 0.1, price_vs_ema21_pct < -0.1], [1, -1], default=0,
    )

    total_score = stack_score + macd_score + price_score
    bias = np.select([total_score >= 2, total_score <= -2], ["BULLISH", "BEARISH"], default="NEUTRAL")
    return pd.Series(bias, index=df.index)


def evaluate_mtf_confirmation(candles_by_timeframe: dict[str, list[dict]]) -> dict:
    """
    candles_by_timeframe: {"15m": [...], "1H": [...], "4H": [...], "1D": [...]}
    ہر timeframe کے لیے کم از کم 55 کینڈلز ہوں تو ہی وہ ووٹ میں شمار ہوگا۔
    صرف LIVE انجن (signal_composer.py) استعمال کرتا ہے۔

    return: {
        "aligned_direction": "BULLISH" | "BEARISH" | "MIXED",
        "agreement_count": int,       # کتنے timeframes متفق ہیں
        "total_evaluated": int,
        "per_timeframe": {...},
        "confirmed": bool             # کیا کم از کم 3/4 متفق ہیں
    }
    """
    per_tf = {}
    for tf, candles in candles_by_timeframe.items():
        if not candles or len(candles) < 55:
            per_tf[tf] = "INSUFFICIENT_DATA"
            continue
        df = to_dataframe(candles)
        per_tf[tf] = _timeframe_bias(df)

    evaluated = {tf: b for tf, b in per_tf.items() if b in ("BULLISH", "BEARISH", "NEUTRAL")}
    bullish_count = sum(1 for b in evaluated.values() if b == "BULLISH")
    bearish_count = sum(1 for b in evaluated.values() if b == "BEARISH")
    total = len(evaluated)

    if total == 0:
        aligned, confirmed, agreement = "MIXED", False, 0
    elif bullish_count >= max(3, total - 1) and bullish_count > bearish_count:
        aligned, confirmed, agreement = "BULLISH", total >= 3, bullish_count
    elif bearish_count >= max(3, total - 1) and bearish_count > bullish_count:
        aligned, confirmed, agreement = "BEARISH", total >= 3, bearish_count
    else:
        aligned, confirmed, agreement = "MIXED", False, max(bullish_count, bearish_count)

    return {
        "aligned_direction": aligned,
        "agreement_count": agreement,
        "total_evaluated": total,
        "per_timeframe": per_tf,
        "confirmed": confirmed,
    }