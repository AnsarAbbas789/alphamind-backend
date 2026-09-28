"""
AlphaMind — core/signal_composer.py
=====================================================================
یہ فائل پورے انجن کا "دماغ" ہے — تمام ماڈیولز کو یکجا کر کے حتمی
سگنل بناتی ہے۔ ترتیب (ہر قدم پچھلے کا محافظ ہے):

  1. Indicators compute (technical.py)
  2. Trend-strength فلٹر (ADX) — کمزور مارکیٹ میں سگنل مکمل بلاک
  3. Multi-timeframe confirmation — کم از کم 3/4 timeframes متفق ہوں
  4. Rule-engine bias score — سمت خالص ریاضی سے طے
  5. (نیا) تازہ خبریں لانا — صرف context، سمت پر کوئی اثر نہیں
  6. LLM ensemble (5 AIs) کی رائے — تصدیق/وضاحت کے لیے
  7. Entry/SL/TP کا ATR-based حساب
  8. Risk:Reward فلٹر — کم از کم 1:1.5 نہ ملے تو سگنل رد
  9. حتمی confidence = ریاضی + AI اتفاق + MTF اتفاق کا ملغوبہ

اگر کسی بھی قدم پر شرط پوری نہ ہو تو سگنل "NEUTRAL" اور واضح وجہ کے
ساتھ لوٹایا جاتا ہے — کبھی بھی کمزور/مشکوک سگنل زبردستی نہیں دیا جاتا۔
یہی اصول "غلطی کی گنجائش کم از کم" کا عملی نفاذ ہے۔
=====================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core import rule_engine
from app.core import news_fetcher
from app.core.prompt_builder import build_prompt
from app.indicators.mtf_confirmation import evaluate_mtf_confirmation
from app.indicators.technical import compute_all, to_dataframe
from app.llm.ensemble import run_ensemble, EnsembleResult
from app.llm.base import BaseLLMProvider

MIN_ADX = 20.0
MIN_RISK_REWARD = 1.5
MIN_CANDLES_REQUIRED = 60

# ATR کے کتنے گنے پر SL/TP رکھے جائیں — پیشہ ورانہ رِسک مینجمنٹ کا
# معیاری طریقہ (فکسڈ % کی بجائے مارکیٹ کی اپنی اتار چڑھاؤ پر مبنی)
ATR_SL_MULTIPLIER = 2.0
ATR_TP_MULTIPLIERS = [2.25, 3.5, 5.0, 7.0]  # TP1..TP4


@dataclass
class ComposedSignal:
    coin_analyzed: str = ""
    signal: str = "NEUTRAL"
    confidence: int = 0
    strength: str = "Weak"
    risk_level: str = "High"
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit_1: float = 0.0
    take_profit_2: float = 0.0
    take_profit_3: float = 0.0
    take_profit_4: float = 0.0
    leverage_suggestion: int | None = None
    risk_reward_ratio: float = 0.0
    indicator_breakdown: dict = field(default_factory=dict)
    reasoning_en: str = ""
    invalidation_note_en: str = ""
    news_summary_en: str = ""
    mtf_confirmation: dict = field(default_factory=dict)
    llm_votes: list = field(default_factory=list)
    llm_agreement_ratio: float = 0.0
    exchange_used: str = ""
    blocked_reason: str | None = None


def _risk_level_from_confidence(confidence: int, adx: float | None) -> str:
    if confidence >= 70 and (adx or 0) >= 30:
        return "Low"
    if confidence >= 50:
        return "Medium"
    return "High"


def _leverage_for(risk_level: str, market_type: str) -> int | None:
    if market_type != "futures":
        return None
    return {"Low": 15, "Medium": 8, "High": 3}.get(risk_level, 3)


async def compose_signal(
    symbol: str,
    market_type: str,
    candles_by_timeframe: dict[str, list[dict]],
    providers: list[BaseLLMProvider],
) -> ComposedSignal:
    result = ComposedSignal(coin_analyzed=symbol)

    primary_candles = candles_by_timeframe.get("1H") or []
    if len(primary_candles) < MIN_CANDLES_REQUIRED:
        result.blocked_reason = "insufficient_candle_data"
        result.reasoning_en = "Not enough historical candles to compute reliable indicators."
        return result

    df = to_dataframe(primary_candles)
    stats = compute_all(df)
    result.indicator_breakdown = stats

    # قدم 2 — ٹرینڈ کی طاقت فلٹر
    trend_ok, trend_msg = rule_engine.trend_strength_ok(stats, min_adx=MIN_ADX)
    if not trend_ok:
        result.blocked_reason = "weak_trend"
        result.reasoning_en = trend_msg
        result.risk_level = "High"
        return result

    # قدم 3 — Multi-timeframe confirmation
    mtf = evaluate_mtf_confirmation(candles_by_timeframe)
    result.mtf_confirmation = mtf
    if not mtf["confirmed"]:
        result.blocked_reason = "mtf_not_aligned"
        result.reasoning_en = (
            f"Timeframes disagree (only {mtf['agreement_count']}/{mtf['total_evaluated']} aligned) "
            "— no reliable directional edge right now."
        )
        return result

    # قدم 4 — Rule-engine bias score (خالص ریاضی سے سمت)
    bias = rule_engine.compute_bias(stats)
    if bias.direction == "NEUTRAL":
        result.blocked_reason = "bias_score_neutral"
        result.reasoning_en = f"Indicator bias score is {bias.net_score} — not decisive enough for a signal."
        return result

    if stats.get("volume_ratio") is None or stats["volume_ratio"] < 1.1:
        result.blocked_reason = "weak_volume"
        result.reasoning_en = "Volume too low relative to average — skipping low-conviction setup."
        return result

    # bias سمت اور MTF سمت میں تضاد نہیں ہونا چاہیے (اضافی حفاظتی چیک)
    bias_dir_simple = "BULLISH" if bias.direction == "LONG" else "BEARISH"
    if bias_dir_simple != mtf["aligned_direction"]:
        result.blocked_reason = "bias_mtf_conflict"
        result.reasoning_en = "Indicator bias and multi-timeframe trend disagree — skipping for safety."
        return result

    # قدم 5 — تازہ خبریں (صرف context؛ ناکام ہو تو خاموشی سے خالی)
    try:
        headlines = await news_fetcher.get_news_headlines(symbol)
    except Exception:  # noqa: BLE001 — خبریں کبھی سگنل انجن نہ روکیں
        headlines = []
    news_summary = news_fetcher.format_news_summary(headlines)
    result.news_summary_en = news_summary

    # قدم 6 — LLM ensemble (تصدیق/وضاحت)
    prompt = build_prompt(symbol, market_type, stats, bias, mtf, primary_candles, news_summary)
    ensemble: EnsembleResult = await run_ensemble(providers, prompt)
    result.llm_votes = ensemble.votes
    result.llm_agreement_ratio = ensemble.agreement_ratio

    if not ensemble.quorum_met:
        result.blocked_reason = "llm_quorum_not_met"
        result.reasoning_en = "Fewer than 2 AI providers responded successfully — refusing to guess."
        return result

    # AI ensemble کی سمت رول انجن کی سمت سے متصادم ہو تو سگنل نہیں بنے گا
    if ensemble.final_signal != bias.direction:
        result.blocked_reason = "llm_disagrees_with_math"
        result.reasoning_en = (
            f"Indicator math says {bias.direction} but AI ensemble majority said "
            f"{ensemble.final_signal} — conflict means low reliability, skipping."
        )
        return result

    # قدم 7 — Entry/SL/TP (ATR-based)
    entry = stats["price"]
    atr_val = stats["atr"] or (entry * 0.01)
    is_long = bias.direction == "LONG"

    stop_loss = entry - (atr_val * ATR_SL_MULTIPLIER) if is_long else entry + (atr_val * ATR_SL_MULTIPLIER)
    tps = [
        entry + (atr_val * m) if is_long else entry - (atr_val * m)
        for m in ATR_TP_MULTIPLIERS
    ]

    # قدم 8 — Risk:Reward فلٹر (TP1 پر جانچا جاتا ہے)
    rr_ok, rr_value = rule_engine.risk_reward_ok(entry, stop_loss, tps[0], min_rr=MIN_RISK_REWARD)
    if not rr_ok:
        result.blocked_reason = "poor_risk_reward"
        result.reasoning_en = f"Risk:Reward ratio only {rr_value}:1 (minimum {MIN_RISK_REWARD}:1 required) — skipping weak setup."
        return result

    # قدم 9 — حتمی confidence: ریاضی + AI اتفاق + MTF اتفاق کا ملغوبہ
    math_conf = bias.raw_confidence_pct
    mtf_conf = (mtf["agreement_count"] / max(mtf["total_evaluated"], 1)) * 100
    ai_conf = ensemble.avg_ai_confidence
    ai_agreement_conf = ensemble.agreement_ratio * 100

    final_confidence = round(
        (math_conf * 0.35) + (mtf_conf * 0.20) + (ai_conf * 0.20) + (ai_agreement_conf * 0.25)
    )
    final_confidence = max(0, min(100, final_confidence))

    risk_level = _risk_level_from_confidence(final_confidence, stats["adx"])
    strength = "Strong" if final_confidence >= 75 else "Moderate" if final_confidence >= 55 else "Weak"

    ai_reasoning_samples = [v.reasoning for v in ensemble.votes if v.ok and v.reasoning]
    combined_reasoning = " | ".join(ai_reasoning_samples[:2]) if ai_reasoning_samples else "; ".join(bias.reasons[:3])

    result.signal = bias.direction
    result.confidence = final_confidence
    result.strength = strength
    result.risk_level = risk_level
    result.entry_price = round(entry, 6)
    result.stop_loss = round(stop_loss, 6)
    result.take_profit_1 = round(tps[0], 6)
    result.take_profit_2 = round(tps[1], 6)
    result.take_profit_3 = round(tps[2], 6)
    result.take_profit_4 = round(tps[3], 6)
    result.leverage_suggestion = _leverage_for(risk_level, market_type)
    result.risk_reward_ratio = rr_value
    result.reasoning_en = (
        f"{bias.direction} bias confirmed across {mtf['agreement_count']}/{mtf['total_evaluated']} "
        f"timeframes and {len([v for v in ensemble.votes if v.ok and v.signal == bias.direction])}/"
        f"{ensemble.responded_count} AI models. {combined_reasoning}"
    )
    result.invalidation_note_en = (
        f"Signal invalidated if price closes beyond {result.stop_loss} "
        f"or if ADX drops below {MIN_ADX} (trend weakening)."
    )
    return result