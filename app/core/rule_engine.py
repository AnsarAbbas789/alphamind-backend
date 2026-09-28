"""
AlphaMind — core/rule_engine.py  (v2 — Trend + Pullback Entry Design)
=====================================================================
یہ فائل دوبارہ ڈیزائن ہوئی ہے کیونکہ backtest نے ثابت کیا کہ پرانا
طریقہ (سب indicators کو ایک ہی سمت میں پوائنٹس دینا) اکثر ٹرینڈ کے
تھکنے کے قریب سگنل بناتا تھا — یعنی دیر سے entry، جو نقصان کا باعث
بن رہا تھا۔

نیا اصول — "Trend + Pullback Entry" (پروفیشنل traders کا معیاری طریقہ):

  گروپ 1 — TREND CONFIRMATION (سمت کا فیصلہ):
    EMA Stack, MACD Histogram, Price-vs-EMA, OBV Trend, Volume-confirmed move
    یہ سب "ٹرینڈ چل رہا ہے اور تسلسل میں ہے" کی تصدیق کرتے ہیں۔

  گروپ 2 — ENTRY TIMING FILTER (کب داخل ہوں):
    RSI, Stochastic, CCI — یہ اب "زیادہ سے زیادہ پوائنٹس" کے لیے نہیں
    بلکہ یہ چیک کرنے کے لیے ہیں کہ آیا قیمت ابھی extreme/تھکی ہوئی حالت
    میں تو نہیں (جس صورت میں entry روک دی جائے، چاہے ٹرینڈ مضبوط ہو)۔
    مثال: LONG کے لیے RSI 68 سے اوپر ہونا ممنوع ہے (already overbought
    میں خریدنا = چوٹی پر خریدنا)۔ بہترین entry تب ہے جب RSI صحت مند
    "pullback zone" میں ہو — نہ overbought، نہ oversold سے بھی نیچے۔

اصول: سمت خالص گروپ 1 سے آتی ہے۔ گروپ 2 صرف ایک veto/فلٹر ہے —
پوائنٹس نہیں دیتا، صرف "ابھی entry کا وقت اچھا ہے یا نہیں" فیصلہ کرتا ہے۔
=====================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BiasResult:
    bull_points: int = 0
    bear_points: int = 0
    net_score: int = 0
    max_possible: int = 0
    reasons: list[str] = field(default_factory=list)
    entry_timing_ok: bool = True
    entry_timing_note: str = ""

    @property
    def direction(self) -> str:
        # صرف trend-confirmation سکور سمت طے کرتا ہے
        if self.net_score >= 4:
            base_direction = "LONG"
        elif self.net_score <= -4:
            base_direction = "SHORT"
        else:
            return "NEUTRAL"

        # چاہے ٹرینڈ کتنا ہی مضبوط ہو، اگر entry timing خراب ہے
        # (یعنی قیمت پہلے ہی extreme/تھکی ہوئی حالت میں ہے) تو سگنل نہیں
        if not self.entry_timing_ok:
            return "NEUTRAL"

        return base_direction

    @property
    def raw_confidence_pct(self) -> float:
        """0-100 — صرف trend-confirmation اتفاق کی بنیاد پر (AI سے پہلے)۔"""
        if self.max_possible == 0:
            return 0.0
        return round((abs(self.net_score) / self.max_possible) * 100, 1)


def compute_bias(stats: dict) -> BiasResult:
    r = BiasResult()
    r.max_possible = 6  # صرف trend-confirmation کے 6 قابلِ سکور اجزاء

    # ═══════════════════════════════════════════════
    # گروپ 1 — TREND CONFIRMATION (یہی سمت طے کرتا ہے)
    # ═══════════════════════════════════════════════

    stack = stats.get("ema_stack")
    if stack == "BULLISH":
        r.bull_points += 1
        r.reasons.append("EMA9>21>50 bullish stack")
    elif stack == "BEARISH":
        r.bear_points += 1
        r.reasons.append("EMA9<21<50 bearish stack")

    hist = stats.get("macd_histogram")
    if hist is not None:
        if hist > 0:
            r.bull_points += 1
            r.reasons.append("MACD histogram positive")
        elif hist < 0:
            r.bear_points += 1
            r.reasons.append("MACD histogram negative")

    price_vs_ema21 = stats.get("price_vs_ema21_pct")
    if price_vs_ema21 is not None:
        if price_vs_ema21 > 0.15:
            r.bull_points += 1
            r.reasons.append("Price above EMA21")
        elif price_vs_ema21 < -0.15:
            r.bear_points += 1
            r.reasons.append("Price below EMA21")

    price_vs_ema50 = stats.get("price_vs_ema50_pct")
    if price_vs_ema50 is not None:
        if price_vs_ema50 > 0.15:
            r.bull_points += 1
        elif price_vs_ema50 < -0.15:
            r.bear_points += 1

    obv_trend_val = stats.get("obv_trend")
    if obv_trend_val == "BULLISH":
        r.bull_points += 1
        r.reasons.append("OBV trend rising (buying pressure)")
    elif obv_trend_val == "BEARISH":
        r.bear_points += 1
        r.reasons.append("OBV trend falling (selling pressure)")

    vol_ratio = stats.get("volume_ratio")
    change_24h = stats.get("change_24h_pct") or 0
    if vol_ratio is not None and vol_ratio > 1.2:
        if change_24h > 0.5:
            r.bull_points += 1
            r.reasons.append(f"Volume-confirmed upmove ({vol_ratio:.2f}x avg)")
        elif change_24h < -0.5:
            r.bear_points += 1
            r.reasons.append(f"Volume-confirmed downmove ({vol_ratio:.2f}x avg)")

    r.net_score = r.bull_points - r.bear_points

    # ═══════════════════════════════════════════════
    # گروپ 2 — ENTRY TIMING FILTER (veto only، پوائنٹس نہیں)
    # صرف اسی سمت کے لیے چیک ہوتا ہے جو گروپ 1 نے طے کی
    # ═══════════════════════════════════════════════
    likely_direction = "LONG" if r.net_score >= 4 else "SHORT" if r.net_score <= -4 else None

    if likely_direction:
        timing_ok, timing_note = _entry_timing_check(stats, likely_direction)
        r.entry_timing_ok = timing_ok
        r.entry_timing_note = timing_note
        if not timing_ok:
            r.reasons.append(f"Entry timing blocked: {timing_note}")

    return r


def _entry_timing_check(stats: dict, direction: str) -> tuple[bool, str]:
    """
    'Buy the pullback in an uptrend' اصول — صرف overbought نہ ہونا کافی
    نہیں، RSI کو ایک متعین 'صحت مند pullback zone' میں ہونا ضروری ہے۔
    یہ یقینی بناتا ہے کہ ٹرینڈ نے واقعی سانس لی ہے، محض ابھی extreme
    تک نہیں پہنچا — یہ زیادہ سخت اور زیادہ قابلِ بھروسہ شرط ہے۔
    """
    rsi = stats.get("rsi")
    stoch_k = stats.get("stoch_k")

    if rsi is None or stoch_k is None:
        return False, "RSI/Stochastic unavailable"

    if direction == "LONG":
        if not (40 <= rsi <= 62):
            return False, f"RSI {rsi:.1f} not in healthy pullback zone (40-62) for LONG"
        if stoch_k > 75:
            return False, f"Stochastic {stoch_k:.1f} still too hot for LONG entry"
    else:  # SHORT
        if not (38 <= rsi <= 60):
            return False, f"RSI {rsi:.1f} not in healthy pullback zone (38-60) for SHORT"
        if stoch_k < 25:
            return False, f"Stochastic {stoch_k:.1f} still too cold for SHORT entry"

    return True, "Entry timing confirms healthy pullback within trend"


def trend_strength_ok(stats: dict, min_adx: float = 20.0) -> tuple[bool, str]:
    """ADX فلٹر — کمزور/بے سمت مارکیٹ میں سگنل دینا سب سے بڑی غلطی ہے۔"""
    adx = stats.get("adx")
    if adx is None:
        return False, "ADX unavailable — insufficient data"
    if adx < min_adx:
        return False, f"ADX {adx:.1f} < {min_adx} — market is ranging, no reliable trend"
    return True, f"ADX {adx:.1f} confirms an active trend"


def risk_reward_ok(entry: float, stop_loss: float, take_profit_1: float, min_rr: float = 1.5) -> tuple[bool, float]:
    """کم از کم 1:1.5 رسک:ریوارڈ نہ ملے تو سگنل رد — کمزور سیٹ اپس
    خودکار فلٹر ہو جاتے ہیں، جو win-rate کو مستقل بلند رکھتا ہے۔"""
    risk = abs(entry - stop_loss)
    reward = abs(take_profit_1 - entry)
    if risk == 0:
        return False, 0.0
    rr = round(reward / risk, 2)
    return rr >= min_rr, rr