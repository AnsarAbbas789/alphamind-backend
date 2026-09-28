"""
AlphaMind — core/prompt_builder.py
=====================================================================
یہ فنکشن indicators + rule-engine کا bias score + multi-timeframe
confirmation + (نیا) تازہ خبروں کا خلاصہ کو ایک واحد prompt میں یکجا
کرتا ہے جو تمام LLMs کو بھیجا جاتا ہے۔

⚠️ اہم اصول (کبھی نہ بدلیں): AI کو سمت تبدیل کرنے کی اجازت نہیں —
صرف واضح، غیر متنازع reversal pattern پر، اور تبھی وہ وضاحت دینے کا
پابند ہے۔ خبریں بھی اسی اصول کے تابع ہیں — وہ صرف اضافی context ہیں،
کبھی سمت کا فیصلہ نہیں کر سکتیں۔ اس سے AI کی "رائے" یا "خبر" کبھی
ریاضی پر حاوی نہیں ہوتی۔
=====================================================================
"""

from __future__ import annotations

from app.core.rule_engine import BiasResult


def build_prompt(
    symbol: str,
    market_type: str,
    stats: dict,
    bias: BiasResult,
    mtf: dict,
    candles_tail: list[dict],
    news_summary: str = "No recent news available.",
) -> str:
    mtf_lines = "\n".join(
        f"  {tf}: {direction}" for tf, direction in mtf.get("per_timeframe", {}).items()
    )

    last_20 = candles_tail[-20:]
    candle_summary = ", ".join(
        f"[{c['time']},O:{c['open']:.4f},H:{c['high']:.4f},L:{c['low']:.4f},C:{c['close']:.4f}]"
        for c in last_20
    )

    return f"""ANALYZE ONLY THIS COIN: {symbol}
Market Type: {market_type.upper()}

=== PRE-COMPUTED TECHNICAL INDICATORS (11 independent indicators) ===
Current Price: {stats['price']}
24h Change: {stats['change_24h_pct']}%   7d Change: {stats['change_7d_pct']}%

1. RSI(14): {stats['rsi']}  (>68 overbought, <32 oversold)
2. MACD: line={stats['macd']} signal={stats['macd_signal']} histogram={stats['macd_histogram']}
3. EMA Stack: {stats['ema_stack']}  (EMA9={stats['ema9']}, EMA21={stats['ema21']}, EMA50={stats['ema50']}, EMA200={stats['ema200']})
4. Price vs EMA9/21/50: {stats['price_vs_ema9_pct']}% / {stats['price_vs_ema21_pct']}% / {stats['price_vs_ema50_pct']}%
5. Bollinger Bands: upper={stats['bb_upper']} lower={stats['bb_lower']} bandwidth={stats['bb_bandwidth']}
6. ADX(14): {stats['adx']}  (>25 = strong trend, <20 = ranging/weak)
7. ATR(14): {stats['atr']}
8. Volume Ratio (vs 20-period avg): {stats['volume_ratio']}x
9. Stochastic %K/%D: {stats['stoch_k']} / {stats['stoch_d']}  (>80 overbought, <20 oversold)
10. OBV Trend (smart-money flow): {stats['obv_trend']}
11. CCI(20): {stats['cci']}  (>100 strong bullish, <-100 strong bearish)

Support: {stats['support']}   Resistance: {stats['resistance']}

=== MULTI-TIMEFRAME CONTEXT ===
{mtf_lines}
Aligned direction across timeframes: {mtf.get('aligned_direction')} ({mtf.get('agreement_count')}/{mtf.get('total_evaluated')} agree)

=== MECHANICALLY-COMPUTED BIAS SCORE (not opinion — pure arithmetic) ===
Bullish points: {bias.bull_points} | Bearish points: {bias.bear_points} | Net: {bias.net_score} (out of ±{bias.max_possible})
Contributing factors: {"; ".join(bias.reasons) if bias.reasons else "none triggered"}

MANDATORY RULE — your "signal" field MUST match this score:
- Net score >= +4  -> signal MUST be "LONG"
- Net score <= -4  -> signal MUST be "SHORT"
- Otherwise        -> signal MUST be "NEUTRAL"
You may only override this if the last 20 candles below show an extremely obvious
reversal pattern contradicting it — and if you do, you MUST explain exactly why in
"reasoning". Overriding should be rare. NEVER default to LONG out of general optimism.

=== RECENT NEWS (background context ONLY — does not change your signal) ===
{news_summary}
This news is provided purely for your written reasoning/awareness. It has NO authority
to change the "signal" field. Only mention it in "reasoning" if it is directly relevant
to explaining the current price action (e.g. a major event coincides with the move).

Last 20 candles (OHLC): {candle_summary}

Respond with ONLY this JSON, nothing else:
{{"signal": "LONG" or "SHORT" or "NEUTRAL", "confidence": 0-100, "reasoning": "max 2 sentences explaining the decision"}}"""