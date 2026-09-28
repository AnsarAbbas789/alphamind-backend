"""
AlphaMind — core/backtester.py  (v4 — fee-adjusted نتیجہ)
=====================================================================
سب سے اہم اصول: "backtest جو دیکھے وہی live دے۔"

⚠️ 3 ستمبر 2026 اضافہ: اب ہر trade پر حقیقی exchange fees (entry +
exit، round-trip) کا حساب بھی رکھا جاتا ہے — پہلے صرف "gross" (fee کے
بغیر) نتیجہ ملتا تھا جو حقیقی trading سے زیادہ خوش کن دکھا سکتا تھا۔
اب دونوں نمبر ملتے ہیں: r_multiple (fee کے بغیر) اور r_multiple_net
(fee کاٹ کر) — کاروباری فیصلہ ہمیشہ net نمبر پر کرنا چاہیے۔

FEE_PCT_PER_SIDE = 0.1% Bybit کی عام spot taker fee ہے (VIP level یا
futures کی صورت میں یہ کم ہو سکتی ہے — signal.py استعمال کرنے سے پہلے
اپنے اصل Bybit account کا fee tier چیک کر کے یہ نمبر درست کریں)۔
=====================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.core import rule_engine
from app.indicators.mtf_confirmation import timeframe_bias_series
from app.indicators.technical import compute_all_series, to_dataframe

MIN_ADX = 20.0
MIN_RISK_REWARD = 1.5
ATR_SL_MULTIPLIER = 2.0
ATR_TP_MULTIPLIERS = [3.0, 4.5, 6.5, 9.0]
WARMUP_BARS = 210  # EMA200 وغیرہ کے لیے کم از کم اتنی پرانی کینڈلز چاہئیں

# ── Fees ──────────────────────────────────────────────────────────
FEE_PCT_PER_SIDE = 0.1  # % — Bybit spot taker (اپنے اصل account کے مطابق بدلیں)


@dataclass
class TradeResult:
    entry_index: int
    direction: str
    entry: float
    stop_loss: float
    take_profit_1: float
    outcome: str  # "WIN" | "LOSS" | "OPEN_AT_END"
    bars_held: int = 0
    r_multiple: float = 0.0       # WIN پر رسک کا کتنا گنا کمایا، LOSS پر -1.0 (fee کے بغیر)
    r_multiple_net: float = 0.0   # اوپر والا مائنس round-trip fee (حقیقی نتیجہ)


@dataclass
class BacktestReport:
    symbol: str = ""
    total_signals: int = 0
    wins: int = 0
    losses: int = 0
    open_at_end: int = 0
    win_rate_pct: float = 0.0
    avg_bars_held: float = 0.0
    expectancy_r: float = 0.0           # fee کے بغیر
    total_r_multiple: float = 0.0
    expectancy_r_net: float = 0.0       # fee کاٹ کر — اصل، قابلِ بھروسہ نمبر
    total_r_multiple_net: float = 0.0
    trades: list[TradeResult] = field(default_factory=list)


def _simulate_forward(candles: list[dict], start_index: int, direction: str, entry: float,
                       stop_loss: float, take_profit_1: float, max_lookahead: int = 200) -> TradeResult:
    is_long = direction == "LONG"
    end = min(len(candles), start_index + max_lookahead)
    risk = abs(entry - stop_loss)
    reward = abs(take_profit_1 - entry)
    win_r = round(reward / risk, 3) if risk else 0.0

    # Round-trip fee کو R کی زبان میں تبدیل کریں (entry fee + exit fee،
    # دونوں entry price کے تناسب سے، risk کے مقابلے میں)
    fee_in_r = round(((FEE_PCT_PER_SIDE / 100) * 2 * entry) / risk, 4) if risk else 0.0

    for i in range(start_index, end):
        c = candles[i]
        if is_long:
            if c["low"] <= stop_loss:
                return TradeResult(start_index, direction, entry, stop_loss, take_profit_1, "LOSS",
                                    i - start_index, r_multiple=-1.0, r_multiple_net=round(-1.0 - fee_in_r, 4))
            if c["high"] >= take_profit_1:
                return TradeResult(start_index, direction, entry, stop_loss, take_profit_1, "WIN",
                                    i - start_index, r_multiple=win_r, r_multiple_net=round(win_r - fee_in_r, 4))
        else:
            if c["high"] >= stop_loss:
                return TradeResult(start_index, direction, entry, stop_loss, take_profit_1, "LOSS",
                                    i - start_index, r_multiple=-1.0, r_multiple_net=round(-1.0 - fee_in_r, 4))
            if c["low"] <= take_profit_1:
                return TradeResult(start_index, direction, entry, stop_loss, take_profit_1, "WIN",
                                    i - start_index, r_multiple=win_r, r_multiple_net=round(win_r - fee_in_r, 4))

    return TradeResult(start_index, direction, entry, stop_loss, take_profit_1,
                        "OPEN_AT_END", end - start_index, r_multiple=0.0, r_multiple_net=0.0)


def run_backtest(
    symbol: str,
    candles_1h: list[dict],
    candles_4h: list[dict] | None = None,
    candles_1d: list[dict] | None = None,
    step: int = 4,
) -> BacktestReport:
    report = BacktestReport(symbol=symbol)

    if len(candles_1h) < WARMUP_BARS + 50:
        return report

    df_1h = to_dataframe(candles_1h)
    if len(df_1h) < WARMUP_BARS + 50:
        return report

    stats_1h = compute_all_series(df_1h)
    bias_1h_tf = timeframe_bias_series(df_1h)
    times_1h = df_1h["time"].to_numpy()
    n = len(df_1h)

    use_mtf = bool(candles_4h) and bool(candles_1d)
    if use_mtf:
        df_4h = to_dataframe(candles_4h)
        df_1d = to_dataframe(candles_1d)
        bias_4h_series = timeframe_bias_series(df_4h).to_numpy()
        bias_1d_series = timeframe_bias_series(df_1d).to_numpy()
        times_4h = df_4h["time"].to_numpy()
        times_1d = df_1d["time"].to_numpy()

    for i in range(WARMUP_BARS, n - 1, step):
        row = stats_1h.iloc[i]

        if pd.isna(row["adx"]) or pd.isna(row["atr"]) or pd.isna(row["rsi"]) or pd.isna(row["stoch_k"]):
            continue

        stats = {
            "price": row["price"],
            "rsi": row["rsi"],
            "macd_histogram": row["macd_histogram"],
            "ema_stack": row["ema_stack"],
            "price_vs_ema21_pct": row["price_vs_ema21_pct"],
            "price_vs_ema50_pct": row["price_vs_ema50_pct"],
            "adx": row["adx"],
            "atr": row["atr"],
            "volume_ratio": row["volume_ratio"],
            "stoch_k": row["stoch_k"],
            "obv_trend": row["obv_trend"],
            "change_24h_pct": row["change_24h_pct"],
        }

        trend_ok, _ = rule_engine.trend_strength_ok(stats, min_adx=MIN_ADX)
        if not trend_ok:
            continue

        bias = rule_engine.compute_bias(stats)
        if bias.direction == "NEUTRAL":
            continue

        if stats["volume_ratio"] is None or pd.isna(stats["volume_ratio"]) or stats["volume_ratio"] < 1.1:
            continue

        if use_mtf:
            current_time = int(times_1h[i])
            idx_4h = np.searchsorted(times_4h, current_time, side="right") - 1
            idx_1d = np.searchsorted(times_1d, current_time, side="right") - 1

            per_tf = {"1H": bias_1h_tf.iloc[i]}
            per_tf["4H"] = bias_4h_series[idx_4h] if idx_4h >= 0 else "INSUFFICIENT_DATA"
            per_tf["1D"] = bias_1d_series[idx_1d] if idx_1d >= 0 else "INSUFFICIENT_DATA"

            evaluated = {tf: b for tf, b in per_tf.items() if b in ("BULLISH", "BEARISH", "NEUTRAL")}
            bullish_count = sum(1 for b in evaluated.values() if b == "BULLISH")
            bearish_count = sum(1 for b in evaluated.values() if b == "BEARISH")
            total = len(evaluated)

            if total == 0:
                aligned_direction, confirmed = "MIXED", False
            elif bullish_count >= max(3, total - 1) and bullish_count > bearish_count:
                aligned_direction, confirmed = "BULLISH", total >= 3
            elif bearish_count >= max(3, total - 1) and bearish_count > bullish_count:
                aligned_direction, confirmed = "BEARISH", total >= 3
            else:
                aligned_direction, confirmed = "MIXED", False

            if not confirmed:
                continue
            bias_dir_simple = "BULLISH" if bias.direction == "LONG" else "BEARISH"
            if bias_dir_simple != aligned_direction:
                continue

        entry = stats["price"]
        atr_val = stats["atr"] or (entry * 0.01)
        is_long = bias.direction == "LONG"
        stop_loss = entry - (atr_val * ATR_SL_MULTIPLIER) if is_long else entry + (atr_val * ATR_SL_MULTIPLIER)
        tp1 = entry + (atr_val * ATR_TP_MULTIPLIERS[0]) if is_long else entry - (atr_val * ATR_TP_MULTIPLIERS[0])

        rr_ok, _ = rule_engine.risk_reward_ok(entry, stop_loss, tp1, min_rr=MIN_RISK_REWARD)
        if not rr_ok:
            continue

        trade = _simulate_forward(candles_1h, i + 1, bias.direction, entry, stop_loss, tp1)
        report.trades.append(trade)

    report.total_signals = len(report.trades)
    report.wins = sum(1 for t in report.trades if t.outcome == "WIN")
    report.losses = sum(1 for t in report.trades if t.outcome == "LOSS")
    report.open_at_end = sum(1 for t in report.trades if t.outcome == "OPEN_AT_END")

    closed = report.wins + report.losses
    report.win_rate_pct = round((report.wins / closed) * 100, 2) if closed else 0.0
    report.avg_bars_held = round(
        sum(t.bars_held for t in report.trades) / len(report.trades), 1
    ) if report.trades else 0.0

    closed_trades = [t for t in report.trades if t.outcome in ("WIN", "LOSS")]
    report.total_r_multiple = round(sum(t.r_multiple for t in closed_trades), 3)
    report.expectancy_r = round(report.total_r_multiple / len(closed_trades), 3) if closed_trades else 0.0

    report.total_r_multiple_net = round(sum(t.r_multiple_net for t in closed_trades), 3)
    report.expectancy_r_net = round(report.total_r_multiple_net / len(closed_trades), 3) if closed_trades else 0.0

    return report