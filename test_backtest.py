"""
AlphaMind — test_backtest.py (v3 — English output, large dataset)
=====================================================================
Standalone backtest runner. No server, no Supabase login needed.
Run with: python test_backtest.py
=====================================================================
"""

import asyncio

from app.core.candle_fetcher import fetch_candles_range
from app.core.backtester import run_backtest

START_DATE = "2023-06-01"
END_DATE = "2025-06-01"


async def main():
    symbol = "BTCUSDT"

    print(f"[1/2] Fetching {symbol} candles from {START_DATE} to {END_DATE}...")
    print("(this may take 1-3 minutes, please wait)")

    candles_1h, ex1 = await fetch_candles_range(symbol, "1H", START_DATE, END_DATE)
    candles_4h, ex2 = await fetch_candles_range(symbol, "4H", START_DATE, END_DATE)
    candles_1d, ex3 = await fetch_candles_range(symbol, "1D", START_DATE, END_DATE)

    print(f"  1H: {len(candles_1h)} candles (from {ex1})")
    print(f"  4H: {len(candles_4h)} candles (from {ex2})")
    print(f"  1D: {len(candles_1d)} candles (from {ex3})")

    print(f"\n[2/2] Running backtest...")
    report = run_backtest(symbol, candles_1h, candles_4h, candles_1d)

    print("\n" + "=" * 50)
    print(f"RESULT — {report.symbol} ({START_DATE} to {END_DATE})")
    print("=" * 50)
    print(f"Total Signals:      {report.total_signals}")
    print(f"Wins:               {report.wins}")
    print(f"Losses:             {report.losses}")
    print(f"Still Open:         {report.open_at_end}")
    print(f"Win Rate:           {report.win_rate_pct}%")
    print(f"Avg Bars Held:      {report.avg_bars_held}")
    print(f"Expectancy (R):     {report.expectancy_r}")
    print(f"Total R Multiple:   {report.total_r_multiple}")
    print("=" * 50)

    if report.expectancy_r > 0:
        print("\n✅ Positive expectancy — system is statistically profitable (before fees).")
    else:
        print("\n⚠️ Negative expectancy — needs tuning.")


if __name__ == "__main__":
    asyncio.run(main())