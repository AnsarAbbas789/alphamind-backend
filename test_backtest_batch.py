"""
AlphaMind — test_backtest_batch.py (v2 — includes fee-adjusted net result)
=====================================================================
Run with: python test_backtest_batch.py
=====================================================================
"""

import asyncio
import time

from app.core.candle_fetcher import fetch_candles_range
from app.core.backtester import run_backtest, FEE_PCT_PER_SIDE

START_DATE = "2023-06-01"
END_DATE = "2025-06-01"

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT",
    "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT",
]


async def test_one_symbol(symbol: str) -> dict:
    try:
        candles_1h, _ = await fetch_candles_range(symbol, "1H", START_DATE, END_DATE)
        candles_4h, _ = await fetch_candles_range(symbol, "4H", START_DATE, END_DATE)
        candles_1d, _ = await fetch_candles_range(symbol, "1D", START_DATE, END_DATE)
        report = run_backtest(symbol, candles_1h, candles_4h, candles_1d)
        return {
            "symbol": symbol,
            "total_signals": report.total_signals,
            "wins": report.wins,
            "losses": report.losses,
            "win_rate_pct": report.win_rate_pct,
            "expectancy_r": report.expectancy_r,
            "total_r_multiple": report.total_r_multiple,
            "expectancy_r_net": report.expectancy_r_net,
            "total_r_multiple_net": report.total_r_multiple_net,
            "error": None,
        }
    except Exception as exc:
        return {
            "symbol": symbol, "total_signals": 0, "wins": 0, "losses": 0,
            "win_rate_pct": 0.0, "expectancy_r": 0.0, "total_r_multiple": 0.0,
            "expectancy_r_net": 0.0, "total_r_multiple_net": 0.0,
            "error": str(exc)[:150],
        }


async def main():
    started = time.monotonic()
    results = []

    for i, symbol in enumerate(SYMBOLS, start=1):
        print(f"[{i}/{len(SYMBOLS)}] Testing {symbol}...")
        result = await test_one_symbol(symbol)
        results.append(result)

        if result["error"]:
            print(f"    ERROR: {result['error']}")
        else:
            print(f"    {result['total_signals']} signals, "
                  f"{result['wins']}W/{result['losses']}L, "
                  f"win rate {result['win_rate_pct']}%, "
                  f"expectancy {result['expectancy_r']}R (gross) / "
                  f"{result['expectancy_r_net']}R (net of fees)")

        if i < len(SYMBOLS):
            await asyncio.sleep(1.5)

    elapsed = round(time.monotonic() - started, 1)

    print("\n" + "=" * 90)
    print(f"PER-SYMBOL RESULTS  ({START_DATE} to {END_DATE})  |  fee assumption: {FEE_PCT_PER_SIDE}% per side")
    print("=" * 90)
    print(f"{'Symbol':<10}{'Signals':>9}{'Wins':>7}{'Losses':>8}{'WinRate%':>10}{'Gross(R)':>11}{'Net(R)':>10}")
    print("-" * 90)
    for r in results:
        if r["error"]:
            print(f"{r['symbol']:<10}  ERROR: {r['error']}")
        else:
            print(f"{r['symbol']:<10}{r['total_signals']:>9}{r['wins']:>7}{r['losses']:>8}"
                  f"{r['win_rate_pct']:>10}{r['expectancy_r']:>11}{r['expectancy_r_net']:>10}")

    total_signals = sum(r["total_signals"] for r in results)
    total_wins = sum(r["wins"] for r in results)
    total_losses = sum(r["losses"] for r in results)
    total_r = round(sum(r["total_r_multiple"] for r in results), 3)
    total_r_net = round(sum(r["total_r_multiple_net"] for r in results), 3)
    closed = total_wins + total_losses
    overall_win_rate = round((total_wins / closed) * 100, 2) if closed else 0.0
    overall_expectancy = round(total_r / closed, 3) if closed else 0.0
    overall_expectancy_net = round(total_r_net / closed, 3) if closed else 0.0

    print("\n" + "=" * 90)
    print("AGGREGATE (all 8 coins combined)")
    print("=" * 90)
    print(f"Total signals:            {total_signals}")
    print(f"Total wins:                {total_wins}")
    print(f"Total losses:              {total_losses}")
    print(f"Overall win rate:          {overall_win_rate}%")
    print(f"Overall expectancy (gross):  {overall_expectancy}R   <- fee کے بغیر")
    print(f"Overall expectancy (NET):    {overall_expectancy_net}R   <- fee کاٹ کر، اصل نمبر")
    print(f"Time taken:                {elapsed} seconds")
    print("=" * 90)

    if closed >= 30:
        if overall_expectancy_net > 0:
            print("\n✅ Positive NET expectancy — system is statistically profitable even after fees.")
        else:
            print("\n⚠️ Expectancy turns negative after fees — needs tuning (bigger TPs or fewer, higher-quality trades).")
    else:
        print(f"\n⚠️ Only {closed} closed trades — still a small sample.")


if __name__ == "__main__":
    asyncio.run(main())