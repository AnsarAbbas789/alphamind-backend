"""
AlphaMind — core/watchlist.py
=====================================================================
Coins کی وہ فہرست جو 2-سالہ backtest میں تصدیق شدہ (validated) ہیں —
یہی فہرست ہر جگہ استعمال ہو (backtest، scanner، future watchlist)
تاکہ کہیں الگ الگ فہرستیں نہ بنیں اور تضاد پیدا نہ ہو۔ AVAXUSDT
جان بوجھ کر شامل نہیں (منفی net expectancy ملا تھا)۔
=====================================================================
"""

VALIDATED_WATCHLIST_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT",
    "XRPUSDT", "ADAUSDT", "DOGEUSDT",
]