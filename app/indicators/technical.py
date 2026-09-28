"""
AlphaMind — indicators/technical.py
=====================================================================
تمام تکنیکی حساب کتاب یہاں ہے۔ JS ورژن سے زیادہ درست اس لیے کہ:
  - Wilder's smoothing (RSI/ATR) پوری تاریخ پر لگاتار apply ہوتی ہے،
    ہر بار سے دوبارہ شروع نہیں ہوتی (جیسا کہ پرانی JS میں غلطی تھی)۔
  - pandas vectorized حساب سے کوئی رَاؤنڈنگ ڈرفٹ نہیں۔
  - ہر فنکشن pure اور مکمل طور پر ٹیسٹ کے قابل ہے (backtester اسی کو
    استعمال کرتا ہے، اس لیے live اور backtest ہمیشہ ایک جیسا نتیجہ
    دیتے ہیں — یہ سب سے اہم اصول ہے: "backtest جو دیکھے وہی live دے")۔

⚠️ 2 ستمبر 2026 پرفارمنس فکس: نیچے compute_all_series() شامل کیا گیا
ہے — یہ compute_all() جیسا ہی حساب پوری تاریخ کے لیے ایک ساتھ
(vectorized) کرتا ہے، صرف backtester.py استعمال کرتا ہے تاکہ ہزاروں
candles پر ہزاروں بار دوبارہ حساب لگانے کی بجائے صرف ایک بار حساب لگے۔
compute_all() (لائیو انجن) بالکل ویسے ہی رہا، کوئی تبدیلی نہیں۔
=====================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def to_dataframe(candles: list[dict]) -> pd.DataFrame:
    """candles: [{time, open, high, low, close, volume}, ...] -> sorted DataFrame"""
    df = pd.DataFrame(candles)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).sort_values("time").reset_index(drop=True)
    return df


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder's smoothing == ewm with alpha = 1/period
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "histogram": hist})


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def bollinger_bands(series: pd.Series, period: int = 20, std_mult: float = 2.0) -> pd.DataFrame:
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    bandwidth = (upper - lower) / mid.replace(0, np.nan)
    return pd.DataFrame({"bb_mid": mid, "bb_upper": upper, "bb_lower": lower, "bb_bandwidth": bandwidth})


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index — ٹرینڈ کی 'طاقت' ماپتا ہے (سمت نہیں)۔
    ADX < 20 = کمزور/بے سمت مارکیٹ (چاپی/رینج) — یہاں سگنل نہ دینا بہتر۔
    ADX > 25 = حقیقی ٹرینڈ موجود ہے۔"""
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    atr_smooth = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr_smooth.replace(0, np.nan))
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr_smooth.replace(0, np.nan))

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean().fillna(0)


def volume_confirmation(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """موجودہ حجم کا اوسط حجم سے تناسب — 1.0 سے اوپر یعنی معمول سے زیادہ
    خریداری/فروخت کی تصدیق (سگنل کو زیادہ قابلِ بھروسہ بناتا ہے)۔"""
    avg_vol = df["volume"].rolling(period).mean()
    return (df["volume"] / avg_vol.replace(0, np.nan)).fillna(1.0)


def support_resistance(df: pd.DataFrame, lookback: int = 50) -> dict:
    """حالیہ swing high/low — انٹری کے قریب اہم لیول کو مدنظر رکھنے کے لیے۔"""
    window = df.tail(lookback)
    return {
        "resistance": float(window["high"].max()),
        "support": float(window["low"].min()),
    }


def stochastic_oscillator(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """Stochastic %K/%D — momentum indicator، RSI سے مختلف انداز میں
    overbought/oversold ماپتا ہے (candle range کی بنیاد پر، صرف closes پر نہیں)۔
    <20 = oversold, >80 = overbought۔"""
    low_min = df["low"].rolling(k_period).min()
    high_max = df["high"].rolling(k_period).max()
    denom = (high_max - low_min).replace(0, np.nan)
    percent_k = 100 * (df["close"] - low_min) / denom
    percent_d = percent_k.rolling(d_period).mean()
    return pd.DataFrame({"stoch_k": percent_k, "stoch_d": percent_d})


def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume — قیمت کی سمت کے ساتھ حجم جمع/گھٹا کر یہ بتاتا
    ہے کہ 'سمارٹ منی' خرید رہی ہے یا بیچ رہی ہے، چاہے قیمت ابھی خاموش ہو۔
    OBV کا اپنا ٹرینڈ (rising/falling) قیمت کی حرکت کی تصدیق کے لیے اہم ہے۔"""
    direction = np.sign(df["close"].diff()).fillna(0)
    return (direction * df["volume"]).cumsum()


def obv_trend(df: pd.DataFrame, period: int = 20) -> str:
    series = obv(df)
    if len(series) < period + 1:
        return "NEUTRAL"
    slope = series.iloc[-1] - series.iloc[-period]
    if slope > 0:
        return "BULLISH"
    if slope < 0:
        return "BEARISH"
    return "NEUTRAL"


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Commodity Channel Index — قیمت کے اپنی اوسط سے انحراف کی پیمائش۔
    >+100 = مضبوط اوپر کا momentum, <-100 = مضبوط نیچے کا momentum۔
    RSI/Stochastic سے آزاد ریاضی استعمال کرتا ہے، اس لیے confluence
    میں اضافی، غیر-مکرر تصدیق فراہم کرتا ہے۔"""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    sma = typical_price.rolling(period).mean()
    mean_dev = typical_price.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (typical_price - sma) / (0.015 * mean_dev.replace(0, np.nan))


def compute_all(df: pd.DataFrame) -> dict:
    """ایک کینڈل سیریز پر تمام indicators چلا کر آخری قدروں کا خلاصہ دیتا ہے۔
    یہی خلاصہ prompt builder اور rule engine دونوں استعمال کرتے ہیں —
    ایک ہی حساب، دو جگہ استعمال (کبھی الگ الگ حساب نہیں لگتے، تضاد ناممکن)۔
    صرف LIVE انجن (ایک وقت میں ایک ہی درخواست) استعمال کرتا ہے۔

    مجموعی طور پر 9+ آزاد indicators استعمال ہوتے ہیں:
    1.RSI  2.MACD  3.EMA-stack  4.Bollinger Bands  5.ADX  6.ATR
    7.Volume ratio  8.Stochastic %K/%D  9.OBV trend  10.CCI  11.Support/Resistance
    """
    closes = df["close"]

    rsi_series = rsi(closes, 14)
    macd_df = macd(closes)
    bb_df = bollinger_bands(closes)
    adx_series = adx(df, 14)
    atr_series = atr(df, 14)
    vol_conf = volume_confirmation(df)
    stoch_df = stochastic_oscillator(df)
    cci_series = cci(df)
    obv_bias = obv_trend(df)

    ema9 = ema(closes, 9)
    ema21 = ema(closes, 21)
    ema50 = ema(closes, 50)
    ema200 = ema(closes, 200) if len(closes) >= 200 else pd.Series([np.nan] * len(closes))

    last = closes.iloc[-1]
    sr = support_resistance(df)

    ema_stack = (
        "BULLISH" if ema9.iloc[-1] > ema21.iloc[-1] > ema50.iloc[-1]
        else "BEARISH" if ema9.iloc[-1] < ema21.iloc[-1] < ema50.iloc[-1]
        else "MIXED"
    )

    change_24h = _pct_change(closes, 24)
    change_7d = _pct_change(closes, 168)

    return {
        "price": float(last),
        "change_24h_pct": change_24h,
        "change_7d_pct": change_7d,
        "rsi": _safe_last(rsi_series),
        "macd": _safe_last(macd_df["macd"]),
        "macd_signal": _safe_last(macd_df["signal"]),
        "macd_histogram": _safe_last(macd_df["histogram"]),
        "ema9": _safe_last(ema9),
        "ema21": _safe_last(ema21),
        "ema50": _safe_last(ema50),
        "ema200": _safe_last(ema200) if not ema200.isna().all() else None,
        "ema_stack": ema_stack,
        "price_vs_ema9_pct": _pct_diff(last, ema9.iloc[-1]),
        "price_vs_ema21_pct": _pct_diff(last, ema21.iloc[-1]),
        "price_vs_ema50_pct": _pct_diff(last, ema50.iloc[-1]),
        "bb_upper": _safe_last(bb_df["bb_upper"]),
        "bb_lower": _safe_last(bb_df["bb_lower"]),
        "bb_bandwidth": _safe_last(bb_df["bb_bandwidth"]),
        "adx": _safe_last(adx_series),
        "atr": _safe_last(atr_series),
        "volume_ratio": _safe_last(vol_conf),
        "stoch_k": _safe_last(stoch_df["stoch_k"]),
        "stoch_d": _safe_last(stoch_df["stoch_d"]),
        "obv_trend": obv_bias,
        "cci": _safe_last(cci_series),
        "resistance": sr["resistance"],
        "support": sr["support"],
    }


def compute_all_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    نیا (پرفارمنس فکس) — compute_all() کی طرح مگر پوری تاریخ کے لیے ایک
    ساتھ (vectorized)۔ صرف وہ اجزاء شامل ہیں جو backtester کو ہر
    candle پر چاہئیں (rule_engine.compute_bias/trend_strength_ok/
    entry-timing کے لیے) — CCI, Bollinger, Support/Resistance شامل
    نہیں کیونکہ وہ backtest کے فیصلے میں استعمال ہی نہیں ہوتے، اس
    لیے انہیں چھوڑ کر رفتار مزید بہتر کی گئی ہے۔

    صرف backtester.py استعمال کرتا ہے — Live انجن اب بھی compute_all()
    استعمال کرتا ہے، بالکل ویسے ہی جیسے پہلے تھا۔
    """
    closes = df["close"]

    rsi_series = rsi(closes, 14)
    macd_df = macd(closes)
    adx_series = adx(df, 14)
    atr_series = atr(df, 14)
    vol_conf = volume_confirmation(df)
    stoch_df = stochastic_oscillator(df)
    obv_series = obv(df)

    ema9 = ema(closes, 9)
    ema21 = ema(closes, 21)
    ema50 = ema(closes, 50)

    ema_stack = pd.Series(
        np.select(
            [(ema9 > ema21) & (ema21 > ema50), (ema9 < ema21) & (ema21 < ema50)],
            ["BULLISH", "BEARISH"],
            default="MIXED",
        ),
        index=df.index,
    )

    price_vs_ema21_pct = ((closes - ema21) / ema21.replace(0, np.nan)) * 100
    price_vs_ema50_pct = ((closes - ema50) / ema50.replace(0, np.nan)) * 100

    obv_slope = obv_series.diff(20)
    obv_trend_series = pd.Series(
        np.select([obv_slope > 0, obv_slope < 0], ["BULLISH", "BEARISH"], default="NEUTRAL"),
        index=df.index,
    )

    change_24h_series = closes.pct_change(24) * 100

    return pd.DataFrame({
        "price": closes,
        "rsi": rsi_series,
        "macd_histogram": macd_df["histogram"],
        "ema_stack": ema_stack,
        "price_vs_ema21_pct": price_vs_ema21_pct.round(2),
        "price_vs_ema50_pct": price_vs_ema50_pct.round(2),
        "adx": adx_series,
        "atr": atr_series,
        "volume_ratio": vol_conf,
        "stoch_k": stoch_df["stoch_k"],
        "obv_trend": obv_trend_series,
        "change_24h_pct": change_24h_series.round(2),
    })


def _safe_last(series: pd.Series) -> float | None:
    val = series.iloc[-1]
    return None if pd.isna(val) else round(float(val), 6)


def _pct_change(series: pd.Series, bars_back: int) -> float:
    if len(series) <= bars_back:
        bars_back = len(series) - 1
    if bars_back <= 0:
        return 0.0
    old = series.iloc[-bars_back - 1]
    new = series.iloc[-1]
    if old == 0:
        return 0.0
    return round(((new - old) / old) * 100, 2)


def _pct_diff(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return round(((a - b) / b) * 100, 2)