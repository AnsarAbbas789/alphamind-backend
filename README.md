# AlphaMind AI Signal Engine (Python / FastAPI)

پروفیشنل، multi-LLM ensemble پر مبنی AI Trading Signal انجن۔ یہ آپ کے
موجودہ AlphaMind JS پروجیکٹ (`dashboard.html`, `ai-signal-fixed.js`)
کے ساتھ مل کر کام کرتا ہے — صرف `CONFIG.AI_SIGNAL_ENDPOINT` کا URL
تبدیل کرنا ہوگا، باقی کسی JS فائل میں تبدیلی درکار نہیں۔

---

## انجن میں کیا شامل ہے

- **11 آزاد تکنیکی انڈیکیٹرز**: RSI, MACD, EMA-stack(9/21/50/200),
  Bollinger Bands, ADX, ATR, Volume-ratio, Stochastic %K/%D, OBV-trend,
  CCI, Support/Resistance
- **Multi-Timeframe Confirmation**: 15m + 1H + 4H + 1D — کم از کم 3/4
  متفق نہ ہوں تو سگنل نہیں بنتا
- **5 LLM Ensemble Voting**: Gemini, Groq, Cerebras, Mistral, OpenRouter
  — majority vote سے حتمی سمت (کوئی ایک ناکام ہو تو باقی چلتے رہتے ہیں)
- **Risk:Reward فلٹر**: کم از کم 1:1.5 نہ ملے تو سگنل رد
- **ADX ٹرینڈ فلٹر**: کمزور/رینج والی مارکیٹ میں سگنل مکمل بلاک
- **Backtester**: پرانے ڈیٹا پر حقیقی win-rate ٹیسٹ کرنے کا ماڈیول
- **5 Exchanges** (candle data): Bybit (اولین ترجیح) → Binance → OKX →
  BingX → KuCoin (خودکار fallback)
- **Supabase JWT Verification**: صرف لاگ ان یوزرز سگنل بنا سکتے ہیں

---

## قدم 1 — لوکل کمپیوٹر پر سیٹ اپ (VS Code)

```bash
# پروجیکٹ فولڈر میں جائیں
cd alphamind-engine

# ورچوئل ماحول بنائیں (تجویز کردہ)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# پیکجز انسٹال کریں
pip install -r requirements.txt
```

## قدم 2 — `.env` فائل بنانا

`.env.example` کی کاپی بنا کر نام `.env` رکھیں، پھر اصل keys ڈالیں:

```bash
copy .env.example .env        # Windows
# cp .env.example .env        # Mac/Linux
```

`.env` میں یہ چیزیں ضرور بھریں:

| متغیر | کہاں سے ملے گا |
|---|---|
| `GEMINI_API_KEY` | پہلے سے آپ کے پاس موجود |
| `GROQ_API_KEY` | پہلے سے آپ کے پاس موجود |
| `CEREBRAS_API_KEY` | cloud.cerebras.ai سے مفت |
| `MISTRAL_API_KEY` | console.mistral.ai سے مفت |
| `OPENROUTER_API_KEY` | openrouter.ai سے مفت |
| `SUPABASE_URL` | پہلے سے `supabase-config.js` میں موجود ہے، وہی یہاں ڈالیں |
| `SUPABASE_JWT_SECRET` | Supabase Dashboard → Project Settings → API → **JWT Secret** |
| `ALLOWED_ORIGINS` | جہاں آپ کی dashboard.html hosted ہوگی (مثلاً `https://alphamind.up.railway.app`) |

**یاد رہے:** `.env` فائل کبھی کسی کو نہ دکھائیں، نہ GitHub پر push کریں (`.gitignore` میں پہلے سے شامل ہے)۔

## قدم 3 — لوکل ٹیسٹ چلانا

```bash
uvicorn app.main:app --reload --port 8000
```

Browser میں کھولیں: `http://localhost:8000/health` — یہاں آپ کو نظر آئے گا کتنی LLM keys فعال ہیں، Supabase سیٹ اپ ہے یا نہیں، وغیرہ۔

## قدم 4 — dashboard.html سے جوڑنا

صرف **ایک لائن** بدلنی ہے — `config.js` میں:

```js
AI_SIGNAL_ENDPOINT: "http://localhost:8000/api/signal"   // لوکل ٹیسٹ کے لیے
// یا deploy کے بعد:
AI_SIGNAL_ENDPOINT: "https://your-app.up.railway.app/api/signal"
```

`ai-signal-fixed.js` میں **کوئی تبدیلی درکار نہیں** — وہ پہلے سے
`CONFIG.AI_SIGNAL_ENDPOINT` کو صحیح شکل میں کال کر رہی ہے۔

## قدم 5 — Railway/Render پر Deploy کرنا

1. اس `alphamind-engine` فولڈر کو ایک نئے **پرائیویٹ** GitHub ریپو میں پش کریں (`.env` خودکار ignore ہوگی)۔
2. Railway.app یا Render.com پر "New Web Service" بنائیں، اسی ریپو سے جوڑیں۔
3. Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. Environment Variables سیکشن میں `.env` کی تمام قدریں manually ڈالیں (یہیں پر keys کا اصل، محفوظ مقام ہے)۔
5. Deploy مکمل ہونے پر ملنے والا URL `config.js` کے `AI_SIGNAL_ENDPOINT` میں ڈال دیں۔

## Win-Rate کی تصدیق (Backtest)

```bash
curl -X POST http://localhost:8000/api/backtest \
  -H "Authorization: Bearer <supabase_access_token>" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "BTCUSDT", "candle_limit": 2000}'
```

جواب میں `win_rate_pct` نظر آئے گا — یہ اصل، تاریخی ڈیٹا پر مبنی نتیجہ
ہے (اندازہ نہیں)۔ اگر 60% سے کم آئے تو `ATR_TP_MULTIPLIERS`,
`MIN_RISK_REWARD`, یا `MIN_ADX` کی قدریں `signal_composer.py` اور
`backtester.py` میں ایڈجسٹ کر کے دوبارہ ٹیسٹ کریں — دونوں فائلیں
ایک جیسی منطق استعمال کرتی ہیں، اس لیے backtest کا نتیجہ live پر بھی
درست لاگو ہوگا۔

## فولڈر ڈھانچہ

```
alphamind-engine/
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── app/
    ├── main.py                    # FastAPI entry point
    ├── config.py                  # Settings/.env loader
    ├── indicators/
    │   ├── technical.py           # 11 انڈیکیٹرز
    │   └── mtf_confirmation.py    # Multi-timeframe confirmation
    ├── core/
    │   ├── rule_engine.py         # Bias scoring + ADX/RR فلٹرز
    │   ├── prompt_builder.py      # AI کے لیے prompt تیار کرنا
    │   ├── signal_composer.py     # سب کچھ یکجا کر کے حتمی سگنل
    │   ├── candle_fetcher.py      # Bybit-first 5-exchange fetcher
    │   ├── auth.py                # Supabase JWT verification
    │   └── backtester.py          # Win-rate تصدیق
    ├── llm/
    │   ├── base.py                # مشترک provider منطق
    │   ├── providers.py           # 5 LLM adapters
    │   ├── registry.py            # صرف موجود keys فعال ہوں
    │   └── ensemble.py            # Voting منطق
    └── routes/
        └── signal.py              # /api/signal, /api/backtest
```