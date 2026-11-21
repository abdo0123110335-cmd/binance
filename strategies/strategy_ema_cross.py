# -*- coding: utf-8 -*-
"""مثال لاستراتيجية جديدة: تقاطع EMA9/EMA21 صاعد على فريم الساعة + سيولة + RSI.

لإضافة استراتيجية خاصة بيك: انسخ الملف ده باسم strategy_<اسمك>.py داخل مجلد strategies
وعدّل NAME و DESC ودالة analyze. التطبيق هيلاقيها لوحده ويعمل لها كارت في الشاشة الرئيسية.
"""
import binance_utils as bu

NAME = "تقاطع EMA صاعد (مثال)"
DESC = "EMA9 يعبر EMA21 لأعلى على 1h\nفوق EMA200 + RSI 50-70 + حجم أعلى من المتوسط"

INTERVAL = "1h"
MIN_QUOTE_VOLUME = 10_000_000
MAX_RISK_PCT = 7.0


def analyze(symbol):
    """يرجع dict بالإشارة أو None."""
    k = bu.get_klines(symbol, INTERVAL, 300)
    c, h, l, v = k["c"], k["h"], k["l"], k["v"]
    if len(c) < 220:
        return None
    e9, e21, e200 = bu.ema(c, 9), bu.ema(c, 21), bu.ema(c, 200)
    rsi = bu.rsi(c, 14)
    atr = bu.atr(h, l, c, 14)
    vol_ma = bu.sma(v, 20)

    crossed = e9[-1] > e21[-1] and (e9[-2] <= e21[-2] or e9[-3] <= e21[-3])
    if not crossed or not c[-1] > e200[-1]:
        return None
    if not 50 <= rsi[-1] <= 70 or not v[-1] > 1.5 * vol_ma[-1]:
        return None

    entry = float(c[-1])
    sl = float(l[-10:].min() - 0.35 * atr[-1])
    risk = entry - sl
    if risk <= 0 or risk / entry * 100 > MAX_RISK_PCT:
        return None
    return {"entry": entry, "sl": sl, "tp1": entry + 1.5 * risk, "tp2": entry + 2.5 * risk,
            "notes": [f"EMA9/21 تقاطع صاعد", f"RSI {rsi[-1]:.1f}", f"حجم {v[-1] / vol_ma[-1]:.1f}x المتوسط"]}


def scan_all():
    symbols = bu.get_symbols(MIN_QUOTE_VOLUME)
    print(f"🔎 {len(symbols)} زوج | فريم {INTERVAL} | {NAME}")
    found = 0
    for i, sym in enumerate(symbols, 1):
        bu.check_stop()
        try:
            sig = analyze(sym)
        except Exception as exc:  # خطأ شبكة/بيانات في زوج واحد ما يوقفش الفحص
            print(f"[{i}/{len(symbols)}] {sym}: تخطي ({type(exc).__name__})")
            continue
        if sig:
            found += 1
            bu.report_buy(sym, sig["entry"], sig["sl"], sig["tp1"], sig["tp2"], sig["notes"], NAME)
        bu.sleep(0.08)
    print(f"\n✅ تم فحص {len(symbols)} | إشارات: {found}")
    if not found:
        print("⚪ لا توجد إشارات الآن.")
