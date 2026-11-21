# -*- coding: utf-8 -*-
"""أدوات مشتركة لكتابة استراتيجيات جديدة بسرعة (numpy + requests فقط).

الاستخدام داخل أي استراتيجية:
    import binance_utils as bu
    symbols = bu.get_symbols()
    k = bu.get_klines("BTCUSDT", "1h", 300)   # dict من مصفوفات numpy
    bu.check_stop()                            # داخل اللوب ليستجيب زر الإيقاف
    bu.report_buy("BTCUSDT", entry, sl, tp1, tp2, ["ملاحظة 1", "ملاحظة 2"])
"""
from __future__ import annotations

import json
import os
import threading
import time

import numpy as np
import requests

API = "https://api.binance.com/api/v3"
STOP = threading.Event()          # يضبطه التطبيق عند الضغط على زر الإيقاف
_SESSION = requests.Session()

STABLE_BASES = {"USDC", "FDUSD", "TUSD", "USDP", "DAI", "EUR", "AEUR", "USD1", "XUSD", "PYUSD", "BFUSD"}
LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR")


class StopScan(BaseException):
    """إشارة إيقاف نظيفة (BaseException عشان ما تبتلعهاش except Exception)."""


def check_stop():
    if STOP.is_set():
        raise StopScan()


def sleep(seconds: float):
    """نوم يستجيب للإيقاف فوراً."""
    if STOP.wait(max(0.0, seconds)):
        raise StopScan()


def api_get(path: str, params: dict | None = None, timeout: int = 20):
    check_stop()
    r = _SESSION.get(f"{API}/{path}", params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def get_symbols(min_quote_volume: float = 10_000_000, quote: str = "USDT", limit: int = 0) -> list[str]:
    """أزواج USDT مرتبة بحجم التداول 24h، بدون العملات المستقرة والرموز المرفوعة."""
    out = []
    for t in api_get("ticker/24hr"):
        s = t.get("symbol", "")
        if not s.endswith(quote):
            continue
        base = s[: -len(quote)]
        if base in STABLE_BASES or base.endswith(LEVERAGED_SUFFIXES):
            continue
        qv = float(t.get("quoteVolume") or 0)
        if qv >= min_quote_volume:
            out.append((qv, s))
    out.sort(reverse=True)
    syms = [s for _, s in out]
    return syms[:limit] if limit else syms


def get_klines(symbol: str, interval: str = "1h", limit: int = 300) -> dict:
    """شموع مغلقة فقط. يرجع dict: t,o,h,l,c,v,tb (taker buy base)."""
    raw = api_get("klines", {"symbol": symbol, "interval": interval, "limit": limit})
    now_ms = int(time.time() * 1000)
    raw = [r for r in raw if int(r[6]) < now_ms]
    if not raw:
        raise ValueError(f"no closed candles for {symbol} {interval}")
    a = np.array([[float(x) for x in r[:11]] for r in raw])
    return {"t": a[:, 0], "o": a[:, 1], "h": a[:, 2], "l": a[:, 3],
            "c": a[:, 4], "v": a[:, 5], "tb": a[:, 9]}


# ---------------------------- مؤشرات ----------------------------
def sma(x, n):
    x = np.asarray(x, float)
    out = np.full(x.shape, np.nan)
    if len(x) >= n:
        cs = np.cumsum(np.insert(x, 0, 0.0))
        out[n - 1:] = (cs[n:] - cs[:-n]) / n
    return out


def ema(x, n):
    x = np.asarray(x, float)
    a = 2.0 / (n + 1.0)
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    out[: n - 1] = np.nan
    return out


def rsi(x, n=14):
    x = np.asarray(x, float)
    d = np.diff(x, prepend=x[0])
    gain, loss = np.clip(d, 0, None), np.clip(-d, 0, None)
    ag = np.empty_like(x); al = np.empty_like(x)
    ag[0] = al[0] = 0.0
    for i in range(1, len(x)):
        ag[i] = (ag[i - 1] * (n - 1) + gain[i]) / n
        al[i] = (al[i - 1] * (n - 1) + loss[i]) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 100 - 100 / (1 + ag / al)
    out = np.where(al == 0, 100.0, out)
    out[:n] = np.nan
    return out


def atr(h, l, c, n=14):
    h, l, c = (np.asarray(v, float) for v in (h, l, c))
    pc = np.roll(c, 1); pc[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    out = np.empty_like(tr)
    out[0] = tr[0]
    for i in range(1, len(tr)):
        out[i] = (out[i - 1] * (n - 1) + tr[i]) / n
    out[: n - 1] = np.nan
    return out


# ---------------------------- مخرجات ----------------------------
def fmt_price(p: float) -> str:
    return f"{p:.6g}"


def report_buy(symbol, entry, sl, tp1, tp2, notes=None, strategy="", log_file="signals_custom.jsonl"):
    """يطبع إشارة شراء بالشكل القياسي ويحفظها في ملف jsonl."""
    risk = (entry - sl) / entry * 100 if entry else 0.0
    rr1 = (tp1 - entry) / (entry - sl) if entry > sl else 0.0
    print(f"\n🟢 BUY {symbol} | {strategy}")
    print(f"   Entry : {fmt_price(entry)}")
    print(f"   SL    : {fmt_price(sl)} ({risk:.2f}%)")
    print(f"   TP1   : {fmt_price(tp1)} | R:R {rr1:.2f}")
    print(f"   TP2   : {fmt_price(tp2)}")
    for n in (notes or []):
        print(f"      • {n}")
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), "symbol": symbol, "strategy": strategy,
                                "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2,
                                "notes": notes or []}, ensure_ascii=False) + "\n")
    except OSError:
        pass
