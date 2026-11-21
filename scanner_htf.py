"""
Complete Spot Scanner HTF — Binance (Anchored VWAP + Volume Profile + Real Order Flow)
فحص مع فلتر اتجاه 4h إلزامي — إشارات شراء سبوت فقط (بدون تنفيذ أوامر).
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import time

import numpy as np
import pdlite as pd
import requests

# ===================== Settings =====================
INTERVAL = "1h"
LOOKBACK = 300
ANCHOR_N = 60
SWING_SL_N = 20
VA_PCT = 0.70
SL_ATR_MULT = 1.0
MAX_RISK_PCT = 7.0
MIN_TP_BUFFER = 1.003
TP2_RR_MULT = 2.5
TOP_N = 0
MIN_VOLUME = 10_000_000
PAUSE = 0.15
MIN_STAGE1 = 5               # Stage 1 (fast) minimum out of 8
MIN_TOTAL = 9                # Final minimum out of 13
FOOT_CANDLES = 3
AGG_PAGES = 5
MAX_WORKERS = 5
OB_BAND_PCT = 1.0
OB_IMB_MIN = 1.15
WALL_MULT = 4.0
WALL_CHECK_DELAY = 0.45
WALL_MIN_RETENTION = 0.65

# ===================== 4h Trend Filter Settings =====================
HTF_INTERVAL = "4h"
HTF_LOOKBACK = 120
HTF_EMA_PERIOD = 50
HTF_SWING_LOOKBACK = 10
REQUIRE_HTF_BULLISH = True   # True = اشتراط توافق اتجاه 4h قبل الدخول في المرحلة الثانية

EXCLUDED_BASES = {
    "USDT", "USDC", "FDUSD", "TUSD", "USDP", "DAI",
    "USD1", "RLUSD", "RWUSD", "U", "PYUSD", "EUR", "BUSD"
}

API = "https://api.binance.com/api/v3"


# ===================== Symbols =====================
def is_excluded_symbol(symbol: str) -> bool:
    if not symbol.endswith("USDT"):
        return True
    if any(x in symbol for x in ("UP", "DOWN", "BULL", "BEAR")):
        return True
    if symbol in (
        "USDCUSDT", "FDUSDUSDT", "TUSDUSDT", "EURUSDT",
        "USD1USDT", "RLUSDUSDT", "RWUSDUSDT", "UUSDT"
    ):
        return True
    base = symbol[:-4]
    return base in EXCLUDED_BASES


def get_symbols():
    r = requests.get(f"{API}/ticker/24hr", timeout=30)
    r.raise_for_status()
    syms = []
    for t in r.json():
        s = t["symbol"]
        q_vol = float(t.get("quoteVolume", 0.0))
        if not is_excluded_symbol(s) and q_vol >= MIN_VOLUME:
            syms.append((s, q_vol))
    syms.sort(key=lambda x: x[1], reverse=True)
    if TOP_N:
        syms = syms[:TOP_N]
    return [s for s, _ in syms]


# ===================== Klines =====================
def get_klines(symbol, interval, limit):
    r = requests.get(
        f"{API}/klines",
        params={"symbol": symbol, "interval": interval, "limit": limit},
        timeout=30,
    )
    r.raise_for_status()
    df = pd.DataFrame(
        r.json(),
        columns=[
            "open_time", "open", "high", "low", "close", "volume", "close_time",
            "qav", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
        ],
    )
    for c in ["open", "high", "low", "close", "volume", "taker_buy_base"]:
        df[c] = df[c].astype(float)
    return df


# ===================== 4h Trend Check =====================
def check_htf_trend(symbol: str):
    """
    يفحص اتجاه الفريم الأعلى (4h):
    - السعر أعلى EMA(50) على 4h
    - ميل EMA(50) صاعد (مقارنة بـ 3 شمعات سابقة)
    - عدم كسر أدنى قاع حديث على 4h بإغلاق هابط
    """
    df_htf = get_klines(symbol, HTF_INTERVAL, HTF_LOOKBACK)
    if len(df_htf) < HTF_EMA_PERIOD + 5:
        return False, "بيانات 4h غير كافية"

    ema = df_htf["close"].ewm(span=HTF_EMA_PERIOD, adjust=False).mean()
    htf_close = float(df_htf["close"].iloc[-1])
    ema_now = float(ema.iloc[-1])
    ema_prev = float(ema.iloc[-4])

    above_ema = htf_close > ema_now
    ema_rising = ema_now > ema_prev

    prev_swing_low = float(df_htf["low"].iloc[-(HTF_SWING_LOOKBACK + 1):-1].min())
    no_recent_breakdown = htf_close > prev_swing_low

    is_bullish = above_ema and (ema_rising or no_recent_breakdown)
    status = (
        f"4h {'صاعد' if is_bullish else 'غير صاعد'} "
        f"(Close {htf_close:.6g} vs EMA{HTF_EMA_PERIOD} {ema_now:.6g})"
    )
    return is_bullish, status


# ===================== 1) Anchored VWAP =====================
def anchored_vwap(df, anchor_idx):
    d = df.iloc[anchor_idx:]
    tp = (d["high"] + d["low"] + d["close"]) / 3
    cum_vol = d["volume"].cumsum().replace(0, np.nan)
    return (tp * d["volume"]).cumsum() / cum_vol


def find_swing_low_idx(df, n):
    return int(df["low"].iloc[-n:].idxmin())

# ===================== 2) Volume Profile =====================
def volume_profile(df, bins=50):
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    vols = df["volume"].to_numpy(dtype=float)
    taker_buys = df["taker_buy_base"].to_numpy(dtype=float)

    lo, hi = float(lows.min()), float(highs.max())
    if hi <= lo:
        return lo, lo, lo

    edges = np.linspace(lo, hi, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    left_edges = edges[:-1]
    right_edges = edges[1:]
    vol = np.zeros(bins, dtype=float)

    for h, l, c, v, tb in zip(highs, lows, closes, vols, taker_buys):
        if v <= 0 or h <= l:
            continue
        mask = (left_edges < h) & (right_edges > l)
        if not np.any(mask):
            continue

        overlaps = np.minimum(h, right_edges[mask]) - np.maximum(l, left_edges[mask])
        base_weights = overlaps / (h - l)

        buy_ratio = np.clip(tb / v, 0.0, 1.0)
        span = max(h - l, 1e-9)
        dist_norm = np.abs(centers[mask] - c) / span
        proximity_weight = 0.75 + 0.50 * np.exp(-2.0 * dist_norm)
        bias_factor = 0.85 + 0.30 * buy_ratio

        combined_weights = base_weights * proximity_weight * bias_factor
        w_sum = float(combined_weights.sum())
        if w_sum > 0:
            vol[mask] += v * (combined_weights / w_sum)

    poc_i = int(np.argmax(vol))
    poc = float(centers[poc_i])
    total = float(vol.sum())
    if total <= 0:
        return poc, lo, hi

    target = total * VA_PCT
    acc = float(vol[poc_i])
    i = j = poc_i
    while acc < target and (i > 0 or j < bins - 1):
        left_v = vol[i - 1] if i > 0 else -1.0
        right_v = vol[j + 1] if j < bins - 1 else -1.0
        if left_v >= right_v:
            i -= 1
            acc += float(vol[i])
        else:
            j += 1
            acc += float(vol[j])

    return poc, float(edges[i]), float(edges[j + 1])


# ===================== 3) Order Flow (real) =====================
def get_agg_trades(symbol, start_ms, end_ms, max_pages=AGG_PAGES):
    trades, from_id = [], None
    for _ in range(max_pages):
        params = {"symbol": symbol, "limit": 1000}
        if from_id is None:
            params.update({"startTime": start_ms, "endTime": end_ms})
        else:
            params["fromId"] = from_id
        r = requests.get(f"{API}/aggTrades", params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        trades.extend(data)
        if data[-1]["T"] >= end_ms or len(data) < 1000:
            break
        from_id = data[-1]["a"] + 1
        time.sleep(0.05)
    return [t for t in trades if start_ms <= t["T"] <= end_ms]


def _process_single_candle(symbol, start_ms, end_ms, close_price):
    trades = get_agg_trades(symbol, start_ms, end_ms, max_pages=AGG_PAGES)
    buys, sells, fp_sell = 0.0, 0.0, {}
    for t in trades:
        q, p = float(t["q"]), float(t["p"])
        if t["m"]:
            sells += q
            fp_sell[p] = fp_sell.get(p, 0.0) + q
        else:
            buys += q
    delta = buys - sells
    sell_lvl = None
    if fp_sell:
        worst_price = max(fp_sell, key=fp_sell.get)
        sell_lvl = {
            "price": float(worst_price),
            "sell_qty": float(fp_sell[worst_price]),
            "buy_qty": float(buys),
            "delta": float(delta),
            "close": float(close_price),
        }
    return delta, sell_lvl


def footprint(symbol, df, n=FOOT_CANDLES):
    tasks = []
    for i in range(n, 0, -1):
        row = df.iloc[-i]
        tasks.append((int(row["open_time"]), int(row["close_time"]), float(row["close"])))

    results = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(tasks) or 1)) as pool:
        future_map = {
            pool.submit(_process_single_candle, symbol, st, en, cl): idx
            for idx, (st, en, cl) in enumerate(tasks)
        }
        for fut in as_completed(future_map):
            idx = future_map[fut]
            try:
                results[idx] = fut.result()
            except Exception:
                results[idx] = (0.0, None)

    deltas = [r[0] for r in results if r is not None]
    sell_levels = [r[1] for r in results if r is not None and r[1] is not None]
    return deltas, sell_levels


def _fetch_depth_snapshot(symbol):
    r = requests.get(f"{API}/depth", params={"symbol": symbol, "limit": 100}, timeout=20)
    r.raise_for_status()
    d = r.json()
    bids = [(float(p), float(q)) for p, q in d.get("bids", []) if float(q) > 0]
    asks = [(float(p), float(q)) for p, q in d.get("asks", []) if float(q) > 0]
    return bids, asks


def _extract_bid_wall(bids, mid, band_dist):
    near_bids = [(p, q) for p, q in bids if 0 <= (mid - p) <= band_dist]
    if len(near_bids) < 5:
        return None
    avg_bid = float(np.mean([q for _, q in near_bids]))
    if avg_bid <= 0:
        return None
    top_p, top_q = max(near_bids, key=lambda x: x[1])
    if top_q >= WALL_MULT * avg_bid:
        return {"price": top_p, "qty": top_q, "mult": top_q / avg_bid}
    return None


def analyze_orderbook(symbol, band_pct=OB_BAND_PCT):
    bids1, asks1 = _fetch_depth_snapshot(symbol)
    if not bids1 or not asks1:
        return {
            "imb": 1.0,
            "weighted_imb": 1.0,
            "persistent_wall": None,
            "ephemeral_wall": None,
            "spread_pct": 0.0,
        }

    best_bid, best_ask = bids1[0][0], asks1[0][0]
    mid = (best_bid + best_ask) / 2
    spread_pct = ((best_ask - best_bid) / mid) * 100 if mid > 0 else 0.0
    band_dist = max(mid * (band_pct / 100.0), 1e-12)

    raw_bid_qty = 0.0
    raw_ask_qty = 0.0
    w_bid_qty = 0.0
    w_ask_qty = 0.0

    for p, q in bids1:
        dist = mid - p
        if 0 <= dist <= band_dist:
            raw_bid_qty += q
            weight = 1.0 / (1.0 + (dist / band_dist) * 2.0)
            w_bid_qty += q * weight

    for p, q in asks1:
        dist = p - mid
        if 0 <= dist <= band_dist:
            raw_ask_qty += q
            weight = 1.0 / (1.0 + (dist / band_dist) * 2.0)
            w_ask_qty += q * weight

    imb = (raw_bid_qty / raw_ask_qty) if raw_ask_qty > 0 else 1.0
    weighted_imb = (w_bid_qty / w_ask_qty) if w_ask_qty > 0 else 1.0

    wall1 = _extract_bid_wall(bids1, mid, band_dist)
    persistent_wall = None
    ephemeral_wall = None

    if wall1 is not None:
        time.sleep(WALL_CHECK_DELAY)
        bids2, _ = _fetch_depth_snapshot(symbol)
        price_tol = max(mid * 0.0015, 1e-12)
        matched_q = sum(q for p, q in bids2 if abs(p - wall1["price"]) <= price_tol)
        retention = matched_q / wall1["qty"] if wall1["qty"] > 0 else 0.0

        if retention >= WALL_MIN_RETENTION:
            persistent_wall = {"price": wall1["price"], "mult": wall1["mult"], "retention": retention}
        else:
            ephemeral_wall = {"price": wall1["price"], "mult": wall1["mult"], "retention": retention}

    return {
        "imb": imb,
        "weighted_imb": weighted_imb,
        "persistent_wall": persistent_wall,
        "ephemeral_wall": ephemeral_wall,
        "spread_pct": spread_pct,
    }


# ===================== ATR =====================
def atr(df, period=14):
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    val = tr.rolling(period).mean().iloc[-1]
    return float(val) if pd.notna(val) else 0.0

# ===================== Stage 1: fast filter =====================
def stage1(symbol):
    df = get_klines(symbol, INTERVAL, LOOKBACK)
    anchor = find_swing_low_idx(df, ANCHOR_N)
    df["avwap"] = np.nan
    df.loc[anchor:, "avwap"] = anchored_vwap(df, anchor).values

    price = float(df["close"].iloc[-1])
    avwap = float(df["avwap"].iloc[-1])
    prev_avwap = float(df["avwap"].iloc[-5]) if len(df) >= 5 and pd.notna(df["avwap"].iloc[-5]) else avwap
    slope = avwap - prev_avwap

    poc, val, vah = volume_profile(df)
    cvd = (2 * df["taker_buy_base"] - df["volume"]).cumsum()
    cvd_slope = float(cvd.iloc[-1] - cvd.iloc[-6]) if len(cvd) >= 6 else 0.0
    a = atr(df)
    swing_low = float(df["low"].iloc[-SWING_SL_N:].min())

    score, notes = 0, []
    if pd.notna(avwap) and price > avwap:
        score += 2
        notes.append("فوق AVWAP")
    if slope > 0:
        score += 1
        notes.append("AVWAP صاعد")
    if cvd_slope > 0:
        score += 2
        notes.append("CVD صاعد")
    near = (pd.notna(avwap) and abs(price - avwap) < 1.5 * a) or (abs(price - val) < 1.5 * a)
    if near:
        score += 2
        notes.append("قرب دعم AVWAP/VAL")
    if pd.notna(avwap) and price > avwap and df["low"].iloc[-3:].min() > avwap:
        score += 1
        notes.append("AVWAP صمد كدعم")

    return {
        "df": df, "score": score, "notes": notes,
        "price": price, "avwap": avwap, "poc": poc, "val": val, "vah": vah,
        "atr": a, "swing_low": swing_low,
    }


# ===================== Stage 2: deep order flow =====================
def stage2(symbol, s1):
    score, notes = 0, []
    deltas, sell_levels = footprint(symbol, s1["df"])
    if deltas:
        if sum(deltas) > 0:
            score += 1
            notes.append("دلتا الصفقات موجبة")
        if len(deltas) >= 2 and deltas[-1] > np.mean(deltas[:-1]):
            score += 1
            notes.append("زخم شرائي متزايد")

        if sell_levels:
            last_lvl = sell_levels[-1]
            p_sell = last_lvl["price"]
            q_sell = last_lvl["sell_qty"]
            q_buy = last_lvl["buy_qty"]
            close = last_lvl["close"]
            prev_delta_negative = len(deltas) >= 2 and deltas[-2] < 0
            buy_support = (q_buy >= 0.85 * q_sell) or (deltas[-1] > 0) or prev_delta_negative
            if q_sell > 0 and close > p_sell and buy_support:
                score += 1
                notes.append("امتصاص بيع مؤكد عند الدعم")

    time.sleep(PAUSE)
    ob = analyze_orderbook(symbol)
    eff_imb = 0.6 * ob["weighted_imb"] + 0.4 * ob["imb"]
    if eff_imb >= OB_IMB_MIN:
        score += 1
        notes.append(f"دفتر أوامر شرائي (وزن {eff_imb:.2f} | خام {ob['imb']:.2f})")

    if ob["persistent_wall"] is not None:
        w = ob["persistent_wall"]
        score += 1
        notes.append(f"جدار شراء ثابت عند {w['price']:.6g} ({w['mult']:.1f}x | بقاء {w['retention']*100:.0f}%)")
    elif ob["ephemeral_wall"] is not None:
        ew = ob["ephemeral_wall"]
        notes.append(f"جدار شراء متذبذب/سريع عند {ew['price']:.6g} (بقاء {ew['retention']*100:.0f}%)")

    return score, notes


# ===================== Full analysis =====================
def analyze(symbol):
    s1 = stage1(symbol)
    if s1["score"] < MIN_STAGE1:
        return None

    time.sleep(PAUSE)
    htf_ok, htf_status = check_htf_trend(symbol)
    if REQUIRE_HTF_BULLISH and not htf_ok:
        return None

    time.sleep(PAUSE)
    s2_score, s2_notes = stage2(symbol, s1)
    total = s1["score"] + s2_score
    if total < MIN_TOTAL:
        return None

    entry = s1["price"]
    if entry <= 0:
        return None

    sl_base = min(s1["swing_low"], s1["avwap"]) if pd.notna(s1["avwap"]) else s1["swing_low"]
    sl = sl_base - SL_ATR_MULT * s1["atr"]
    risk = entry - sl
    if risk <= 0:
        return None

    risk_pct = (risk / entry) * 100
    if risk_pct > MAX_RISK_PCT:
        return None

    tp2_default = entry + TP2_RR_MULT * risk
    min_target_price = entry * MIN_TP_BUFFER

    valid_levels = sorted([lvl for lvl in (s1["poc"], s1["vah"]) if lvl >= min_target_price])

    if len(valid_levels) >= 2:
        tp1, tp2 = valid_levels[0], valid_levels[1]
    elif len(valid_levels) == 1:
        tp1 = valid_levels[0]
        tp2 = max(tp2_default, tp1 * 1.01)
    else:
        tp1 = entry + min(1.5 * risk, 0.5 * (tp2_default - entry))
        tp2 = tp2_default

    if tp2 <= tp1:
        tp2 = tp1 + max(0.5 * risk, tp1 * 0.005)

    extra_notes = [htf_status] if htf_ok else []
    return {
        "symbol": symbol, "score": total,
        "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2,
        "risk_pct": risk_pct,
        "rr1": (tp1 - entry) / risk, "rr2": (tp2 - entry) / risk,
        "notes": s1["notes"] + extra_notes + s2_notes,
    }


# ===================== Scan all =====================
def scan_all():
    symbols = get_symbols()
    htf_label = f"مطلوب ({HTF_INTERVAL} صاعد)" if REQUIRE_HTF_BULLISH else "اختياري"
    print(f"🔎 {len(symbols)} زوج مؤهل | فريم {INTERVAL} | إشارة: {MIN_TOTAL}/13 | حد المخاطرة: {MAX_RISK_PCT}% | فلتر {HTF_INTERVAL}: {htf_label}")
    print("=" * 60)

    signals, checked = [], 0
    for i, sym in enumerate(symbols, 1):
        try:
            sig = analyze(sym)
            checked += 1
            if sig:
                signals.append(sig)
                print(f"  [{i}/{len(symbols)}] {sym}: 🟢 إشارة {sig['score']}/13 (مخاطرة {sig['risk_pct']:.2f}%)")
        except requests.exceptions.HTTPError:
            print(f"  [{i}/{len(symbols)}] {sym}: تخطي")
        except Exception:
            print(f"  [{i}/{len(symbols)}] {sym}: ⚠️ تخطي (خطأ مؤقت)")
        time.sleep(PAUSE)

    print(f"\n{'=' * 60}\n✅ تم فحص {checked} | إشارات: {len(signals)}\n{'=' * 60}")
    if not signals:
        print("⚪ مفيش إشارات قوية دلوقتي - استنى.")
        return

    signals.sort(key=lambda s: (s["score"], s["rr1"]), reverse=True)
    for s in signals:
        print(f"\n🟢 شراء {s['symbol']}  (قوة {s['score']}/13)")
        print(f"   الدخول      : {s['entry']:.6g}")
        print(f"   وقف الخسارة : {s['sl']:.6g}  (مخاطرة {s['risk_pct']:.2f}%)")
        print(f"   الهدف 1     : {s['tp1']:.6g}  (R:R = {s['rr1']:.2f})")
        print(f"   الهدف 2     : {s['tp2']:.6g}  (R:R = {s['rr2']:.2f})")
        print(f"   التأكيدات   : {' | '.join(s['notes'])}")
    print("\n⚠️ إشارات آلية للمراجعة - مش نصيحة مالية.")


if __name__ == "__main__":
    scan_all()
