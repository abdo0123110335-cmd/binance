"""
Complete Spot Scanner V5 — Binance Spot, BUY signals only.
موديل متوازن تكيفي + Order Flow حقيقي (aggTrades) + تأكيد 15m.
Analytical scanner only — NO order execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
import json
import math
import time

import numpy as np
import pdlite as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ============================== CONFIG ==============================

API = "https://api.binance.com/api/v3"

QUOTE_ASSET = "USDT"
INTERVAL_MAIN = "1h"
INTERVAL_CONFIRM = "15m"
LOOKBACK = 300
MIN_24H_QUOTE_VOLUME = 10_000_000
TOP_N = 0                    # 0 = scan all eligible pairs
MAX_SYMBOLS = 0              # optional hard cap; 0 = no cap

# Stage 1 and final thresholds (0..100)
MIN_STAGE1_SCORE = 12
MIN_FINAL_SCORE = 65
MIN_CONFIRM_SCORE = 3
MIN_FLOW_SCORE = 7
STRONG_SCORE = 76

# Risk / targets
MAX_RISK_PCT = 7.0
ATR_PERIOD = 14
SWING_LOOKBACK = 20
SL_ATR_BUFFER = 0.35
TP1_RR_MIN = 1.0
TP2_RR_TARGET = 2.5
MIN_TARGET_BUFFER_PCT = 0.20

# Volume Profile
VP_BINS = 60
VALUE_AREA_PCT = 0.70

# Order flow
FLOW_CANDLES = 3
AGG_TRADES_MAX_PAGES = 10
FLOW_MIN_TRADES = 20
MIN_DELTA_ABS_PCT = 0.15

# Order book
DEPTH_LIMIT = 100
OB_BAND_PCT = 1.0
OB_IMBALANCE_MIN = 1.15
WALL_MULTIPLIER = 4.0
WALL_SNAPSHOT_DELAY = 1.0
WALL_MIN_RETENTION = 0.65
WALL_PRICE_TOLERANCE_PCT = 0.15
MAX_SPREAD_PCT = 0.25

# Runtime
REQUEST_TIMEOUT = 20
REQUEST_PAUSE = 0.08
MAX_WORKERS = 5
LOG_FILE = Path("signals_v5.jsonl")
PRINT_REJECTED = True
SHOW_ERROR_DETAILS = True

REJECTION_COUNTS = {
    "stage1_score": 0, "wide_spread": 0, "no_flow_confirmation": 0,
    "insufficient_flow_data": 0, "weak_15m_confirmation": 0,
    "weak_flow_score": 0, "final_score": 0, "invalid_trade_plan": 0,
    "insufficient_candles": 0,
}
STAGE1_SCORES: list[int] = []


class InsufficientCandlesError(ValueError):
    """The exchange returned fewer closed candles than this analysis needs."""


EXCLUDED_BASES = {
    "USDT", "USDC", "FDUSD", "TUSD", "USDP", "DAI", "USD1",
    "RLUSD", "RWUSD", "U", "PYUSD", "EUR", "BUSD", "UST"
}


# ============================== HTTP ==============================

def build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=4, connect=4, read=3, backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=MAX_WORKERS * 2,
                          pool_maxsize=MAX_WORKERS * 2)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": "BinanceSpotScannerV5/1.0"})
    return session


SESSION = build_session()


def api_get(path: str, params: Optional[dict] = None) -> Any:
    response = SESSION.get(f"{API}/{path.lstrip('/')}", params=params,
                           timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


# ============================== HELPERS ==============================

def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def is_excluded_symbol(symbol: str) -> bool:
    if not symbol.endswith(QUOTE_ASSET):
        return True
    base = symbol[:-len(QUOTE_ASSET)]
    if base in EXCLUDED_BASES:
        return True
    if any(tag in base for tag in ("UP", "DOWN", "BULL", "BEAR")):
        return True
    return False


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def fmt_price(value: float) -> str:
    return f"{value:.8g}"


# ============================== MARKET UNIVERSE ==============================

def get_symbols() -> list[str]:
    tickers = api_get("ticker/24hr")
    selected = []
    for ticker in tickers:
        symbol = ticker.get("symbol", "")
        if is_excluded_symbol(symbol):
            continue
        quote_volume = safe_float(ticker.get("quoteVolume"))
        if quote_volume >= MIN_24H_QUOTE_VOLUME:
            selected.append((symbol, quote_volume))
    selected.sort(key=lambda item: item[1], reverse=True)
    if TOP_N > 0:
        selected = selected[:TOP_N]
    if MAX_SYMBOLS > 0:
        selected = selected[:MAX_SYMBOLS]
    return [symbol for symbol, _ in selected]


# ============================== KLINES ==============================

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]


def get_klines(symbol: str, interval: str, limit: int = LOOKBACK) -> pd.DataFrame:
    raw = api_get("klines", {"symbol": symbol, "interval": interval, "limit": limit})
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"Invalid/empty klines response for {symbol} {interval}: {str(raw)[:240]}")
    if not isinstance(raw[0], (list, tuple)) or len(raw[0]) < len(KLINE_COLUMNS):
        raise ValueError(f"Malformed kline row for {symbol} {interval}: {str(raw[0])[:240]}")

    df = pd.DataFrame(raw, columns=KLINE_COLUMNS)
    numeric = ["open_time", "open", "high", "low", "close", "volume", "close_time",
               "quote_volume", "trades", "taker_buy_base", "taker_buy_quote"]
    for column in numeric:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["open_time", "high", "low", "close", "volume"])
    # Exclude the currently-forming candle. Binance close_time is in milliseconds.
    current_ms = int(time.time() * 1000)
    df = df[df["close_time"] < current_ms].copy()
    df.reset_index(drop=True, inplace=True)

    minimum_candles = max(ATR_PERIOD + 5, 60)
    if len(df) < minimum_candles:
        raise InsufficientCandlesError(
            f"Insufficient closed candles for {symbol} {interval}: "
            f"received={len(df)}, required={minimum_candles}"
        )
    return df

# ============================== INDICATORS ==============================

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    result = result.where(avg_loss != 0, 100.0)
    result = result.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
    return result


def macd(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    line = ema(series, 12) - ema(series, 26)
    signal = line.ewm(span=9, adjust=False, min_periods=9).mean()
    return line, signal, line - signal


def atr_series(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ema20"] = ema(out["close"], 20)
    out["ema50"] = ema(out["close"], 50)
    out["ema200"] = ema(out["close"], 200)
    out["rsi14"] = rsi(out["close"], 14)
    out["macd"], out["macd_signal"], out["macd_hist"] = macd(out["close"])
    out["atr"] = atr_series(out)
    return out


# ============================== ANCHORED VWAP ==============================

def anchored_vwap(df: pd.DataFrame, lookback: int = 60) -> tuple[float, float]:
    window = df.tail(lookback).copy()
    anchor_idx = int(window["low"].idxmin())
    anchored = df.loc[anchor_idx:]
    typical = (anchored["high"] + anchored["low"] + anchored["close"]) / 3.0
    volume = anchored["volume"].clip(lower=0)
    total_volume = float(volume.sum())
    if total_volume <= 0:
        return float("nan"), float("nan")
    avwap_series = (typical * volume).cumsum() / volume.cumsum().replace(0, np.nan)
    avwap = safe_float(avwap_series.iloc[-1], float("nan"))
    prev = safe_float(avwap_series.iloc[-5], avwap) if len(avwap_series) >= 5 else avwap
    return avwap, avwap - prev


# ============================== VOLUME PROFILE ==============================

@dataclass
class VolumeProfile:
    poc: float
    val: float
    vah: float
    hvn: list[float]
    lvn: list[float]
    approximate: bool = True


def volume_profile(df: pd.DataFrame, bins: int = VP_BINS) -> VolumeProfile:
    highs = df["high"].to_numpy(float)
    lows = df["low"].to_numpy(float)
    closes = df["close"].to_numpy(float)
    volumes = df["volume"].to_numpy(float)
    taker_buys = df["taker_buy_base"].to_numpy(float)

    low_bound = float(np.min(lows))
    high_bound = float(np.max(highs))
    if not np.isfinite(low_bound + high_bound) or high_bound <= low_bound:
        return VolumeProfile(low_bound, low_bound, high_bound, [], [])

    edges = np.linspace(low_bound, high_bound, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    profile = np.zeros(bins, dtype=float)

    for high, low, close, volume, taker_buy in zip(highs, lows, closes, volumes, taker_buys):
        if volume <= 0 or high <= low:
            continue
        mask = (edges[:-1] < high) & (edges[1:] > low)
        indexes = np.flatnonzero(mask)
        if len(indexes) == 0:
            continue
        overlap = np.minimum(high, edges[indexes + 1]) - np.maximum(low, edges[indexes])
        weights = overlap / (high - low)
        proximity = np.abs(centers[indexes] - close) / max(high - low, 1e-12)
        proximity_weight = 0.8 + 0.4 * np.exp(-2.0 * proximity)
        buy_ratio = np.clip(taker_buy / volume, 0.0, 1.0)
        directional_weight = 0.9 + 0.2 * buy_ratio
        weights = weights * proximity_weight * directional_weight
        weight_sum = float(weights.sum())
        if weight_sum > 0:
            profile[indexes] += volume * weights / weight_sum

    total = float(profile.sum())
    if total <= 0:
        return VolumeProfile(float(centers[0]), low_bound, high_bound, [], [])

    poc_idx = int(np.argmax(profile))
    left = right = poc_idx
    accumulated = float(profile[poc_idx])
    target = total * VALUE_AREA_PCT

    while accumulated < target and (left > 0 or right < bins - 1):
        left_volume = profile[left - 1] if left > 0 else -1.0
        right_volume = profile[right + 1] if right < bins - 1 else -1.0
        if left_volume >= right_volume:
            left -= 1
            accumulated += float(profile[left])
        else:
            right += 1
            accumulated += float(profile[right])

    hvn_idx = [
        i for i in range(1, bins - 1)
        if profile[i] >= profile[i - 1] and profile[i] >= profile[i + 1]
        and profile[i] >= np.quantile(profile, 0.75)
    ]
    lvn_idx = [
        i for i in range(1, bins - 1)
        if profile[i] <= profile[i - 1] and profile[i] <= profile[i + 1]
        and profile[i] <= np.quantile(profile, 0.25)
    ]

    return VolumeProfile(
        poc=float(centers[poc_idx]),
        val=float(edges[left]),
        vah=float(edges[right + 1]),
        hvn=[float(centers[i]) for i in hvn_idx[:5]],
        lvn=[float(centers[i]) for i in lvn_idx[:5]],
    )


# ============================== AGGREGATE TRADES / DELTA ==============================

def get_agg_trades(symbol: str, start_ms: int, end_ms: int, max_pages: int = AGG_TRADES_MAX_PAGES) -> tuple[list[dict], bool]:
    trades: list[dict] = []
    from_id: Optional[int] = None
    capped = False
    for page in range(max_pages):
        params: dict[str, Any] = {"symbol": symbol, "limit": 1000}
        if from_id is None:
            params.update({"startTime": start_ms, "endTime": end_ms})
        else:
            params["fromId"] = from_id
        data = api_get("aggTrades", params)
        if not data:
            break
        trades.extend(data)
        last = data[-1]
        if int(last["T"]) >= end_ms or len(data) < 1000:
            break
        from_id = int(last["a"]) + 1
        if page == max_pages - 1:
            capped = True
        time.sleep(0.03)
    filtered = [trade for trade in trades if start_ms <= int(trade["T"]) <= end_ms]
    return filtered, capped

def candle_trade_flow(symbol: str, row: pd.Series) -> dict:
    trades, capped = get_agg_trades(symbol, int(row["open_time"]), int(row["close_time"]))
    buy_qty = 0.0
    sell_qty = 0.0
    buy_quote = 0.0
    sell_quote = 0.0
    price_volume: dict[float, dict[str, float]] = {}

    for trade in trades:
        price = safe_float(trade.get("p"))
        qty = safe_float(trade.get("q"))
        if price <= 0 or qty <= 0:
            continue
        quote = price * qty
        # Binance aggTrades: m=True means buyer is maker => aggressive seller.
        if bool(trade.get("m")):
            sell_qty += qty
            sell_quote += quote
            bucket = price_volume.setdefault(price, {"buy": 0.0, "sell": 0.0})
            bucket["sell"] += qty
        else:
            buy_qty += qty
            buy_quote += quote
            bucket = price_volume.setdefault(price, {"buy": 0.0, "sell": 0.0})
            bucket["buy"] += qty

    delta = buy_qty - sell_qty
    total = buy_qty + sell_qty
    dominant_sell_level = None
    if price_volume:
        price, values = max(price_volume.items(), key=lambda item: item[1]["sell"])
        if values["sell"] > 0:
            dominant_sell_level = {
                "price": price,
                "sell_qty": values["sell"],
                "buy_qty": values["buy"],
            }

    return {
        "buy_qty": buy_qty, "sell_qty": sell_qty,
        "buy_quote": buy_quote, "sell_quote": sell_quote,
        "delta": delta,
        "delta_pct": (delta / total * 100.0) if total > 0 else 0.0,
        "trade_count": len(trades),
        "capped": capped,
        "dominant_sell_level": dominant_sell_level,
    }


def analyze_trade_flow(symbol: str, df: pd.DataFrame) -> dict:
    rows = [row for _, row in df.tail(FLOW_CANDLES).iterrows()]
    flows = []
    for row in rows:
        try:
            flows.append(candle_trade_flow(symbol, row))
        except requests.RequestException:
            flows.append({
                "delta": 0.0, "buy_qty": 0.0, "sell_qty": 0.0,
                "trade_count": 0, "capped": True,
                "dominant_sell_level": None,
            })

    deltas = [safe_float(flow.get("delta")) for flow in flows]
    total_delta = float(sum(deltas))
    total_volume = sum(safe_float(flow.get("buy_qty")) + safe_float(flow.get("sell_qty")) for flow in flows)
    last = flows[-1] if flows else {}
    divergence = False

    if len(flows) >= 3 and len(df) >= 4:
        recent_price_change = float(df["close"].iloc[-1] - df["close"].iloc[-4])
        recent_delta = float(sum(deltas[-3:]))
        divergence = recent_delta > 0 and recent_price_change <= 0

    return {
        "flows": flows,
        "total_delta": total_delta,
        "delta_pct": (total_delta / total_volume * 100.0) if total_volume > 0 else 0.0,
        "last_delta": safe_float(last.get("delta")),
        "delta_rising": len(deltas) >= 2 and deltas[-1] > deltas[-2],
        "positive_delta": total_delta > 0,
        "bullish_delta_divergence": divergence,
        "incomplete_sample": any(bool(flow.get("capped")) for flow in flows),
        "trade_count": sum(int(flow.get("trade_count", 0)) for flow in flows),
    }


# ============================== ORDER BOOK ==============================

def fetch_depth(symbol: str) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    data = api_get("depth", {"symbol": symbol, "limit": DEPTH_LIMIT})
    bids = [(safe_float(p), safe_float(q)) for p, q in data.get("bids", [])]
    asks = [(safe_float(p), safe_float(q)) for p, q in data.get("asks", [])]
    bids = [(p, q) for p, q in bids if p > 0 and q > 0]
    asks = [(p, q) for p, q in asks if p > 0 and q > 0]
    bids.sort(key=lambda x: x[0], reverse=True)
    asks.sort(key=lambda x: x[0])
    return bids, asks


def extract_bid_wall(bids: list[tuple[float, float]], mid: float, band_distance: float) -> Optional[dict]:
    nearby = [(price, qty) for price, qty in bids if 0 <= mid - price <= band_distance]
    if len(nearby) < 5:
        return None
    quantities = np.array([qty for _, qty in nearby], dtype=float)
    avg_qty = float(np.mean(quantities))
    if avg_qty <= 0:
        return None
    price, qty = max(nearby, key=lambda item: item[1])
    multiple = qty / avg_qty
    if multiple < WALL_MULTIPLIER:
        return None
    return {"price": price, "qty": qty, "multiple": multiple}


def analyze_orderbook(symbol: str) -> dict:
    bids1, asks1 = fetch_depth(symbol)
    if not bids1 or not asks1:
        return {
            "imbalance": 1.0, "weighted_imbalance": 1.0, "spread_pct": 999.0,
            "persistent_wall": None, "ephemeral_wall": None,
            "bid_quote": 0.0, "ask_quote": 0.0,
        }

    best_bid, best_ask = bids1[0][0], asks1[0][0]
    mid = (best_bid + best_ask) / 2.0
    spread_pct = ((best_ask - best_bid) / mid * 100.0) if mid > 0 else 999.0
    band_distance = mid * OB_BAND_PCT / 100.0

    bid_quote = 0.0
    ask_quote = 0.0
    weighted_bid = 0.0
    weighted_ask = 0.0

    for price, qty in bids1:
        distance = mid - price
        if 0 <= distance <= band_distance:
            quote = price * qty
            weight = 1.0 / (1.0 + 2.0 * distance / max(band_distance, 1e-12))
            bid_quote += quote
            weighted_bid += quote * weight

    for price, qty in asks1:
        distance = price - mid
        if 0 <= distance <= band_distance:
            quote = price * qty
            weight = 1.0 / (1.0 + 2.0 * distance / max(band_distance, 1e-12))
            ask_quote += quote
            weighted_ask += quote * weight

    imbalance = bid_quote / ask_quote if ask_quote > 0 else 1.0
    weighted_imbalance = weighted_bid / weighted_ask if weighted_ask > 0 else 1.0

    wall1 = extract_bid_wall(bids1, mid, band_distance)
    persistent_wall = None
    ephemeral_wall = None

    if wall1:
        time.sleep(WALL_SNAPSHOT_DELAY)
        bids2, _ = fetch_depth(symbol)
        tolerance = mid * WALL_PRICE_TOLERANCE_PCT / 100.0
        matched_qty = sum(qty for price, qty in bids2 if abs(price - wall1["price"]) <= tolerance)
        retention = matched_qty / wall1["qty"] if wall1["qty"] > 0 else 0.0
        wall_info = {"price": wall1["price"], "multiple": wall1["multiple"], "retention": retention}
        if retention >= WALL_MIN_RETENTION:
            persistent_wall = wall_info
        else:
            ephemeral_wall = wall_info

    return {
        "imbalance": imbalance,
        "weighted_imbalance": weighted_imbalance,
        "spread_pct": spread_pct,
        "persistent_wall": persistent_wall,
        "ephemeral_wall": ephemeral_wall,
        "bid_quote": bid_quote,
        "ask_quote": ask_quote,
    }


# ============================== LIQUIDITY / STRUCTURE ==============================

def detect_liquidity_sweep(df: pd.DataFrame, lookback: int = 20) -> dict:
    if len(df) < lookback + 2:
        return {"bullish_sweep": False, "swept_low": None}
    prior = df.iloc[-(lookback + 1):-1]
    last = df.iloc[-1]
    prior_low = float(prior["low"].min())
    swept = float(last["low"]) < prior_low and float(last["close"]) > prior_low
    return {"bullish_sweep": bool(swept), "swept_low": prior_low if swept else None}


def detect_absorption(flow: dict, df: pd.DataFrame) -> bool:
    flows = flow.get("flows", [])
    if not flows or len(df) < 3:
        return False
    last_flow = flows[-1]
    last = df.iloc[-1]
    prior_low = float(df["low"].iloc[-3:-1].min())
    candle_range = max(float(last["high"] - last["low"]), 1e-12)
    close_position = (float(last["close"]) - float(last["low"])) / candle_range
    net_selling = safe_float(last_flow.get("delta")) < 0
    held_support = float(last["low"]) >= prior_low * 0.998
    return bool(net_selling and close_position >= 0.60 and held_support)

# ============================== SCORING ==============================

def score_main_timeframe(df: pd.DataFrame) -> tuple[int, list[str], dict]:
    """Balanced 1h score (0..50): graded evidence instead of all-or-nothing gates."""
    df = add_indicators(df)
    last = df.iloc[-1]
    prev = df.iloc[-2]
    price = float(last["close"])
    avwap, avwap_slope = anchored_vwap(df, 60)
    vp = volume_profile(df)
    atr_value = safe_float(last["atr"])
    notes: list[str] = []
    score = 0

    ema20 = safe_float(last["ema20"])
    ema50 = safe_float(last["ema50"])
    ema200 = safe_float(last["ema200"])
    prev_ema20 = safe_float(prev["ema20"])
    rsi_value = safe_float(last["rsi14"], 50.0)
    macd_now = safe_float(last["macd_hist"])
    macd_prev = safe_float(prev["macd_hist"])

    if price > ema200:
        score += 5; notes.append("السعر فوق EMA200")
    elif price > ema50 and ema50 > ema200 * 0.98:
        score += 2; notes.append("السعر قريب من منطقة الاتجاه المتوسط")
    if ema20 > ema50:
        score += 4; notes.append("EMA20 أعلى EMA50")
    elif ema20 >= ema50 * 0.997:
        score += 2; notes.append("تقارب إيجابي بين EMA20 وEMA50")
    if ema50 > ema200:
        score += 3; notes.append("EMA50 أعلى EMA200")
    elif ema50 >= ema200 * 0.995:
        score += 1
    if ema20 > prev_ema20:
        score += 3; notes.append("EMA20 يميل للصعود")

    if np.isfinite(avwap) and avwap > 0:
        if price > avwap:
            score += 5; notes.append("السعر فوق Anchored VWAP")
        elif price >= avwap - 0.35 * max(atr_value, 1e-12):
            score += 2; notes.append("السعر قريب من AVWAP")
    if np.isfinite(avwap_slope):
        if avwap_slope > 0:
            score += 3; notes.append("ميل AVWAP موجب")
        elif avwap_slope >= -0.10 * max(atr_value, 1e-12):
            score += 1
    if np.isfinite(avwap) and abs(price - avwap) <= 0.75 * max(atr_value, 1e-12):
        score += 2; notes.append("السعر ضمن نطاق تفاعل AVWAP")

    if price > vp.poc:
        score += 5; notes.append("السعر فوق POC")
    elif price >= vp.poc - 0.5 * max(atr_value, 1e-12):
        score += 2; notes.append("السعر قريب من POC")
    if vp.val <= price <= vp.vah:
        score += 4; notes.append("السعر داخل Value Area")
    elif price > vp.vah:
        score += 3; notes.append("السعر أعلى VAH")
    elif price >= vp.val - 0.35 * max(atr_value, 1e-12):
        score += 2; notes.append("السعر قريب من VAL")
    if any(level < price and (price - level) <= 1.5 * max(atr_value, 1e-12) for level in vp.hvn):
        score += 3; notes.append("HVN قريب أسفل السعر")
    if any(level > price and (level - price) <= 1.5 * max(atr_value, 1e-12) for level in vp.lvn):
        score += 3; notes.append("LVN قريب أعلى السعر")

    if 48 <= rsi_value <= 68:
        score += 4; notes.append(f"RSI مناسب ({rsi_value:.1f})")
    elif 42 <= rsi_value < 48 or 68 < rsi_value <= 74:
        score += 2; notes.append(f"RSI مقبول ({rsi_value:.1f})")
    if macd_now > 0:
        score += 3; notes.append("MACD histogram موجب")
    elif macd_now > macd_prev:
        score += 1; notes.append("MACD histogram يتحسن")
    if macd_now > macd_prev:
        score += 2; notes.append("زخم MACD يتحسن")
    if safe_float(last["volume"]) > safe_float(df["volume"].tail(20).mean()):
        score += 1; notes.append("حجم التداول أعلى من متوسط 20 شمعة")

    score = min(score, 50)
    extras = {
        "df": df, "price": price, "avwap": avwap, "avwap_slope": avwap_slope,
        "vp": vp, "atr": atr_value, "rsi": rsi_value,
        "ema20": ema20, "ema50": ema50, "ema200": ema200,
    }
    return score, notes, extras


def score_confirmation(df15: pd.DataFrame) -> tuple[int, list[str]]:
    df = add_indicators(df15)
    last = df.iloc[-1]
    prev = df.iloc[-2]
    score = 0
    notes: list[str] = []
    if float(last["close"]) > float(last["ema20"]):
        score += 4
        notes.append("تأكيد 15m فوق EMA20")
    if float(last["ema20"]) >= float(last["ema50"]):
        score += 2
        notes.append("ترتيب EMA صاعد على 15m")
    if 45 <= safe_float(last["rsi14"], 50) <= 70:
        score += 2
        notes.append("زخم RSI مناسب على 15m")
    if safe_float(last["macd_hist"]) > safe_float(prev["macd_hist"]):
        score += 2
        notes.append("تحسن MACD على 15m")
    return score, notes


def score_order_flow(flow: dict, orderbook: dict, df: pd.DataFrame) -> tuple[int, list[str]]:
    """Order-flow score (0..25), weighted toward actual executed aggressor trades."""
    score = 0
    notes: list[str] = []
    delta_pct = safe_float(flow.get("delta_pct"))

    if flow.get("positive_delta"):
        score += 4
        notes.append("صافي Delta موجب من صفقات Binance المنفذة")
        if delta_pct >= 1.0:
            score += 2
            notes.append(f"ضغط شراء واضح Delta%={delta_pct:.2f}")
    elif delta_pct >= -0.25:
        score += 1
        notes.append("Delta قريب من التعادل")

    if flow.get("delta_rising"):
        score += 3
        notes.append("Delta يتحسن بين الشموع")
    if flow.get("bullish_delta_divergence"):
        score += 3
        notes.append("Bullish Delta Divergence")
    if detect_absorption(flow, df):
        score += 4
        notes.append("اشتباه امتصاص بيع مع ثبات السعر")

    effective_imbalance = (
        0.6 * safe_float(orderbook.get("weighted_imbalance"), 1.0)
        + 0.4 * safe_float(orderbook.get("imbalance"), 1.0)
    )
    if effective_imbalance >= 1.15:
        score += 4
        notes.append(f"Imbalance شرائي {effective_imbalance:.2f}")
    elif effective_imbalance >= 1.0:
        score += 2
        notes.append(f"دفتر الأوامر متوازن/يميل للشراء {effective_imbalance:.2f}")

    if orderbook.get("persistent_wall"):
        score += 3
        wall = orderbook["persistent_wall"]
        notes.append(f"جدار Bid مستمر عند {fmt_price(wall['price'])} (احتفاظ {wall['retention'] * 100:.0f}%)")
    elif orderbook.get("ephemeral_wall"):
        notes.append("جدار Bid لم يحافظ على حجمه؛ لم تُمنح نقاط")

    if safe_float(orderbook.get("spread_pct"), 999) <= MAX_SPREAD_PCT:
        score += 2
        notes.append(f"Spread منخفض {orderbook['spread_pct']:.3f}%")

    if flow.get("trade_count", 0) >= FLOW_MIN_TRADES * 5:
        score += 2
        notes.append("عينة صفقات فعلية جيدة")
    return min(score, 25), notes

# ============================== TRADE PLAN ==============================

def build_trade_plan(symbol, main, score, notes, confirmation_score, flow_score, flow, orderbook, sweep) -> Optional[dict]:
    entry = safe_float(main["price"])
    atr_value = safe_float(main["atr"])
    avwap = safe_float(main["avwap"])
    swing_low = float(main["df"]["low"].tail(SWING_LOOKBACK).min())

    if entry <= 0 or atr_value <= 0:
        return None

    support_candidates = [swing_low]
    if np.isfinite(avwap) and avwap < entry:
        support_candidates.append(avwap)
    if sweep.get("bullish_sweep") and sweep.get("swept_low"):
        support_candidates.append(float(sweep["swept_low"]))

    structural_support = max(support_candidates)
    sl = structural_support - SL_ATR_BUFFER * atr_value
    risk = entry - sl
    if risk <= 0:
        return None

    risk_pct = risk / entry * 100.0
    if risk_pct > MAX_RISK_PCT:
        return None

    min_buffer = entry * (1.0 + MIN_TARGET_BUFFER_PCT / 100.0)
    vp: VolumeProfile = main["vp"]

    overhead = sorted({level for level in [vp.poc, vp.vah, *vp.hvn] if np.isfinite(level) and level >= min_buffer})

    rr_tp1 = max(TP1_RR_MIN, 1.0)
    tp1_rr_fallback = entry + rr_tp1 * risk
    tp1 = overhead[0] if overhead and overhead[0] <= tp1_rr_fallback * 1.8 else tp1_rr_fallback

    tp2_default = entry + TP2_RR_TARGET * risk
    later_levels = [level for level in overhead if level > tp1]
    tp2 = later_levels[0] if later_levels else tp2_default
    tp2 = max(tp2, tp1 + 0.5 * risk)
    tp2 = min(tp2, entry + 4.0 * risk)
    if tp2 <= tp1:
        tp2 = tp1 + max(0.5 * risk, entry * 0.005)

    rr1 = (tp1 - entry) / risk
    rr2 = (tp2 - entry) / risk
    if rr1 <= 0 or rr2 <= rr1:
        return None

    wall = orderbook.get("persistent_wall")
    return {
        "timestamp_utc": now_utc(),
        "symbol": symbol,
        "signal": "BUY",
        "score": int(score),
        "score_breakdown": {
            "main_1h": int(score - confirmation_score - flow_score),
            "confirm_15m": int(confirmation_score),
            "order_flow": int(flow_score),
        },
        "entry": entry,
        "stop_loss": sl,
        "tp1": tp1,
        "tp2": tp2,
        "risk_pct": risk_pct,
        "rr1": rr1,
        "rr2": rr2,
        "price_context": {
            "avwap": avwap,
            "poc": vp.poc,
            "val": vp.val,
            "vah": vp.vah,
            "volume_profile_is_approximate": vp.approximate,
            "atr_1h": atr_value,
            "rsi_1h": main["rsi"],
        },
        "order_flow": {
            "delta": flow["total_delta"],
            "delta_pct": flow["delta_pct"],
            "trades_sampled": flow["trade_count"],
            "sample_incomplete": flow["incomplete_sample"],
            "bid_ask_imbalance": orderbook["imbalance"],
            "weighted_imbalance": orderbook["weighted_imbalance"],
            "spread_pct": orderbook["spread_pct"],
            "persistent_bid_wall": wall,
        },
        "liquidity_sweep": sweep,
        "notes": notes,
    }


# ============================== SYMBOL ANALYSIS ==============================

def adaptive_final_threshold(main_score: int) -> int:
    strength = max(0, main_score - MIN_STAGE1_SCORE)
    return max(58, min(64, int(round(64 - strength * 0.18))))


def log_rejection(symbol: str, stage: str, reason: str, **details) -> None:
    if not PRINT_REJECTED:
        return
    detail_text = " | ".join(f"{key}={value}" for key, value in details.items())
    suffix = f" | {detail_text}" if detail_text else ""
    print(f"   REJECT {symbol} [{stage}] {reason}{suffix}")


def analyze_symbol(symbol: str) -> Optional[dict]:
    df1h = get_klines(symbol, INTERVAL_MAIN, LOOKBACK)
    main_score, main_notes, main = score_main_timeframe(df1h)
    STAGE1_SCORES.append(main_score)
    if main_score < MIN_STAGE1_SCORE:
        REJECTION_COUNTS["stage1_score"] += 1
        log_rejection(symbol, "1H", "score_below_minimum", score=f"{main_score}/50", minimum=MIN_STAGE1_SCORE)
        return None

    time.sleep(REQUEST_PAUSE)

    df15m = get_klines(symbol, INTERVAL_CONFIRM, min(LOOKBACK, 250))
    confirm_score, confirm_notes = score_confirmation(df15m)

    flow = analyze_trade_flow(symbol, main["df"])
    time.sleep(REQUEST_PAUSE)

    orderbook = analyze_orderbook(symbol)
    flow_score, flow_notes = score_order_flow(flow, orderbook, main["df"])

    sweep = detect_liquidity_sweep(main["df"])
    sweep_notes = []
    if sweep["bullish_sweep"]:
        sweep_notes.append(f"Liquidity Sweep ثم استعادة {fmt_price(sweep['swept_low'])}")

    total_score = main_score + confirm_score + flow_score
    all_notes = main_notes + confirm_notes + flow_notes + sweep_notes

    if orderbook["spread_pct"] > MAX_SPREAD_PCT:
        REJECTION_COUNTS["wide_spread"] += 1
        log_rejection(symbol, "ORDER_BOOK", "spread_too_wide", spread_pct=f"{orderbook['spread_pct']:.4f}", maximum_pct=MAX_SPREAD_PCT)
        return None

    if flow["trade_count"] < FLOW_MIN_TRADES:
        REJECTION_COUNTS["insufficient_flow_data"] += 1
        log_rejection(symbol, "TRADE_FLOW", "insufficient_real_aggTrades", trades=flow["trade_count"], required=FLOW_MIN_TRADES, sample_incomplete=flow.get("incomplete_sample"))
        return None

    has_tape_confirmation = (
        (flow["positive_delta"] and flow["delta_pct"] >= MIN_DELTA_ABS_PCT)
        or flow["bullish_delta_divergence"]
        or detect_absorption(flow, main["df"])
        or (flow["delta_rising"] and flow["delta_pct"] >= -0.10)
    )
    if not has_tape_confirmation:
        REJECTION_COUNTS["no_flow_confirmation"] += 1
        log_rejection(symbol, "TRADE_FLOW", "no_tape_confirmation",
                      delta=f"{flow.get('total_delta', 0.0):.8g}",
                      delta_pct=f"{flow.get('delta_pct', 0.0):.3f}",
                      positive_delta=flow.get("positive_delta"),
                      delta_rising=flow.get("delta_rising"),
                      bullish_divergence=flow.get("bullish_delta_divergence"),
                      trades=flow.get("trade_count"))
        return None

    if confirm_score < MIN_CONFIRM_SCORE:
        REJECTION_COUNTS["weak_15m_confirmation"] += 1
        log_rejection(symbol, "15M", "confirmation_score_below_minimum",
                      main_1h=f"{main_score}/50", confirm_15m=f"{confirm_score}/10",
                      required=MIN_CONFIRM_SCORE, flow_score=f"{flow_score}/25",
                      delta=f"{flow.get('total_delta', 0.0):.8g}", trades=flow.get("trade_count"))
        return None

    if flow_score < MIN_FLOW_SCORE:
        REJECTION_COUNTS["weak_flow_score"] += 1
        effective_imbalance = 0.6 * safe_float(orderbook["weighted_imbalance"]) + 0.4 * safe_float(orderbook["imbalance"])
        log_rejection(symbol, "ORDER_FLOW", "flow_score_below_minimum",
                      main_1h=f"{main_score}/50", confirm_15m=f"{confirm_score}/10",
                      flow_score=f"{flow_score}/25", required=MIN_FLOW_SCORE,
                      delta=f"{flow.get('total_delta', 0.0):.8g}",
                      delta_pct=f"{flow.get('delta_pct', 0.0):.3f}",
                      positive_delta=flow.get("positive_delta"),
                      delta_rising=flow.get("delta_rising"),
                      bullish_divergence=flow.get("bullish_delta_divergence"),
                      imbalance=f"{effective_imbalance:.3f}",
                      spread_pct=f"{orderbook['spread_pct']:.4f}",
                      persistent_wall=bool(orderbook.get("persistent_wall")),
                      trades=flow.get("trade_count"))
        return None

    effective_imbalance = 0.6 * safe_float(orderbook["weighted_imbalance"]) + 0.4 * safe_float(orderbook["imbalance"])
    has_book_support = effective_imbalance >= 1.05 or bool(orderbook.get("persistent_wall"))
    if not has_book_support:
        REJECTION_COUNTS["no_flow_confirmation"] += 1
        log_rejection(symbol, "ORDER_BOOK", "no_bid_support",
                      effective_imbalance=f"{effective_imbalance:.3f}",
                      required_imbalance=1.05,
                      persistent_wall=bool(orderbook.get("persistent_wall")),
                      spread_pct=f"{orderbook['spread_pct']:.4f}")
        return None

    required_score = adaptive_final_threshold(main_score)
    if total_score < required_score:
        REJECTION_COUNTS["final_score"] += 1
        log_rejection(symbol, "FINAL", "total_score_below_adaptive_threshold",
                      main_1h=f"{main_score}/50", confirm_15m=f"{confirm_score}/10",
                      order_flow=f"{flow_score}/25", total=f"{total_score}/85", required=required_score)
        return None

    plan = build_trade_plan(symbol, main, total_score, all_notes, confirm_score, flow_score, flow, orderbook, sweep)
    if plan is None:
        REJECTION_COUNTS["invalid_trade_plan"] += 1
        log_rejection(symbol, "TRADE_PLAN", "invalid_entry_sl_tp", entry=main.get("price"), atr=main.get("atr"), main_score=main_score, flow_score=flow_score)
        return None

    plan["score_breakdown"]["main_1h"] = main_score
    plan["score_breakdown"]["confirm_15m"] = confirm_score
    plan["score_breakdown"]["order_flow"] = flow_score
    plan["adaptive_required_score"] = required_score
    plan["order_flow_quality"] = {
        "minimum_trades_required": FLOW_MIN_TRADES,
        "trade_count": flow["trade_count"],
        "has_tape_confirmation": has_tape_confirmation,
        "has_orderbook_support": has_book_support,
    }
    return plan


# ============================== OUTPUT / LOGGING ==============================

def append_signal_log(signal: dict) -> None:
    with LOG_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(signal, ensure_ascii=False) + "\n")


def print_signal(signal: dict) -> None:
    score = signal["score"]
    label = "قوية جداً" if score >= STRONG_SCORE else "مؤهلة"
    print(f"\n🟢 BUY {signal['symbol']} | {score}/85 | {label} | required {signal.get('adaptive_required_score', MIN_FINAL_SCORE)}")
    print(f"   Entry : {fmt_price(signal['entry'])}")
    print(f"   SL    : {fmt_price(signal['stop_loss'])} ({signal['risk_pct']:.2f}%)")
    print(f"   TP1   : {fmt_price(signal['tp1'])} | R:R {signal['rr1']:.2f}")
    print(f"   TP2   : {fmt_price(signal['tp2'])} | R:R {signal['rr2']:.2f}")
    print("   Score : 1H {} | 15m {} | Flow {}".format(
        signal['score_breakdown']['main_1h'],
        signal['score_breakdown']['confirm_15m'],
        signal['score_breakdown']['order_flow'],
    ))
    print("   Confirmations:")
    for note in signal["notes"]:
        print(f"      • {note}")

    flow = signal["order_flow"]
    if flow["sample_incomplete"]:
        print("   ⚠️ عينة الصفقات وصلت حد الصفحات؛ تحليل Order Flow قد يكون غير مكتمل.")
    print("   ℹ️ Volume Profile مبني على OHLCV تقريبي.")


def scan_all() -> None:
    started = time.time()
    for key in REJECTION_COUNTS:
        REJECTION_COUNTS[key] = 0
    STAGE1_SCORES.clear()
    symbols = get_symbols()
    print("=" * 76)
    print("BINANCE SPOT SCANNER V5 | SMART ADAPTIVE BALANCED + REAL ORDER FLOW | BUY ONLY | NO ORDER EXECUTION")
    print(f"Pairs: {len(symbols)} | Main: {INTERVAL_MAIN} | Confirm: {INTERVAL_CONFIRM} | Adaptive final: 58-64/85 | Stage-1 min: {MIN_STAGE1_SCORE}/50 | Min 24h quote volume: {MIN_24H_QUOTE_VOLUME:,.0f} USDT")
    print("=" * 76)

    signals: list[dict] = []
    checked = 0
    errors = 0

    for index, symbol in enumerate(symbols, 1):
        try:
            signal = analyze_symbol(symbol)
            checked += 1
            if signal:
                signals.append(signal)
                print(f"[{index}/{len(symbols)}] {symbol}: BUY score {signal['score']} | risk {signal['risk_pct']:.2f}%")
            elif PRINT_REJECTED:
                print(f"[{index}/{len(symbols)}] {symbol}: rejected (see reason above)")
        except requests.HTTPError as exc:
            errors += 1
            status = exc.response.status_code if exc.response is not None else "HTTP"
            print(f"[{index}/{len(symbols)}] {symbol}: HTTP error {status}")
            if status == 429:
                print("⚠️ Binance rate limit response. Stopping scan to avoid escalation.")
                break
        except InsufficientCandlesError as exc:
            checked += 1
            REJECTION_COUNTS["insufficient_candles"] += 1
            print(f"[{index}/{len(symbols)}] {symbol}: skipped (insufficient history: {exc})")
        except (requests.RequestException, ValueError, KeyError, IndexError) as exc:
            errors += 1
            print(f"[{index}/{len(symbols)}] {symbol}: skipped ({type(exc).__name__}: {exc})")
        except Exception as exc:
            errors += 1
            print(f"[{index}/{len(symbols)}] {symbol}: unexpected error ({type(exc).__name__}: {exc})")

        time.sleep(REQUEST_PAUSE)

    signals.sort(key=lambda item: (item["score"], item["rr2"]), reverse=True)

    print("\n" + "=" * 76)
    print(f"Finished | checked={checked} | signals={len(signals)} | errors={errors}")
    print(f"Elapsed: {time.time() - started:.1f}s")
    print("Rejections: " + " | ".join(f"{key}={value}" for key, value in REJECTION_COUNTS.items()))
    if STAGE1_SCORES:
        scores = np.asarray(STAGE1_SCORES, dtype=float)
        print(f"Stage-1 distribution: min={scores.min():.0f} | median={np.median(scores):.1f} | mean={scores.mean():.1f} | max={scores.max():.0f}")
    print("=" * 76)

    if not signals:
        print("⚪ لا توجد فرص BUY مستوفية للشروط حالياً.")
        return

    for signal in signals:
        print_signal(signal)
        append_signal_log(signal)

    print(f"\n💾 تم حفظ الإشارات في: {LOG_FILE.resolve()}")
    print("\n⚠️ إشارات تحليلية فقط وليست نصيحة مالية. لا يوجد تنفيذ أوامر.")


if __name__ == "__main__":
    scan_all()
