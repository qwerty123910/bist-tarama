"""
Çekirdek Tarama ve Puanlama Motoru
====================================
Bu modül veri çekme, teknik/temel gösterge hesaplama ve puanlama mantığını
içerir. Hem GUI uygulaması (gui_app.py) hem de komut satırından çalıştırmak
isteyenler için kullanılabilir.

Veri kaynağı: Yahoo Finance (yfinance kütüphanesi, API anahtarı gerekmez)
"""

import time
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Callable

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None  # GUI, kütüphane eksikse kullanıcıya net bir hata gösterecek

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("scanner_core")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TICKERS_FILE = os.path.join(BASE_DIR, "bist100_tickers.txt")


# ============================================================
# KONFİGÜRASYON — varsayılanlar. GUI üzerinden bir kısmı değiştirilebilir.
# ============================================================

DEFAULT_TOP_N = 10
DEFAULT_MIN_AVG_VOLUME_TRY = 5_000_000
HISTORY_PERIOD = "1y"
REQUEST_DELAY_SEC = 0.5

WEIGHTS = {
    "pe_ratio":        {"weight": 0.15, "direction": "low"},
    "pb_ratio":        {"weight": 0.10, "direction": "low"},
    "roe":             {"weight": 0.15, "direction": "high"},
    "debt_to_equity":  {"weight": 0.10, "direction": "low"},
    "profit_margin":   {"weight": 0.10, "direction": "high"},
    "momentum_3m":     {"weight": 0.15, "direction": "high"},
    "sma_trend":       {"weight": 0.10, "direction": "high"},
    "rsi_score":       {"weight": 0.10, "direction": "high"},
    "volume_ratio":    {"weight": 0.05, "direction": "high"},
}


# ============================================================
# TICKER LİSTESİ
# ============================================================

def load_tickers(path: str = TICKERS_FILE) -> list[str]:
    """bist100_tickers.txt dosyasından kodları okur, .IS uzantısı ekler."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Ticker dosyası bulunamadı: {path}\n"
            "bist100_tickers.txt dosyasının bu script ile aynı klasörde olduğundan emin olun."
        )
    tickers = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            code = line.upper()
            if not code.endswith(".IS"):
                code += ".IS"
            tickers.append(code)
    return tickers


# ============================================================
# VERİ ÇEKME
# ============================================================

@dataclass
class StockData:
    ticker: str
    hist: Optional[pd.DataFrame] = None
    info: dict = field(default_factory=dict)
    error: Optional[str] = None


def fetch_stock_data(ticker: str) -> StockData:
    data = StockData(ticker=ticker)
    if yf is None:
        data.error = "yfinance kurulu değil"
        return data
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=HISTORY_PERIOD, auto_adjust=True)
        if hist.empty:
            data.error = "Fiyat verisi boş döndü"
            return data
        data.hist = hist
        data.info = t.info or {}
    except Exception as e:
        data.error = str(e)
    return data


def fetch_universe(
    tickers: list[str],
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> dict[str, StockData]:
    """
    progress_callback(current_index, total, ticker) -> GUI ilerleme çubuğunu
    güncellemek için her hisse çekildikten sonra çağrılır.
    """
    results = {}
    total = len(tickers)
    for i, ticker in enumerate(tickers, 1):
        results[ticker] = fetch_stock_data(ticker)
        if progress_callback:
            progress_callback(i, total, ticker)
        time.sleep(REQUEST_DELAY_SEC)
    return results


# ============================================================
# TEKNİK GÖSTERGELER
# ============================================================

def compute_rsi(closes: pd.Series, period: int = 14) -> float:
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not rsi.empty and not np.isnan(rsi.iloc[-1]) else np.nan


def rsi_to_score(rsi: float) -> float:
    if np.isnan(rsi):
        return np.nan
    ideal = 57.5
    return max(0.0, 100 - abs(rsi - ideal) * 2)


def compute_technical_metrics(hist: pd.DataFrame) -> dict:
    closes = hist["Close"]
    volumes = hist["Volume"]

    sma50 = closes.rolling(50).mean().iloc[-1]
    sma200 = closes.rolling(200).mean().iloc[-1] if len(closes) >= 200 else np.nan
    last_close = closes.iloc[-1]

    sma_trend = np.nan
    if not np.isnan(sma50):
        sma_trend = ((last_close / sma50) - 1) * 100
        if not np.isnan(sma200):
            sma_trend += ((last_close / sma200) - 1) * 100

    momentum_3m = np.nan
    if len(closes) >= 63:
        momentum_3m = (last_close / closes.iloc[-63] - 1) * 100

    avg_vol_recent = volumes.tail(20).mean()
    avg_vol_long = volumes.tail(120).mean() if len(volumes) >= 120 else volumes.mean()
    volume_ratio = (avg_vol_recent / avg_vol_long) if avg_vol_long else np.nan

    rsi = compute_rsi(closes)
    avg_liquidity_try = avg_vol_recent * last_close

    return {
        "last_price": last_close,
        "sma_trend": sma_trend,
        "momentum_3m": momentum_3m,
        "volume_ratio": volume_ratio,
        "rsi": rsi,
        "rsi_score": rsi_to_score(rsi),
        "avg_liquidity_try": avg_liquidity_try,
    }


def extract_fundamental_metrics(info: dict) -> dict:
    return {
        "pe_ratio": info.get("trailingPE", np.nan),
        "pb_ratio": info.get("priceToBook", np.nan),
        "roe": (info.get("returnOnEquity") or np.nan) * 100 if info.get("returnOnEquity") else np.nan,
        "debt_to_equity": info.get("debtToEquity", np.nan),
        "profit_margin": (info.get("profitMargins") or np.nan) * 100 if info.get("profitMargins") else np.nan,
    }


# ============================================================
# PUANLAMA
# ============================================================

def build_metrics_table(universe: dict[str, StockData]) -> pd.DataFrame:
    rows = []
    for ticker, data in universe.items():
        if data.error or data.hist is None:
            log.warning(f"{ticker} atlandı: {data.error}")
            continue
        row = {"ticker": ticker}
        row.update(compute_technical_metrics(data.hist))
        row.update(extract_fundamental_metrics(data.info))
        rows.append(row)
    return pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame()


def apply_liquidity_filter(df: pd.DataFrame, min_volume_try: float) -> pd.DataFrame:
    if df.empty:
        return df
    return df[df["avg_liquidity_try"] >= min_volume_try]


def percentile_score(series: pd.Series, direction: str) -> pd.Series:
    ranked = series.rank(pct=True, na_option="keep") * 100
    if direction == "low":
        ranked = 100 - ranked
    return ranked


def compute_final_scores(df: pd.DataFrame, weights: dict = None) -> pd.DataFrame:
    if df.empty:
        return df
    weights = weights or WEIGHTS
    scored = df.copy()

    score_cols = []
    for metric, cfg in weights.items():
        if metric not in scored.columns:
            continue
        col_name = f"score_{metric}"
        scored[col_name] = percentile_score(scored[metric], cfg["direction"])
        score_cols.append((col_name, cfg["weight"]))

    def weighted_row_score(row):
        available = [(c, w) for c, w in score_cols if not pd.isna(row[c])]
        if not available:
            return np.nan
        w_sum = sum(w for _, w in available)
        return sum(row[c] * w for c, w in available) / w_sum

    scored["final_score"] = scored.apply(weighted_row_score, axis=1)
    return scored.sort_values("final_score", ascending=False)


# ============================================================
# TAM TARAMA AKIŞI (GUI ve CLI'nin çağıracağı tek fonksiyon)
# ============================================================

def run_scan(
    top_n: int = DEFAULT_TOP_N,
    min_avg_volume_try: float = DEFAULT_MIN_AVG_VOLUME_TRY,
    tickers: Optional[list[str]] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> pd.DataFrame:
    """Tüm taramayı çalıştırır, puanlanmış tam tabloyu döndürür (en yüksek puan üstte)."""
    if yf is None:
        raise RuntimeError("yfinance kurulu değil. 'pip install -r requirements.txt' çalıştırın.")

    tickers = tickers or load_tickers()
    universe = fetch_universe(tickers, progress_callback=progress_callback)

    df = build_metrics_table(universe)
    if df.empty:
        raise RuntimeError("Hiçbir hisse için veri alınamadı. İnternet bağlantınızı kontrol edin.")

    df = apply_liquidity_filter(df, min_avg_volume_try)
    if df.empty:
        raise RuntimeError(
            "Likidite filtresinden geçen hisse kalmadı. Minimum hacim eşiğini düşürün."
        )

    scored = compute_final_scores(df)
    return scored
