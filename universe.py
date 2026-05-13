"""
Stock universe management — S&P 500, NASDAQ 100, Russell 2000, NYSE+NASDAQ全市場, custom CSV.
"""

import io
import requests
import pandas as pd
import streamlit as st

_NASDAQ_TRADER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def _fetch_nyse_nasdaq() -> tuple[list[str], str]:
    """
    Download complete NYSE + NASDAQ listed stocks from NASDAQ Trader.
    Returns (tickers, status_message). No caching — callers decide.
    """
    nasdaq_url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
    other_url  = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

    tickers: list[str] = []

    try:
        # ── NASDAQ listed (nasdaqlisted.txt) ──────────────────────────────
        # Columns: Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares
        r = requests.get(nasdaq_url, headers=_NASDAQ_TRADER_HEADERS, timeout=15)
        r.raise_for_status()
        for line in r.text.splitlines()[1:]:
            if line.startswith("File Creation Time"):
                continue
            parts = line.split("|")
            if len(parts) < 7:
                continue
            symbol     = parts[0].strip()
            test_issue = parts[3].strip()
            is_etf     = parts[6].strip()
            if test_issue == "Y" or is_etf == "Y":
                continue
            if 1 < len(symbol) <= 5 and symbol.replace("-", "").isalpha():
                tickers.append(symbol)

        # ── NYSE + other exchanges (otherlisted.txt) ──────────────────────
        # Columns: ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol
        r2 = requests.get(other_url, headers=_NASDAQ_TRADER_HEADERS, timeout=15)
        r2.raise_for_status()
        for line in r2.text.splitlines()[1:]:
            if line.startswith("File Creation Time"):
                continue
            parts = line.split("|")
            if len(parts) < 7:
                continue
            symbol     = parts[0].strip()
            exchange   = parts[2].strip()   # N=NYSE, A=NYSE American, Z=BATS
            is_etf     = parts[4].strip()
            test_issue = parts[6].strip()
            if test_issue == "Y" or is_etf == "Y":
                continue
            if exchange not in ("N", "A", "Z"):
                continue
            if 1 < len(symbol) <= 5 and symbol.replace("-", "").isalpha():
                tickers.append(symbol)

        tickers = list(set(tickers))
        if len(tickers) < 1000:
            return [], f"取得的股票數量過少（{len(tickers)} 支），請稍後再試。"
        return tickers, f"成功取得 {len(tickers)} 支 NYSE + NASDAQ 上市股票"

    except Exception as exc:
        return [], f"自動下載失敗（{exc}）"


@st.cache_data(ttl=86400, show_spinner=False)
def get_sp500_tickers() -> list[str]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    return tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()


@st.cache_data(ttl=86400, show_spinner=False)
def get_nasdaq100_tickers() -> list[str]:
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    tables = pd.read_html(url)
    for table in tables:
        for col in ["Ticker", "Symbol", "ticker", "symbol"]:
            if col in table.columns:
                tickers = (
                    table[col]
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .tolist()
                )
                # Sanity-check: must look like tickers
                valid = [t for t in tickers if 1 < len(t) <= 5 and t.replace("-", "").isalpha()]
                if len(valid) > 50:
                    return valid
    return []


@st.cache_data(ttl=86400, show_spinner=False)
def get_russell2000_tickers() -> tuple[list[str], str]:
    """
    Try to download Russell 2000 holdings from iShares IWM.
    Returns (tickers, status_message).
    """
    url = (
        "https://www.ishares.com/us/products/239710/"
        "ishares-russell-2000-etf/1467271812596.ajax"
        "?fileType=csv&fileName=IWM_holdings&dataType=fund"
    )
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Referer": "https://www.ishares.com/",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        # iShares CSV has metadata rows at the top; skip until we find the header
        lines = resp.text.splitlines()
        header_row = next(
            (i for i, line in enumerate(lines) if "Ticker" in line or "ticker" in line),
            9,  # fallback: skip first 9 rows
        )
        df = pd.read_csv(io.StringIO("\n".join(lines[header_row:])), on_bad_lines="skip")
        ticker_col = next(
            (c for c in df.columns if c.strip().lower() in ("ticker", "symbol")), None
        )
        if ticker_col is None:
            return [], "iShares CSV 欄位解析失敗，請改為上傳自訂清單。"
        tickers = (
            df[ticker_col]
            .dropna()
            .astype(str)
            .str.strip()
            .tolist()
        )
        tickers = [t for t in tickers if 1 < len(t) <= 5 and t.replace("-", "").isalpha()]
        if len(tickers) < 100:
            return [], "下載的清單過短，請改為上傳自訂清單。"
        return tickers, f"成功從 iShares 下載 {len(tickers)} 支 Russell 2000 成分股"
    except Exception as exc:
        return [], f"自動下載失敗（{exc}）。請手動上傳 CSV。"


@st.cache_data(ttl=86400, show_spinner=False)
def get_nyse_nasdaq_all_tickers() -> tuple[list[str], str]:
    """Streamlit-cached wrapper around _fetch_nyse_nasdaq()."""
    return _fetch_nyse_nasdaq()


def parse_uploaded_file(uploaded_file) -> list[str]:
    """Extract ticker list from a user-uploaded CSV."""
    df = pd.read_csv(uploaded_file)
    for col in ["Ticker", "Symbol", "ticker", "symbol", "TICKER", "SYMBOL"]:
        if col in df.columns:
            return df[col].dropna().astype(str).str.strip().tolist()
    # Fall back to first column
    return df.iloc[:, 0].dropna().astype(str).str.strip().tolist()
