"""
測試 52 週新高篩選邏輯
- 前一交易日收盤價 >= 過去 252 個交易日最高價的 98%
執行：python test_52w_high.py
"""
import yfinance as yf
import pandas as pd

TEST_TICKERS = ["AAPL", "NVDA", "MSFT", "META", "GOOGL", "TSLA", "AMZN", "AVGO"]

def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance 有時回傳 MultiIndex，統一拍平成普通欄位。"""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(1, axis=1)
    return df

def check_52w_high(ticker: str) -> None:
    raw = yf.download(ticker, period="1y", interval="1d", auto_adjust=True, progress=False)
    df = _flatten(raw)
    if df is None or len(df) < 2:
        print(f"{ticker}: 資料不足")
        return

    # 修正後的邏輯
    prev_close    = float(df["Close"].iloc[-2])
    high_window   = df["High"].iloc[max(0, len(df) - 253):-1]
    high_52w      = float(high_window.max())
    threshold     = high_52w * 0.98
    passes        = prev_close >= threshold
    gap_pct       = (prev_close - high_52w) / high_52w * 100

    print(
        f"{ticker:<6} | 前日收盤={prev_close:>8.2f} | 52週最高={high_52w:>8.2f} | "
        f"門檻(98%)={threshold:>8.2f} | 距高點={gap_pct:>+6.2f}% | "
        f"資料列數={len(df)} | {'✅ 通過' if passes else '❌ 未通過'}"
    )

if __name__ == "__main__":
    print("=" * 100)
    print(f"{'Ticker':<6}   {'前日收盤':>8}   {'52週最高':>8}   {'門檻(98%)':>9}   {'距高點':>8}   {'資料列數':>6}   結果")
    print("=" * 100)
    for t in TEST_TICKERS:
        check_52w_high(t)
    print("=" * 100)
    print("說明：距高點% 接近 0 或為正表示股價接近或突破52週新高")
