"""
SEPA trend-template check (Minervini's 7-point trend template):
  1. Price > SMA200
  2. Price > SMA150
  3. Price > SMA50
  4. SMA50 > SMA150 > SMA200  (proper stack)
  5. SMA200 trending up for ≥ 1 month
  6. Price ≥ 75 % of 52-week high  (within 25 % of high)
  7. Price ≥ 125 % of 52-week low  (≥ 25 % above low)
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf


# ─────────────────────────────────────────────
# Fundamental screen
# ─────────────────────────────────────────────

def check_fundamentals(
    ticker: str,
    min_eps_growth: float = 20.0,
    min_rev_growth: float = 15.0,
    min_roe: float = 17.0,
) -> dict:
    """
    Verify four fundamental conditions via yfinance:
      1. Quarterly YoY EPS growth  > min_eps_growth  (%)
      2. Annual revenue growth     > min_rev_growth   (%)
      3. Net profit margin rising  (current year vs prior)
      4. ROE                       > min_roe           (%)

    Returns:
        {
          "pass": bool,
          "eps_growth":    float | None,   # quarterly YoY %
          "rev_growth":    float | None,   # annual YoY %
          "margin_rising": bool  | None,   # True = margin improved
          "roe":           float | None,   # % (e.g. 18.5)
          "reason":        str,
        }
    """
    result: dict = {
        "pass": False,
        "eps_growth": None,
        "rev_growth": None,
        "margin_rising": None,
        "roe": None,
        "reason": "",
    }

    try:
        tk = yf.Ticker(ticker)
        q_stmt = tk.quarterly_income_stmt
        a_stmt = tk.income_stmt
        info   = tk.info or {}
    except Exception as exc:
        result["reason"] = f"API 錯誤: {exc}"
        return result

    reasons: list[str] = []

    # ── 1. Quarterly EPS YoY growth ──────────────────
    eps_growth: float | None = None
    if q_stmt is not None and not q_stmt.empty and q_stmt.shape[1] >= 5:
        for name in ["Basic EPS", "Diluted EPS"]:
            if name in q_stmt.index:
                row = q_stmt.loc[name].dropna()
                if len(row) >= 5:
                    current  = float(row.iloc[0])
                    year_ago = float(row.iloc[4])
                    if year_ago != 0:
                        eps_growth = (current - year_ago) / abs(year_ago) * 100
                    break

    result["eps_growth"] = round(eps_growth, 1) if eps_growth is not None else None
    eps_ok = eps_growth is not None and eps_growth >= min_eps_growth
    if not eps_ok:
        reasons.append(
            f"季EPS年增率不足 ({eps_growth:.1f}%)" if eps_growth is not None else "缺少季度 EPS 資料"
        )

    # ── 2. Annual revenue growth & 3. Profit margin ──
    rev_growth: float | None = None
    margin_rising: bool | None = None

    if a_stmt is not None and not a_stmt.empty and a_stmt.shape[1] >= 2:
        rev_row = None
        for name in ["Total Revenue", "Revenue"]:
            if name in a_stmt.index:
                rev_row = a_stmt.loc[name].dropna()
                break

        if rev_row is not None and len(rev_row) >= 2:
            curr_rev = float(rev_row.iloc[0])
            prev_rev = float(rev_row.iloc[1])
            if prev_rev > 0:
                rev_growth = (curr_rev - prev_rev) / prev_rev * 100

        ni_row = None
        for name in ["Net Income", "Net Income Common Stockholders"]:
            if name in a_stmt.index:
                ni_row = a_stmt.loc[name].dropna()
                break

        if rev_row is not None and ni_row is not None and len(rev_row) >= 2 and len(ni_row) >= 2:
            curr_rev_f = float(rev_row.iloc[0])
            prev_rev_f = float(rev_row.iloc[1])
            if curr_rev_f != 0 and prev_rev_f != 0:
                curr_margin = float(ni_row.iloc[0]) / curr_rev_f
                prev_margin = float(ni_row.iloc[1]) / prev_rev_f
                margin_rising = curr_margin > prev_margin

    result["rev_growth"]    = round(rev_growth, 1) if rev_growth is not None else None
    result["margin_rising"] = margin_rising

    rev_ok = rev_growth is not None and rev_growth >= min_rev_growth
    if not rev_ok:
        reasons.append(
            f"年度營收增長不足 ({rev_growth:.1f}%)" if rev_growth is not None else "缺少年度營收資料"
        )

    margin_ok = margin_rising is True
    if not margin_ok:
        reasons.append("缺少利潤率資料" if margin_rising is None else "利潤率未上升")

    # ── 4. ROE ───────────────────────────────────────
    roe: float | None = None
    roe_raw = info.get("returnOnEquity")
    if roe_raw is not None:
        roe = round(float(roe_raw) * 100, 1)

    result["roe"] = roe
    roe_ok = roe is not None and roe >= min_roe
    if not roe_ok:
        reasons.append(
            f"ROE 不足 ({roe:.1f}%)" if roe is not None else "缺少 ROE 資料"
        )

    result["pass"]   = eps_ok and rev_ok and margin_ok and roe_ok
    result["reason"] = "；".join(reasons)
    return result


# ─────────────────────────────────────────────
# SEPA
# ─────────────────────────────────────────────

def check_sepa(close: pd.Series) -> dict | None:
    """
    Returns a dict with per-criterion booleans and summary values.
    Returns None if there is insufficient history.
    """
    if len(close) < 210:
        return None

    price = float(close.iloc[-1])
    sma50  = float(close.rolling(50).mean().iloc[-1])
    sma150 = float(close.rolling(150).mean().iloc[-1])
    sma200 = float(close.rolling(200).mean().iloc[-1])

    # SMA200 one month ago (≈21 trading days)
    sma200_1m = float(close.rolling(200).mean().iloc[-22])

    tail252 = close.tail(252)
    high52 = float(tail252.max())
    low52  = float(tail252.min())

    c = {
        "c1_price_above_sma200": price > sma200,
        "c2_price_above_sma150": price > sma150,
        "c3_price_above_sma50":  price > sma50,
        "c4_sma_stack":          (sma50 > sma150) and (sma150 > sma200),
        "c5_sma200_uptrend":     sma200 > sma200_1m,
        "c6_near_52w_high":      price >= high52 * 0.75,
        "c7_above_52w_low":      price >= low52  * 1.25,
    }
    c["sepa_count"] = sum(c.values())
    c["sepa_pass"]  = c["sepa_count"] == 7

    c.update(
        price=price,
        sma50=sma50,
        sma150=sma150,
        sma200=sma200,
        high52=high52,
        low52=low52,
    )
    return c


# ─────────────────────────────────────────────
# RS Rating
# ─────────────────────────────────────────────

def compute_rs_ratings(close_map: dict[str, pd.Series]) -> pd.Series:
    """
    IBD-style RS Rating: weighted composite of 4 quarterly returns,
    ranked as a percentile (1–99) across the universe.
    Weights: 40 % last 3 m, 20 % each of the three prior quarters.
    """
    scores: dict[str, float] = {}

    for ticker, close in close_map.items():
        n = len(close)
        if n < 63:
            continue
        try:
            def _ret(a, b):
                return float(close.iloc[-a] / close.iloc[-min(b, n)] - 1)

            r3  = _ret(1,  63)
            r6  = _ret(min(63,  n), min(126, n))
            r9  = _ret(min(126, n), min(189, n))
            r12 = _ret(min(189, n), min(252, n))
            scores[ticker] = 0.40 * r3 + 0.20 * r6 + 0.20 * r9 + 0.20 * r12
        except Exception:
            pass

    s = pd.Series(scores)
    return (s.rank(pct=True) * 99).round(0)


