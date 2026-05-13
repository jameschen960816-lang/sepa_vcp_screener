"""
SEPA trend-template check and VCP pattern detection.

SEPA criteria (Minervini's 7-point trend template):
  1. Price > SMA200
  2. Price > SMA150
  3. Price > SMA50
  4. SMA50 > SMA150 > SMA200  (proper stack)
  5. SMA200 trending up for ≥ 1 month
  6. Price ≥ 75 % of 52-week high  (within 25 % of high)
  7. Price ≥ 125 % of 52-week low  (≥ 25 % above low)

VCP criteria (Volatility Contraction Pattern):
  • 2-4 successive contractions, each smaller than the previous
  • Volume declining across contractions
  • Pivot point = last swing high before final tight area
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf


# ─────────────────────────────────────────────
# Fundamental screen
# ─────────────────────────────────────────────

def check_fundamentals(ticker: str, min_growth_pct: float = 25.0) -> dict:
    """
    Verify quarterly YoY EPS and revenue growth via yfinance.

    Returns:
        {
          "pass": bool,
          "eps_growth": float | None,   # YoY % change, most-recent quarter
          "rev_growth": float | None,   # YoY % change, most-recent quarter
          "reason": str,               # human-readable failure reason
        }
    """
    result: dict = {"pass": False, "eps_growth": None, "rev_growth": None, "reason": ""}

    try:
        stmt = yf.Ticker(ticker).quarterly_income_stmt
    except Exception as exc:
        result["reason"] = f"API 錯誤: {exc}"
        return result

    if stmt is None or stmt.empty or stmt.shape[1] < 5:
        result["reason"] = "季度財報資料不足（需至少 5 季）"
        return result

    def _yoy(row_names: list[str]) -> float | None:
        for name in row_names:
            if name in stmt.index:
                row = stmt.loc[name].dropna()
                if len(row) >= 5:
                    current  = float(row.iloc[0])
                    year_ago = float(row.iloc[4])
                    if year_ago > 0:
                        return (current - year_ago) / year_ago * 100
        return None

    rev_growth = _yoy(["Total Revenue", "Revenue"])
    eps_growth = _yoy(["Basic EPS", "Diluted EPS"])

    result["rev_growth"] = round(rev_growth, 1) if rev_growth is not None else None
    result["eps_growth"] = round(eps_growth, 1) if eps_growth is not None else None

    rev_ok = rev_growth is not None and rev_growth >= min_growth_pct
    eps_ok = eps_growth is not None and eps_growth >= min_growth_pct

    reasons = []
    if not rev_ok:
        reasons.append(
            f"營收增長不足 ({rev_growth:.1f}%)" if rev_growth is not None else "缺少營收資料"
        )
    if not eps_ok:
        reasons.append(
            f"EPS 增長不足 ({eps_growth:.1f}%)" if eps_growth is not None else "缺少 EPS 資料"
        )

    result["pass"]   = rev_ok and eps_ok
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


# ─────────────────────────────────────────────
# VCP helpers
# ─────────────────────────────────────────────

def _find_swing_points(prices: np.ndarray, window: int = 8
                        ) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """Return (swing_highs, swing_lows) as (index, price) pairs."""
    n = len(prices)
    highs, lows = [], []

    for i in range(window, n - window):
        segment = prices[i - window: i + window + 1]
        if prices[i] >= segment.max() - 1e-9:
            if not highs or i - highs[-1][0] > window:
                highs.append((i, float(prices[i])))
        if prices[i] <= segment.min() + 1e-9:
            if not lows or i - lows[-1][0] > window:
                lows.append((i, float(prices[i])))

    return highs, lows


def _interleave(highs, lows) -> list[tuple[int, float, str]]:
    """Merge highs and lows into chronological sequence."""
    seq = [(i, p, "H") for i, p in highs] + [(i, p, "L") for i, p in lows]
    seq.sort(key=lambda x: x[0])

    # Keep only alternating H/L (remove consecutive same-type, keep extremes)
    clean: list[tuple[int, float, str]] = []
    for point in seq:
        if not clean or clean[-1][2] != point[2]:
            clean.append(point)
        else:
            # Same type: keep the more extreme one
            if point[2] == "H" and point[1] > clean[-1][1]:
                clean[-1] = point
            elif point[2] == "L" and point[1] < clean[-1][1]:
                clean[-1] = point
    return clean


# ─────────────────────────────────────────────
# VCP main detector
# ─────────────────────────────────────────────

def detect_vcp(df: pd.DataFrame, min_contractions: int = 2) -> dict:
    """
    Detect Volatility Contraction Pattern.

    A contraction is a swing-high → swing-low leg within the base.
    VCP requires:
      • ≥ min_contractions successive legs
      • Each leg's depth < 80 % of the previous (i.e., tightening)
      • Volume declining across legs (optional flag in caller)

    Returns a dict:
      is_vcp, num_contractions, contraction_depths (list of %),
      volume_declining, pivot_point, tightness_pct
    """
    result = dict(
        is_vcp=False,
        num_contractions=0,
        contraction_depths=[],
        volume_declining=False,
        pivot_point=None,
        tightness_pct=None,
    )

    if df is None or len(df) < 100:
        return result

    # Analyse last ~6 months
    lookback = min(len(df), 126)
    base = df.tail(lookback)
    prices = base["Close"].values.astype(float)
    volumes = base["Volume"].values.astype(float)

    highs, lows = _find_swing_points(prices, window=8)
    if len(highs) < 2 or len(lows) < 1:
        return result

    # Use swings from the last meaningful high onwards
    base_start = highs[-3][0] if len(highs) >= 3 else highs[0][0]
    b_highs = [(i, p) for i, p in highs if i >= base_start]
    b_lows  = [(i, p) for i, p in lows  if i >= base_start]

    if not b_highs or not b_lows:
        return result

    seq = _interleave(b_highs, b_lows)

    # Measure contraction depth for each H→L pair
    contractions: list[dict] = []
    pending_high: tuple | None = None

    for idx, price, ptype in seq:
        if ptype == "H":
            pending_high = (idx, price)
        elif ptype == "L" and pending_high is not None:
            h_idx, h_price = pending_high
            depth_pct = (h_price - price) / h_price * 100
            avg_vol = float(np.mean(volumes[max(0, h_idx - 3): idx + 4]))
            contractions.append(
                dict(high=h_price, low=price, depth=depth_pct, avg_vol=avg_vol)
            )
            pending_high = None

    result["num_contractions"] = len(contractions)
    if len(contractions) < min_contractions:
        return result

    depths = [c["depth"] for c in contractions]
    vols   = [c["avg_vol"] for c in contractions]

    # Each contraction must be ≤ 80 % of the previous
    is_contracting = all(depths[i] <= depths[i - 1] * 0.80 for i in range(1, len(depths)))

    # Volume trend across contractions
    vol_declining = all(vols[i] < vols[i - 1] for i in range(1, len(vols))) if len(vols) > 1 else True

    # Pivot = last swing high in the base
    pivot = float(b_highs[-1][1]) if b_highs else None
    tightness = depths[-1] if depths else None

    result.update(
        is_vcp=is_contracting,
        contraction_depths=depths,
        volume_declining=vol_declining,
        pivot_point=pivot,
        tightness_pct=tightness,
    )
    return result
