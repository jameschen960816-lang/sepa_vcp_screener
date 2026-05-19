"""
monitor.py — 每日自動監控腳本
=================================
流程：
  1. 從 NASDAQ Trader 取得 NYSE + NASDAQ 全部股票（~7000–8000 支）
  2. 批次下載近兩日收盤，篩選出今日漲幅 ≥ 5% 的股票
  3. 對漲幅達標股票下載一年歷史，執行 SEPA 趨勢模板篩選
  4. 對通過 SEPA 的股票查詢基本面（季EPS年增 ≥ 20%、年度營收增 ≥ 15%、利潤率上升、ROE ≥ 17%）
  5. 全部條件通過者發送 Discord 通知

用法：
  python monitor.py

建議在 Windows 工作排程器設定每個美股交易日
美東時間 16:30 (台灣時間隔日 04:30) 收盤後執行一次。

Discord Webhook URL 設定方式（三擇一，優先順序由高至低）：
  1. 環境變數: set DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
  2. 同目錄 config.json: {"discord_webhook_url": "https://..."}
  3. 直接修改本檔案下方的 WEBHOOK_URL 變數
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from screener import check_fundamentals, check_sepa, compute_rs_ratings
from discord_notifier import send_discord_alert
from universe import _fetch_nyse_nasdaq

# ── 設定 ──────────────────────────────────────────────────────────
WEBHOOK_URL      = ""          # ← 若不用環境變數或 config.json，在此填入 Webhook URL
MIN_DAILY_GAIN   = 5.0         # 單日漲幅門檻 (%)
MIN_EPS_GROWTH   = 20.0        # 季度 EPS YoY 增長門檻 (%)
MIN_REV_GROWTH   = 15.0        # 年度營收成長門檻 (%)
MIN_ROE          = 17.0        # ROE 門檻 (%)
SEPA_MIN_PASS    = 7           # SEPA 最少需通過幾項（共 7 項）
MIN_RS           = 70          # 最低 RS 評分
MAX_FROM_HIGH_PCT = 10.0       # 距52週新高最大距離 (%)
BATCH            = 100         # yfinance 批次大小

CONFIG_FILE = Path(__file__).parent / "config.json"


def _load_webhook() -> str:
    """Priority: env var > config.json > module constant."""
    import os
    env = os.environ.get("DISCORD_WEBHOOK_URL", "")
    if env:
        return env
    if CONFIG_FILE.exists():
        try:
            cfg = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            return cfg.get("discord_webhook_url", "") or ""
        except Exception:
            pass
    return WEBHOOK_URL


def _batch_download(tickers: list[str], period: str) -> dict[str, pd.DataFrame]:
    """Download price data for a list of tickers in batches."""
    results: dict[str, pd.DataFrame] = {}
    n = len(tickers)

    for start in range(0, n, BATCH):
        batch = tickers[start: start + BATCH]
        try:
            raw = yf.download(
                batch,
                period=period,
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            for t in batch:
                try:
                    if not isinstance(raw.columns, pd.MultiIndex):
                        df = raw.copy()
                    else:
                        level0 = raw.columns.get_level_values(0).unique().tolist()
                        df = raw[t].copy() if t in level0 else raw.droplevel(1, axis=1).copy()
                    df = df.dropna(subset=["Close"])
                    if len(df) >= 2:
                        results[t] = df
                except Exception:
                    pass
        except Exception:
            pass
        time.sleep(0.1)

    return results


def run() -> None:
    webhook = _load_webhook()
    if not webhook:
        print("[警告] 未設定 Discord Webhook URL，通知將不會發送。")

    # ── Step 1: 取得股票清單 ──────────────────────────────────────
    print("正在取得 NYSE + NASDAQ 股票清單…")
    tickers, msg = _fetch_nyse_nasdaq()
    print(msg)
    if not tickers:
        print("[錯誤] 無法取得股票清單，請確認網路連線。")
        return

    # ── Step 2: 下載近 2 日資料，篩選今日漲幅 ≥ 5% ─────────────
    print(f"正在下載 {len(tickers)} 支股票近兩日資料…（這需要幾分鐘）")
    two_day_data = _batch_download(tickers, period="2d")
    print(f"成功取得 {len(two_day_data)} 支股票的資料")

    gainers: dict[str, dict] = {}
    for t, df in two_day_data.items():
        if len(df) < 2:
            continue
        prev_close = float(df["Close"].iloc[-2])
        today_close = float(df["Close"].iloc[-1])
        if prev_close <= 0:
            continue
        gain_pct = (today_close - prev_close) / prev_close * 100
        if gain_pct >= MIN_DAILY_GAIN:
            gainers[t] = {"price": today_close, "gain_pct": round(gain_pct, 2)}

    print(f"今日漲幅 ≥ {MIN_DAILY_GAIN}% 的股票：{len(gainers)} 支")
    if not gainers:
        print("今日無符合漲幅條件的股票。")
        return

    # ── Step 3: 對漲幅達標股票下載完整歷史 + 執行 SEPA ──────────
    print(f"正在下載 {len(gainers)} 支股票一年歷史資料並執行 SEPA 篩選…")
    full_data = _batch_download(list(gainers.keys()), period="1y")

    close_map = {t: df["Close"] for t, df in full_data.items()}
    rs_ratings = compute_rs_ratings(close_map)

    sepa_pass: list[str] = []
    for t, df in full_data.items():
        sepa = check_sepa(df["Close"])
        if sepa is None:
            continue
        rs = float(rs_ratings.get(t, 0))
        if sepa["sepa_count"] < SEPA_MIN_PASS or rs < MIN_RS:
            continue
        from_high_pct = (sepa["high52"] - sepa["price"]) / sepa["high52"] * 100
        if from_high_pct > MAX_FROM_HIGH_PCT:
            continue
        gainers[t]["rs"] = int(rs)
        gainers[t]["sepa_count"] = sepa["sepa_count"]
        gainers[t]["from_high_pct"] = round(from_high_pct, 1)
        sepa_pass.append(t)

    print(f"通過 SEPA（{SEPA_MIN_PASS}/7，RS ≥ {MIN_RS}，距高點 ≤ {MAX_FROM_HIGH_PCT}%）：{len(sepa_pass)} 支")
    if not sepa_pass:
        print("無股票同時通過漲幅 + SEPA 條件。")
        return

    # ── Step 4: 基本面篩選 ────────────────────────────────────────
    print(f"正在查詢 {len(sepa_pass)} 支股票的基本面資料…")
    final: list[str] = []
    for t in sepa_pass:
        fundamentals = check_fundamentals(
            t,
            min_eps_growth=MIN_EPS_GROWTH,
            min_rev_growth=MIN_REV_GROWTH,
            min_roe=MIN_ROE,
        )
        if fundamentals["pass"]:
            gainers[t]["eps_growth"]    = fundamentals["eps_growth"]
            gainers[t]["rev_growth"]    = fundamentals["rev_growth"]
            gainers[t]["margin_rising"] = fundamentals["margin_rising"]
            gainers[t]["roe"]           = fundamentals["roe"]
            final.append(t)
            margin_str = "↑" if fundamentals["margin_rising"] else "↓"
            print(f"  ✓ {t}  漲幅={gainers[t]['gain_pct']:.1f}%  距高點={gainers[t]['from_high_pct']:.1f}%  季EPS年增={fundamentals['eps_growth']}%  年營收增={fundamentals['rev_growth']}%  利潤率={margin_str}  ROE={fundamentals['roe']}%")
        else:
            print(f"  ✗ {t}  基本面未達標：{fundamentals['reason']}")
        time.sleep(0.3)   # avoid rate-limiting

    print(f"\n最終通過所有條件：{len(final)} 支股票")

    # ── Step 5: 發送 Discord 通知 ──────────────────────────────────
    if not final:
        print("今日無股票同時通過漲幅 + SEPA + 基本面條件。")
        return

    sent = 0
    for t in final:
        g = gainers[t]
        ok = send_discord_alert(
            webhook_url=webhook,
            ticker=t,
            gain_pct=g["gain_pct"],
            price=g["price"],
            eps_growth=g.get("eps_growth"),
            rev_growth=g.get("rev_growth"),
            from_high_pct=g.get("from_high_pct"),
        )
        status = "已發送" if ok else "發送失敗（請確認 Webhook URL）"
        print(f"  Discord 通知 {t}: {status}")
        if ok:
            sent += 1

    print(f"\n完成！共發送 {sent}/{len(final)} 則通知。")


if __name__ == "__main__":
    run()
