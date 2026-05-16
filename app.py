"""
SEPA / VCP 美股篩選器 — Streamlit 主程式
基於 Mark Minervini 的 SEPA 趨勢模板與 VCP 價格模型
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

from screener import check_sepa, check_fundamentals, compute_rs_ratings, detect_vcp
from universe import (
    get_nasdaq100_tickers,
    get_nyse_nasdaq_all_tickers,
    get_russell2000_tickers,
    get_sp500_tickers,
    parse_uploaded_file,
)
from discord_notifier import send_discord_alert

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SEPA / VCP 股票篩選器",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; }
    .stDataFrame thead tr th { background-color: #1e1e2e !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────
# Data download helper
# ─────────────────────────────────────────────

BATCH = 50  # tickers per yfinance call (keep low for Streamlit Cloud thread limits)
CONFIG_FILE = Path(__file__).parent / "config.json"


def _load_webhook_url() -> str:
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text(encoding="utf-8")).get("discord_webhook_url", "")
        except Exception:
            pass
    return ""


def _save_webhook_url(url: str) -> None:
    cfg = {}
    if CONFIG_FILE.exists():
        try:
            cfg = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    cfg["discord_webhook_url"] = url
    CONFIG_FILE.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


def _extract_ticker(raw, ticker: str, n_tickers: int) -> pd.DataFrame | None:
    """
    Safe extraction from yfinance batch download result.

    yfinance 1.3.0 always returns MultiIndex columns:
    - Multi-ticker batch: level 0 = ticker, level 1 = OHLCV
    - Single-ticker batch: level 0 = OHLCV,  level 1 = ticker
    """
    try:
        if not isinstance(raw.columns, pd.MultiIndex):
            # Plain columns — just use as-is
            df = raw.copy()
        else:
            level0 = raw.columns.get_level_values(0).unique().tolist()
            if ticker in level0:
                # Multi-ticker: level 0 = ticker symbols
                df = raw[ticker].copy()
            else:
                # Single-ticker: level 0 = OHLCV names, drop the ticker level
                df = raw.droplevel(1, axis=1).copy()

        df = df.dropna(subset=["Close"])
        return df if len(df) >= 110 else None
    except Exception:
        return None


def download_all(tickers: list[str], status_placeholder) -> dict[str, pd.DataFrame]:
    results: dict[str, pd.DataFrame] = {}
    n = len(tickers)

    for start in range(0, n, BATCH):
        batch = tickers[start: start + BATCH]
        pct = start / n
        status_placeholder.progress(pct, f"下載資料 {start}/{n}…")

        try:
            raw = yf.download(
                batch,
                period="1y",
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            for t in batch:
                df = _extract_ticker(raw, t, len(batch))
                if df is not None:
                    results[t] = df
        except Exception:
            pass

        time.sleep(0.1)  # be polite to Yahoo

    status_placeholder.progress(1.0, f"下載完成，共取得 {len(results)} 支股票資料")
    return results


# ─────────────────────────────────────────────
# Chart
# ─────────────────────────────────────────────

def show_chart(df: pd.DataFrame, ticker: str, rs: float | None, vcp: dict) -> None:
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.72, 0.28],
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"], high=df["High"],
            low=df["Low"],   close=df["Close"],
            name=ticker,
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            showlegend=False,
        ),
        row=1, col=1,
    )

    # Moving averages
    for period, colour, label in [
        (50,  "#2196F3", "SMA 50"),
        (150, "#FF9800", "SMA 150"),
        (200, "#9C27B0", "SMA 200"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Close"].rolling(period).mean(),
                name=label,
                line=dict(color=colour, width=1.5),
            ),
            row=1, col=1,
        )

    # Pivot line
    if vcp.get("pivot_point"):
        pivot = vcp["pivot_point"]
        fig.add_hline(
            y=pivot,
            line=dict(color="#FFD700", width=1.5, dash="dash"),
            annotation_text=f"Pivot {pivot:.2f}",
            annotation_position="top left",
            row=1, col=1,
        )

    # Volume bars
    colors = [
        "#26a69a" if float(df["Close"].iloc[i]) >= float(df["Open"].iloc[i]) else "#ef5350"
        for i in range(len(df))
    ]
    fig.add_trace(
        go.Bar(x=df.index, y=df["Volume"], marker_color=colors, name="Volume", showlegend=False),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Volume"].rolling(50).mean(),
            name="Vol MA50",
            line=dict(color="orange", width=1.2),
        ),
        row=2, col=1,
    )

    title = f"{ticker}"
    if rs is not None:
        title += f"  |  RS Rating: {int(rs)}"
    if vcp.get("num_contractions", 0):
        title += f"  |  VCP 收縮次數: {vcp['num_contractions']}"

    fig.update_layout(
        title=title,
        height=580,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

def render_sidebar() -> dict:
    with st.sidebar:
        st.header("⚙️ 篩選設定")

        st.subheader("📋 股票池")
        universe_choice = st.radio(
            "選擇股票池",
            [
                "NYSE + NASDAQ 全市場",
                "S&P 500",
                "NASDAQ 100",
                "S&P 500 + NASDAQ 100",
                "Russell 2000 (自動下載)",
                "上傳自訂清單",
            ],
            index=0,
        )

        uploaded_file = None
        if universe_choice == "上傳自訂清單":
            st.markdown(
                "上傳含 `Ticker` 欄位的 CSV 檔案。"
                "Russell 2000 清單可至 "
                "[iShares IWM](https://www.ishares.com/us/products/239710/) "
                "頁面下載。"
            )
            uploaded_file = st.file_uploader("選擇 CSV", type=["csv"])

        if universe_choice == "NYSE + NASDAQ 全市場":
            st.info("全市場約 7000+ 支股票，掃描需 30–60 分鐘，建議改用每日監控腳本 monitor.py。")

        st.divider()
        st.subheader("📐 SEPA 趨勢模板")
        use_sepa = st.checkbox("啟用 SEPA 篩選", value=True)
        sepa_min = 7
        min_rs = 70
        if use_sepa:
            sepa_min = st.slider("最少符合幾項條件（共 7 項）", 4, 7, 7)
            min_rs   = st.slider("最低 RS 評分", 0, 99, 70)

        st.divider()
        st.subheader("🌀 VCP 型態識別（日線）")
        use_vcp = st.checkbox("啟用 VCP 篩選", value=True)
        min_contractions = 2
        req_vol = True
        max_below_pivot_pct = 10.0
        if use_vcp:
            min_contractions = st.slider("最少收縮次數", 2, 4, 2)
            req_vol = st.checkbox("要求成交量逐段遞減", value=True)
            max_below_pivot_pct = float(st.slider("距樞軸點最大距離 (%)", 1, 20, 10))
            st.caption("只篩選正在形成 VCP、尚未突破樞軸點的股票")

        st.divider()
        st.subheader("🎯 前一交易日創52週新高（市況觀察）")
        use_near_high = st.checkbox("啟用：前一交易日創52週新高觀察", value=False)
        if use_near_high:
            st.caption("獨立觀察清單，不受其他篩選條件影響，供判斷市場整體強勢狀況")

        st.divider()
        st.subheader("📉 前一交易日創52週新低（市況觀察）")
        use_near_low = st.checkbox("啟用：前一交易日創52週新低觀察", value=False)
        if use_near_low:
            st.caption("獨立觀察清單，不受其他篩選條件影響，供判斷市場整體弱勢狀況")

        st.divider()
        st.subheader("📊 基本面篩選")
        use_fundamentals = st.checkbox("啟用基本面篩選（EPS & 營收 YoY）", value=False)
        min_eps_growth = 25.0
        min_rev_growth = 25.0
        if use_fundamentals:
            min_eps_growth = float(st.slider("季度 EPS 增長門檻 (%)", 10, 100, 25))
            min_rev_growth = float(st.slider("季度營收增長門檻 (%)", 10, 100, 25))
            st.caption("⚠️ 基本面篩選每支股票需額外 API 請求，會大幅增加掃描時間。")

        st.divider()
        st.subheader("🔔 Discord 通知")
        saved_url = _load_webhook_url()
        webhook_input = st.text_input(
            "Webhook URL",
            value=saved_url,
            placeholder="https://discord.com/api/webhooks/…",
            type="password",
        )
        if webhook_input and webhook_input != saved_url:
            _save_webhook_url(webhook_input)

        min_gain_alert = float(st.slider("單日漲幅通知門檻 (%)", 3, 20, 5))
        st.caption(
            "如何取得 Webhook URL：Discord #stock 頻道 → 編輯頻道 → 整合 → Webhooks → 建立 Webhook → 複製連結"
        )

        st.divider()
        run = st.button("🔍 開始掃描", type="primary", use_container_width=True)

    return dict(
        universe_choice=universe_choice,
        uploaded_file=uploaded_file,
        use_sepa=use_sepa,
        sepa_min=sepa_min,
        min_rs=min_rs,
        use_vcp=use_vcp,
        min_contractions=min_contractions,
        req_vol=req_vol,
        max_below_pivot_pct=max_below_pivot_pct,
        use_near_high=use_near_high,
        use_near_low=use_near_low,
        use_fundamentals=use_fundamentals,
        min_eps_growth=min_eps_growth,
        min_rev_growth=min_rev_growth,
        webhook_url=webhook_input.strip() if webhook_input else "",
        min_gain_alert=min_gain_alert,
        run=run,
    )


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main() -> None:
    st.title("📈 SEPA / VCP 美股篩選器")
    st.caption(
        "基於 Mark Minervini 的 **SEPA 趨勢模板** 與 **VCP 價格型態** 篩選潛力成長股"
    )

    cfg = render_sidebar()

    if not cfg["run"]:
        # Landing page info
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("SEPA 趨勢模板（7 項條件）")
            st.markdown(
                """
                1. 股價 > SMA 200
                2. 股價 > SMA 150
                3. 股價 > SMA 50
                4. SMA 50 > SMA 150 > SMA 200（正確排列）
                5. SMA 200 至少上升 1 個月
                6. 股價在 52 週高點的 75 % 以上
                7. 股價在 52 週低點的 125 % 以上（漲幅 ≥ 25 %）
                """
            )
        with col2:
            st.subheader("VCP 型態識別條件")
            st.markdown(
                """
                - 2–4 次連續收縮（每次波動幅度遞減）
                - 每次回調深度 ≤ 前一次的 80 %
                - 成交量在各收縮段逐漸萎縮
                - 樞軸點（Pivot Point）= 最後一個波段高點
                - 目前股價接近最終緊縮區上方
                """
            )
        st.divider()
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("基本面篩選條件")
            st.markdown(
                """
                - 近期季度 EPS（每股盈餘）YoY 增長 > 25%
                - 近期季度總營收 YoY 增長 > 25%
                - 使用 Yahoo Finance 季度損益表計算
                - 基準：同一季度對比去年同期
                """
            )
        with col4:
            st.subheader("52週高低點篩選")
            st.markdown(
                """
                - **創52週新高**：前一交易日盤中高點創下52週新高
                - **創52週新低**：前一交易日盤中低點創下52週新低
                - 新低清單為獨立觀察區，不受 SEPA / VCP 影響
                - 適合判斷強勢突破股與市場整體弱勢狀況
                """
            )
        st.info("設定好篩選條件後，點選左側的「開始掃描」按鈕。")
        return

    # ── Load tickers ──────────────────────────
    status = st.empty()
    status.info("正在載入股票清單…")

    choice = cfg["universe_choice"]
    if choice == "NYSE + NASDAQ 全市場":
        status.info("從 NASDAQ Trader 下載完整市場清單…")
        tickers, msg = get_nyse_nasdaq_all_tickers()
        if tickers:
            st.success(msg)
        else:
            st.warning(msg)
            st.stop()
    elif choice == "S&P 500":
        tickers = get_sp500_tickers()
    elif choice == "NASDAQ 100":
        tickers = get_nasdaq100_tickers()
    elif choice == "S&P 500 + NASDAQ 100":
        tickers = list(set(get_sp500_tickers() + get_nasdaq100_tickers()))
    elif choice == "Russell 2000 (自動下載)":
        status.info("嘗試從 iShares 下載 Russell 2000 成分股…")
        tickers, msg = get_russell2000_tickers()
        if tickers:
            st.success(msg)
        else:
            st.warning(msg)
            st.stop()
    else:  # custom upload
        if cfg["uploaded_file"] is None:
            st.warning("請先上傳 CSV 檔案。")
            st.stop()
        tickers = parse_uploaded_file(cfg["uploaded_file"])

    if not tickers:
        st.error("無法取得股票清單，請確認網路連線或改為上傳自訂清單。")
        st.stop()

    status.success(f"股票池載入完成：共 **{len(tickers)}** 支股票")

    # Warn on large universes
    if len(tickers) > 500:
        st.warning(
            f"股票池有 {len(tickers)} 支股票，掃描可能需要 10–20 分鐘，請耐心等候。"
        )

    # ── Download price data ───────────────────
    prog = st.progress(0.0)
    all_data = download_all(tickers, prog)

    if not all_data:
        st.error("資料下載失敗，請稍後再試。")
        st.stop()

    # ── Compute RS Ratings ────────────────────
    st.info("計算 RS 評分中…")
    close_map = {t: df["Close"] for t, df in all_data.items()}
    rs_ratings = compute_rs_ratings(close_map)

    # ── Screen ────────────────────────────────
    st.info("執行 SEPA / VCP 篩選中…")
    prog2 = st.progress(0.0)
    results: list[dict] = []
    high52_results: list[dict] = []
    low52_results: list[dict] = []
    total = len(all_data)

    for idx, (ticker, df) in enumerate(all_data.items()):
        prog2.progress((idx + 1) / total)

        try:
            close = df["Close"]
            row: dict = {"Ticker": ticker}
            sepa_ok = True
            vcp_ok  = True
            vcp_result: dict = {}

            # ── SEPA ──
            sepa = None
            if cfg["use_sepa"]:
                sepa = check_sepa(close)
                if sepa is None:
                    continue

                rs = float(rs_ratings.get(ticker, 0))
                if sepa["sepa_count"] < cfg["sepa_min"] or rs < cfg["min_rs"]:
                    sepa_ok = False

                from_high_pct = (sepa["high52"] - sepa["price"]) / sepa["high52"] * 100
                row.update(
                    Price        = round(sepa["price"], 2),
                    SMA50        = round(sepa["sma50"], 2),
                    SMA150       = round(sepa["sma150"], 2),
                    SMA200       = round(sepa["sma200"], 2),
                    SEPA條件     = f"{sepa['sepa_count']}/7",
                    RS評分       = int(rs),
                    高52週       = round(sepa["high52"], 2),
                    低52週       = round(sepa["low52"], 2),
                    距高點pct    = round(from_high_pct, 1),
                )

            # ── 前一交易日創52週新高（市況觀察，獨立於其他篩選）──
            if cfg["use_near_high"] and len(df) >= 253:
                prev_high = float(df["High"].iloc[-2])
                high_52w_prev = float(df["High"].tail(253).iloc[:-1].max())
                if prev_high >= high_52w_prev:
                    high52_results.append({
                        "Ticker": ticker,
                        "現價": round(float(close.iloc[-1]), 2),
                        "昨日高點": round(prev_high, 2),
                        "52週最高": round(high_52w_prev, 2),
                    })

            # ── 前一交易日創52週新低（市況觀察，獨立於其他篩選）──
            if cfg["use_near_low"] and len(df) >= 253:
                prev_low_price = float(df["Low"].iloc[-2])
                low_52w_prev = float(df["Low"].tail(253).iloc[:-1].min())
                if prev_low_price <= low_52w_prev:
                    low52_results.append({
                        "Ticker": ticker,
                        "現價": round(float(close.iloc[-1]), 2),
                        "昨日低點": round(prev_low_price, 2),
                        "52週最低": round(low_52w_prev, 2),
                    })

            # ── VCP ──
            if cfg["use_vcp"]:
                vcp_result = detect_vcp(
                    df,
                    min_contractions=cfg["min_contractions"],
                    max_below_pivot_pct=cfg["max_below_pivot_pct"],
                )
                if not vcp_result["pre_breakout"]:
                    vcp_ok = False
                if cfg["req_vol"] and not vcp_result["volume_declining"]:
                    vcp_ok = False

                row.update(
                    VCP收縮次數  = vcp_result["num_contractions"],
                    樞軸點       = round(vcp_result["pivot_point"], 2) if vcp_result["pivot_point"] else None,
                    距樞軸pct    = vcp_result["pct_below_pivot"],
                    最終緊縮幅度 = f"{vcp_result['tightness_pct']:.1f}%" if vcp_result["tightness_pct"] else None,
                    量能遞減     = "✓" if vcp_result["volume_declining"] else "✗",
                )

            if sepa_ok and vcp_ok:
                # Compute today's daily gain from last two closes
                if len(df) >= 2:
                    prev  = float(df["Close"].iloc[-2])
                    today = float(df["Close"].iloc[-1])
                    daily_gain = (today - prev) / prev * 100 if prev > 0 else 0.0
                else:
                    daily_gain = 0.0
                row["今日漲幅%"] = round(daily_gain, 2)
                row["_daily_gain"] = daily_gain

                results.append(row)
                results[-1]["_vcp"] = vcp_result  # keep for chart detail

        except Exception:
            continue

    prog2.progress(1.0)

    # ── 52週新高觀察（獨立區塊，掃描結束後立即呈現）──
    if cfg["use_near_high"]:
        st.divider()
        if high52_results:
            st.subheader(f"🚀 前一交易日創52週新高的股票（共 {len(high52_results)} 支）")
            st.caption("以下股票昨日盤中高點創下近52週新高，不受 SEPA / VCP 篩選條件影響，供觀察市場整體強勢狀況。")
            high_df = pd.DataFrame(high52_results).sort_values("Ticker")
            st.dataframe(high_df, use_container_width=True, hide_index=True)
            ts_high = datetime.now().strftime("%Y%m%d_%H%M")
            st.download_button(
                "⬇️ 下載52週新高清單 CSV",
                high_df.to_csv(index=False),
                f"52w_high_{ts_high}.csv",
                "text/csv",
                key="dl_high",
            )
        else:
            st.info("🚀 前一交易日沒有股票創下52週新高。")

    # ── 52週新低觀察（獨立區塊，掃描結束後立即呈現）──
    if cfg["use_near_low"]:
        st.divider()
        if low52_results:
            st.subheader(f"📉 前一交易日創52週新低的股票（共 {len(low52_results)} 支）")
            st.caption("以下股票昨日盤中低點創下近52週新低，可用於觀察市場整體弱勢狀況。不受 SEPA / VCP 篩選條件影響。")
            low_df = pd.DataFrame(low52_results).sort_values("Ticker")
            st.dataframe(low_df, use_container_width=True, hide_index=True)
            ts_low = datetime.now().strftime("%Y%m%d_%H%M")
            st.download_button(
                "⬇️ 下載52週新低清單 CSV",
                low_df.to_csv(index=False),
                f"52w_low_{ts_low}.csv",
                "text/csv",
                key="dl_low",
            )
        else:
            st.info("📉 前一交易日沒有股票創下52週新低。")

    # ── Fundamental screen (optional, slow) ──
    if cfg["use_fundamentals"] and results:
        st.info(f"執行基本面篩選中（{len(results)} 支股票，每支需額外 API 請求，請耐心等候）…")
        prog3 = st.progress(0.0)
        filtered: list[dict] = []
        for i, row in enumerate(results):
            prog3.progress((i + 1) / len(results))
            fund = check_fundamentals(
                row["Ticker"],
                min_growth_pct=min(cfg["min_eps_growth"], cfg["min_rev_growth"]),
            )
            if fund["pass"]:
                row["EPS增長%"] = fund["eps_growth"]
                row["營收增長%"] = fund["rev_growth"]
                filtered.append(row)
            time.sleep(0.25)
        prog3.progress(1.0)
        results = filtered

    # ── Results ───────────────────────────────
    st.success(f"掃描完成！找到 **{len(results)}** 支符合條件的股票（共掃描 {total} 支）")

    if not results:
        st.warning("沒有找到符合條件的股票，請嘗試放寬篩選條件（例如降低 SEPA 最低項數或 RS 門檻）。")
        return

    # Build display dataframe (drop internal columns, rename display keys)
    _internal = {"_vcp", "_daily_gain"}
    _rename = {"距高點pct": "距52週高%", "距樞軸pct": "距樞軸%"}
    display_rows = [
        {_rename.get(k, k): v for k, v in r.items() if k not in _internal}
        for r in results
    ]
    df_result = pd.DataFrame(display_rows)

    if "RS評分" in df_result.columns:
        df_result = df_result.sort_values("RS評分", ascending=False)

    st.dataframe(df_result, use_container_width=True, hide_index=True)

    # Download CSV
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    st.download_button(
        "⬇️ 下載結果 CSV",
        df_result.to_csv(index=False),
        f"sepa_vcp_{ts}.csv",
        "text/csv",
    )

    # ── Discord 通知：單日漲幅達門檻的股票 ──
    gainers_alert = [
        r for r in results
        if r.get("_daily_gain", 0) >= cfg["min_gain_alert"]
    ]
    if gainers_alert:
        st.divider()
        st.subheader(f"🚀 今日漲幅 ≥ {cfg['min_gain_alert']:.0f}% 的篩選通過股票")
        alert_df = pd.DataFrame([
            {
                "Ticker":   r["Ticker"],
                "今日漲幅%": r.get("_daily_gain", 0),
                "現價":      r.get("Price", ""),
                "EPS增長%":  r.get("EPS增長%", "—"),
                "營收增長%": r.get("營收增長%", "—"),
            }
            for r in gainers_alert
        ])
        st.dataframe(alert_df, use_container_width=True, hide_index=True)

        webhook = cfg["webhook_url"]
        if webhook:
            if st.button("📣 發送 Discord 通知", type="primary"):
                sent = 0
                for r in gainers_alert:
                    ok = send_discord_alert(
                        webhook_url=webhook,
                        ticker=r["Ticker"],
                        gain_pct=r.get("_daily_gain", 0),
                        price=r.get("Price", 0),
                        eps_growth=r.get("EPS增長%"),
                        rev_growth=r.get("營收增長%"),
                        from_high_pct=r.get("距高點pct"),
                    )
                    if ok:
                        sent += 1
                if sent:
                    st.success(f"✅ 已發送 {sent} 則通知到 Discord #stock 頻道！")
                else:
                    st.error("發送失敗，請確認 Webhook URL 是否正確。")
        else:
            st.info("在左側欄填入 Discord Webhook URL 即可一鍵發送通知。")

    # ── Detail chart ─────────────────────────
    st.divider()
    st.subheader("📊 個股詳細圖表")

    ticker_list = df_result["Ticker"].tolist()
    selected = st.selectbox("選擇股票", ticker_list)

    if selected and selected in all_data:
        # Retrieve saved vcp result
        vcp_detail: dict = next(
            (r["_vcp"] for r in results if r["Ticker"] == selected), {}
        )
        rs_val = rs_ratings.get(selected)
        show_chart(all_data[selected], selected, rs_val, vcp_detail)

        # Contraction table
        depths = vcp_detail.get("contraction_depths", [])
        if depths:
            st.markdown("**VCP 收縮幅度明細**")
            contraction_df = pd.DataFrame(
                {"收縮": [f"第 {i+1} 次" for i in range(len(depths))],
                 "回調幅度 (%)": [f"{d:.1f}" for d in depths]}
            )
            st.table(contraction_df)


if __name__ == "__main__":
    main()
