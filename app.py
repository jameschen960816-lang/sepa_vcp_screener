"""
SEPA 美股篩選器 — Streamlit 主程式
基於 Mark Minervini 的 SEPA 趨勢模板
"""

from __future__ import annotations

import time
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

from screener import check_sepa, check_fundamentals, compute_rs_ratings
from universe import (
    get_nasdaq100_tickers,
    get_nyse_nasdaq_all_tickers,
    get_russell2000_tickers,
    get_sp500_tickers,
    parse_uploaded_file,
)

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SEPA 股票篩選器",
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

def show_chart(df: pd.DataFrame, ticker: str, rs: float | None) -> None:
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.72, 0.28],
    )

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

    title = ticker
    if rs is not None:
        title += f"  |  RS Rating: {int(rs)}"

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

        # ── 基本篩選條件（所有股票均適用）───────────
        st.divider()
        st.subheader("📌 基本篩選條件")
        st.caption("適用於所有股票（與其他篩選條件搭配）")
        min_price = float(st.number_input("最低股價（美元）", min_value=0.0, value=8.0, step=0.5))
        min_avg_volume = int(st.number_input("最低日均成交量（股）", min_value=0, value=500000, step=50000))

        # ── 群組一：技術面 & 基本面篩選 ──────────────
        st.divider()
        st.subheader("🔬 群組一：技術面 & 基本面篩選")
        st.caption("可自由勾選，勾選的條件同時套用（AND 邏輯）")

        use_sepa = st.checkbox("📐 啟用 SEPA 篩選", value=True)
        sepa_min = 7
        min_rs = 70
        if use_sepa:
            sepa_min = st.slider("最少符合幾項條件（共 7 項）", 4, 7, 7)
            min_rs   = st.slider("最低 RS 評分", 0, 99, 70)

        use_fundamentals = st.checkbox("📊 啟用基本面篩選（EPS / 營收 / 利潤率 / ROE）", value=False)
        min_eps_growth = 20.0
        min_rev_growth = 15.0
        if use_fundamentals:
            min_eps_growth = float(st.slider("季度 EPS 年增率門檻 (%)", 10, 100, 20))
            min_rev_growth = float(st.slider("年度營收成長門檻 (%)", 10, 100, 15))
            st.caption("⚠️ 基本面篩選每支股票需額外 API 請求，會大幅增加掃描時間。")

        # ── 群組二：52週高低點觀察 ─────────────────
        st.divider()
        st.subheader("📡 群組二：52週高低點觀察")
        st.caption("可自由勾選，獨立於群組一，不受其他篩選條件影響")

        use_near_high = st.checkbox("🚀 前一交易日創52週新高", value=False)
        if use_near_high:
            st.caption("獨立觀察清單，供判斷市場整體強勢狀況")

        use_near_low = st.checkbox("📉 前一交易日創52週新低", value=False)
        if use_near_low:
            st.caption("獨立觀察清單，供判斷市場整體弱勢狀況")

        # ── 群組三：新上市公司篩選 ─────────────────
        st.divider()
        st.subheader("🆕 群組三：新上市公司觀察")
        st.caption("可自由勾選，獨立於群組一，不受其他篩選條件影響")

        use_new_listing = st.checkbox("🆕 上市未滿半年的公司", value=False)
        if use_new_listing:
            st.caption("獨立觀察清單，篩選出近 180 天內首次交易的新股")

        st.divider()
        run = st.button("🔍 開始掃描", type="primary", use_container_width=True)
        reset = st.button("🔄 重新設定", use_container_width=True)

    return dict(
        universe_choice=universe_choice,
        uploaded_file=uploaded_file,
        min_price=min_price,
        min_avg_volume=min_avg_volume,
        use_sepa=use_sepa,
        sepa_min=sepa_min,
        min_rs=min_rs,
        use_near_high=use_near_high,
        use_near_low=use_near_low,
        use_new_listing=use_new_listing,
        use_fundamentals=use_fundamentals,
        min_eps_growth=min_eps_growth,
        min_rev_growth=min_rev_growth,
        run=run,
        reset=reset,
    )


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main() -> None:
    st.title("📈 SEPA 美股篩選器")
    st.caption(
        "基於 Mark Minervini 的 **SEPA 趨勢模板** 篩選潛力成長股"
    )

    # Session state controls whether the scan pipeline runs.
    # st.button() resets to False on every re-run, so we persist the trigger
    # in session_state to avoid restarting mid-scan on incidental re-runs.
    if "do_scan" not in st.session_state:
        st.session_state.do_scan = False

    cfg = render_sidebar()

    if cfg["run"]:
        st.session_state.do_scan = True
    if cfg["reset"]:
        st.session_state.do_scan = False
        st.rerun()

    if not st.session_state.do_scan:
        # Landing page info — renders instantly, zero network calls
        st.subheader("📌 基本篩選條件（所有股票均適用）")
        st.markdown(
            """
            - 股價 ≥ **$8**
            - 日均成交量 ≥ **50 萬股**（近 50 日平均）
            """
        )
        st.divider()
        st.subheader("🔬 群組一：技術面 & 基本面篩選")
        st.caption("兩個條件可自由勾選，勾選的條件同時套用（AND 邏輯）")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**📐 SEPA 趨勢模板（7 項）**")
            st.markdown(
                """
                1. 股價 > SMA 200
                2. 股價 > SMA 150
                3. 股價 > SMA 50
                4. SMA 50 > SMA 150 > SMA 200
                5. SMA 200 至少上升 1 個月
                6. 股價在 52 週高點的 75% 以上
                7. 股價在 52 週低點的 125% 以上
                """
            )
        with col2:
            st.markdown("**📊 基本面篩選**")
            st.markdown(
                """
                - 季度 EPS（每股盈餘）YoY 增長 > 20%
                - 年度總營收增長 > 15%
                - 利潤率逐年上升
                - ROE > 17%
                - ⚠️ 此條件會增加掃描時間
                """
            )
        st.divider()
        st.subheader("📡 群組二：52週高低點觀察")
        st.caption("兩個條件可自由勾選，獨立於群組一，不受其他篩選條件影響")
        col4, col5 = st.columns(2)
        with col4:
            st.markdown("**🚀 創52週新高**")
            st.markdown(
                """
                - 前一交易日盤中高點創下52週新高
                - 獨立觀察清單，不受 SEPA 影響
                - 適合判斷強勢突破股與市場整體強勢
                """
            )
        with col5:
            st.markdown("**📉 創52週新低**")
            st.markdown(
                """
                - 前一交易日盤中低點創下52週新低
                - 獨立觀察清單，不受 SEPA 影響
                - 適合判斷市場整體弱勢狀況
                """
            )
        st.divider()
        st.subheader("🆕 群組三：新上市公司觀察")
        st.caption("可自由勾選，獨立於群組一，不受其他篩選條件影響")
        st.markdown("**🆕 上市未滿半年的公司**")
        st.markdown(
            """
            - 首次交易日在近 180 天內的新股
            - 獨立觀察清單，不受 SEPA 影響
            - 顯示上市日期、上市天數與日均成交量
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
    st.info("執行 SEPA 篩選中…")
    prog2 = st.progress(0.0)
    results: list[dict] = []
    high52_results: list[dict] = []
    low52_results: list[dict] = []
    new_listing_results: list[dict] = []
    total = len(all_data)

    for idx, (ticker, df) in enumerate(all_data.items()):
        prog2.progress((idx + 1) / total)

        try:
            close = df["Close"]
            current_price = float(close.iloc[-1])
            avg_volume = float(df["Volume"].tail(50).mean())

            # ── 基本篩選（股價 & 成交量）──
            if current_price < cfg["min_price"] or avg_volume < cfg["min_avg_volume"]:
                continue

            row: dict = {"Ticker": ticker}
            sepa_ok = True

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
                    Price     = round(sepa["price"], 2),
                    日均量萬股 = round(avg_volume / 10000, 1),
                    SMA50     = round(sepa["sma50"], 2),
                    SMA150    = round(sepa["sma150"], 2),
                    SMA200    = round(sepa["sma200"], 2),
                    SEPA條件  = f"{sepa['sepa_count']}/7",
                    RS評分    = int(rs),
                    高52週    = round(sepa["high52"], 2),
                    低52週    = round(sepa["low52"], 2),
                    距高點pct = round(from_high_pct, 1),
                )

            # ── 前一交易日創52週新高（市況觀察，獨立於其他篩選）──
            if cfg["use_near_high"] and len(df) >= 2:
                prev_close = float(df["Close"].iloc[-2])
                high_window = df["High"].iloc[max(0, len(df) - 253):-1]
                high_52w_prev = float(high_window.max())
                if prev_close >= high_52w_prev * 0.98:
                    gap_pct = round((prev_close - high_52w_prev) / high_52w_prev * 100, 2)
                    high52_results.append({
                        "Ticker": ticker,
                        "現價": round(current_price, 2),
                        "前日收盤": round(prev_close, 2),
                        "52週最高": round(high_52w_prev, 2),
                        "距高點%": gap_pct,
                    })

            # ── 前一交易日創52週新低（市況觀察，獨立於其他篩選）──
            if cfg["use_near_low"] and len(df) >= 2:
                prev_close_low = float(df["Close"].iloc[-2])
                low_window = df["Low"].iloc[max(0, len(df) - 253):-1]
                low_52w_prev = float(low_window.min())
                if prev_close_low <= low_52w_prev * 1.02:
                    gap_pct_low = round((prev_close_low - low_52w_prev) / low_52w_prev * 100, 2)
                    low52_results.append({
                        "Ticker": ticker,
                        "現價": round(current_price, 2),
                        "前日收盤": round(prev_close_low, 2),
                        "52週最低": round(low_52w_prev, 2),
                        "距低點%": gap_pct_low,
                    })

            # ── 上市未滿半年（市況觀察，獨立於其他篩選）──
            if cfg["use_new_listing"]:
                first_date = df.index[0]
                last_date = df.index[-1]
                cutoff = last_date - pd.Timedelta(days=180)
                if first_date >= cutoff:
                    days_listed = (last_date - first_date).days
                    new_listing_results.append({
                        "Ticker": ticker,
                        "現價": round(current_price, 2),
                        "上市日期": first_date.strftime("%Y-%m-%d"),
                        "上市天數": days_listed,
                        "日均量萬股": round(avg_volume / 10000, 1),
                    })

            if sepa_ok:
                if len(df) >= 2:
                    prev  = float(df["Close"].iloc[-2])
                    today = float(df["Close"].iloc[-1])
                    daily_gain = (today - prev) / prev * 100 if prev > 0 else 0.0
                else:
                    daily_gain = 0.0
                row["今日漲幅%"] = round(daily_gain, 2)
                results.append(row)

        except Exception:
            continue

    prog2.progress(1.0)

    # ── 52週新高觀察（獨立區塊，掃描結束後立即呈現）──
    if cfg["use_near_high"]:
        st.divider()
        if high52_results:
            st.subheader(f"🚀 前一交易日創52週新高的股票（共 {len(high52_results)} 支）")
            st.caption("以下股票昨日盤中高點創下近52週新高，不受 SEPA 篩選條件影響，供觀察市場整體強勢狀況。")
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
            st.caption("以下股票昨日盤中低點創下近52週新低，可用於觀察市場整體弱勢狀況。不受 SEPA 篩選條件影響。")
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

    # ── 上市未滿半年觀察（獨立區塊）──
    if cfg["use_new_listing"]:
        st.divider()
        if new_listing_results:
            st.subheader(f"🆕 上市未滿半年的公司（共 {len(new_listing_results)} 支）")
            st.caption("以下股票首次交易日在近 180 天內，不受 SEPA 篩選條件影響，供觀察新股表現。")
            new_listing_df = pd.DataFrame(new_listing_results).sort_values("上市天數")
            st.dataframe(new_listing_df, use_container_width=True, hide_index=True)
            ts_nl = datetime.now().strftime("%Y%m%d_%H%M")
            st.download_button(
                "⬇️ 下載新上市清單 CSV",
                new_listing_df.to_csv(index=False),
                f"new_listing_{ts_nl}.csv",
                "text/csv",
                key="dl_new_listing",
            )
        else:
            st.info("🆕 目前股票池中沒有上市未滿半年的公司。")

    # ── Fundamental screen (optional, slow) ──
    if cfg["use_fundamentals"] and results:
        st.info(f"執行基本面篩選中（{len(results)} 支股票，每支需額外 API 請求，請耐心等候）…")
        prog3 = st.progress(0.0)
        filtered: list[dict] = []
        for i, row in enumerate(results):
            prog3.progress((i + 1) / len(results))
            fund = check_fundamentals(
                row["Ticker"],
                min_eps_growth=cfg["min_eps_growth"],
                min_rev_growth=cfg["min_rev_growth"],
            )
            if fund["pass"]:
                row["季EPS年增%"] = fund["eps_growth"]
                row["年營收增%"] = fund["rev_growth"]
                row["利潤率上升"] = "✓" if fund["margin_rising"] else "✗"
                row["ROE%"] = fund["roe"]
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
    _internal = {"_daily_gain"}
    _rename = {"距高點pct": "距52週高%"}
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
        f"sepa_{ts}.csv",
        "text/csv",
    )

    # ── Detail chart ─────────────────────────
    st.divider()
    st.subheader("📊 個股詳細圖表")

    ticker_list = df_result["Ticker"].tolist()
    selected = st.selectbox("選擇股票", ticker_list)

    if selected and selected in all_data:
        rs_val = rs_ratings.get(selected)
        show_chart(all_data[selected], selected, rs_val)


if __name__ == "__main__":
    main()
