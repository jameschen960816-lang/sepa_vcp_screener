"""Discord Webhook 通知模組"""

import requests


def send_discord_alert(
    webhook_url: str,
    ticker: str,
    gain_pct: float,
    price: float,
    eps_growth: float | None = None,
    rev_growth: float | None = None,
    exchange: str = "",
) -> bool:
    """Send a Discord embed notification for a 5%+ daily gainer passing all filters."""
    if not webhook_url or not webhook_url.startswith("https://discord.com/api/webhooks/"):
        return False

    fields = [
        {"name": "交易所", "value": exchange or "NYSE/NASDAQ", "inline": True},
        {"name": "現價", "value": f"${price:.2f}", "inline": True},
        {"name": "今日漲幅", "value": f"**+{gain_pct:.1f}%**", "inline": True},
    ]

    if eps_growth is not None:
        fields.append({
            "name": "季度 EPS 增長 (YoY)",
            "value": f"{eps_growth:+.1f}%",
            "inline": True,
        })
    if rev_growth is not None:
        fields.append({
            "name": "季度營收增長 (YoY)",
            "value": f"{rev_growth:+.1f}%",
            "inline": True,
        })

    embed = {
        "title": f"🚀  {ticker}  +{gain_pct:.1f}%  單日突破",
        "description": (
            "通過 **SEPA 趨勢模板** + **基本面篩選**（季度 EPS & 營收 YoY ≥ 25%）"
        ),
        "color": 0x00C853,
        "fields": fields,
        "footer": {"text": "SEPA/VCP 篩選器 — NYSE & NASDAQ 全市場監控"},
    }

    try:
        resp = requests.post(
            webhook_url,
            json={"embeds": [embed]},
            timeout=10,
        )
        return resp.status_code == 204
    except Exception:
        return False
