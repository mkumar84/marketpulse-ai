import re
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.agents.compliance_agent import add_ai_disclaimer
from src.agents.executive_summary_agent import generate_summary
from src.agents.market_agents import run_market_intelligence_workflow
from src.agents.trend_agent import detect_risks, detect_top_themes
from src.ingestion.market_data_agent import fetch_market_prices
from src.ingestion.news_agent import fetch_news
from src.ingestion.sample_loader import load_sample_data
from src.rag.research_copilot import answer_question
from src.rag.vector_store import delete_collection, get_collection, index_dataframe
from src.sentiment.sentiment_engine import score_text, sentiment_label
from src.utils.config import ALPHA_VANTAGE_API_KEY, NEWS_API_KEY, OPENAI_API_KEY

st.set_page_config(
    page_title="MarketPulse AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Design system ──────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
/* ── Wealthsimple-inspired design system ── */
:root {
    --mp-bg:     #f5f7f6;
    --mp-card:   #ffffff;
    --mp-text:   #0d1b2a;
    --mp-muted:  #6b7280;
    --mp-border: #e4e8e4;
    --mp-accent: #00c896;
    --mp-green:  #00c896;
    --mp-amber:  #d97706;
    --mp-red:    #dc2626;
}

/* ── App shell ── */
.stApp {
    background: #f5f7f6;
    color: var(--mp-text);
    font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", sans-serif;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1a2540 100%);
}
[data-testid="stSidebar"] * { color: #f1f5f9 !important; }
.block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1300px; }

/* ── Hero (Command Center) ── */
.hero {
    padding: 2rem 2.5rem;
    border: 1px solid var(--mp-border);
    border-left: 5px solid var(--mp-accent);
    border-radius: 20px;
    background: #ffffff;
    box-shadow: 0 2px 16px rgba(0,0,0,0.06);
    margin-bottom: 1.5rem;
}
.eyebrow {
    color: var(--mp-accent);
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-size: 0.7rem;
    margin-bottom: 0.45rem;
}
.hero-title {
    font-size: 2.4rem; line-height: 1.05; font-weight: 800;
    margin-bottom: 0.4rem; color: var(--mp-text);
}
.hero-subtitle {
    color: var(--mp-muted); font-size: 0.97rem; max-width: 800px;
    margin-bottom: 1.1rem; line-height: 1.65;
}
.hero-chips { display: flex; flex-wrap: wrap; gap: 0.5rem; }
.hero-chip {
    display: inline-block; padding: 0.3rem 0.8rem; border-radius: 999px;
    font-size: 0.72rem; font-weight: 600;
    background: #e6faf5; color: #007a5a; border: 1px solid #a7f3d8;
}

/* ── Page header (sub-pages) ── */
.page-header {
    display: flex; align-items: center; gap: 1rem;
    padding: 1rem 1.5rem;
    border: 1px solid var(--mp-border);
    border-left: 5px solid var(--mp-accent);
    border-radius: 16px;
    background: #ffffff;
    box-shadow: 0 2px 12px rgba(0,0,0,0.05);
    margin-bottom: 1.5rem;
}
.page-header-title { font-size: 1.25rem; font-weight: 800; color: var(--mp-text); }
.page-header-sub   { font-size: 0.82rem; color: var(--mp-muted); margin-top: 0.1rem; }
.persona-tag {
    display: inline-block; padding: 0.25rem 0.65rem; border-radius: 999px;
    font-size: 0.7rem; font-weight: 700;
    background: #e6faf5; color: #007a5a; border: 1px solid #a7f3d8;
    white-space: nowrap;
}

/* ── Section title ── */
.section-title { font-weight: 800; font-size: 1.1rem; margin: 0.6rem 0 0.8rem 0; color: var(--mp-text); }

/* ── Metric cards ── */
.metric-card {
    background: #ffffff;
    border: 1px solid var(--mp-border);
    border-top: 3px solid var(--mp-accent);
    border-radius: 16px;
    padding: 1.25rem 1.2rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    min-height: 118px; position: relative;
    transition: box-shadow 0.18s ease, transform 0.18s ease;
}
.metric-card:hover {
    box-shadow: 0 8px 28px rgba(0,200,150,0.12);
    transform: translateY(-2px);
}
.metric-label {
    color: var(--mp-muted); font-size: 0.7rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.5rem;
}
.metric-value { color: var(--mp-text); font-size: 1.5rem; font-weight: 800; line-height: 1.15; word-break: break-word; overflow-wrap: break-word; white-space: normal; }
.metric-help  { color: var(--mp-muted); font-size: 0.73rem; margin-top: 0.5rem; line-height: 1.4; }
.metric-badge {
    display: inline-block; padding: 0.18rem 0.5rem; border-radius: 999px;
    font-size: 0.65rem; font-weight: 700;
    background: #e6faf5; color: #007a5a; border: 1px solid #a7f3d8;
    position: absolute; top: 0.9rem; right: 0.9rem;
}
.metric-badge-live { background: #fef3c7; color: #92400e; border-color: #fde68a; }
.metric-badge-warn { background: #fff1f2; color: #9f1239; border-color: #fecdd3; }

/* ── Insight card ── */
.insight-card {
    background: #ffffff; border: 1px solid var(--mp-border); border-radius: 18px;
    padding: 1.4rem 1.5rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.05);
    margin-bottom: 1rem;
}

/* ── Pills ── */
.pill {
    display: inline-block; padding: 0.3rem 0.65rem; border-radius: 999px;
    font-size: 0.74rem; font-weight: 600;
    margin: 0.14rem 0.18rem 0.14rem 0;
    border: 1px solid var(--mp-border); background: #f5f7f6; color: #374151;
}
.risk-pill  { background: #fff7ed; color: #9a3412; border-color: #fed7aa; }
.theme-pill { background: #e6faf5; color: #007a5a; border-color: #a7f3d8; }

/* ── Source rows ── */
.source-row {
    padding: 1rem 1.25rem; border: 1px solid var(--mp-border); border-radius: 14px;
    background: #ffffff; margin-bottom: 0.65rem;
    transition: box-shadow 0.15s ease, border-color 0.15s ease;
}
.source-row:hover { box-shadow: 0 4px 18px rgba(0,0,0,0.07); border-color: var(--mp-accent); }
.small-muted { color: var(--mp-muted); font-size: 0.82rem; }

/* ── Status dots ── */
.dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 5px; vertical-align: middle; }
.dot-green { background: #00c896; }
.dot-red   { background: #ef4444; }
.dot-amber { background: #f59e0b; }

/* ── Data Health cards ── */
.health-card {
    background: #ffffff; border: 1px solid var(--mp-border); border-radius: 14px;
    padding: 1rem 1.2rem; box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    margin-bottom: 0.75rem;
}
.health-label { font-size: 0.72rem; color: var(--mp-muted); font-weight: 700; margin-bottom: 0.25rem; text-transform: uppercase; letter-spacing: 0.05em; }
.health-value { font-size: 1.1rem; font-weight: 800; color: var(--mp-text); }

/* ── Roadmap timeline ── */
.timeline-card {
    background: #ffffff; border: 1px solid var(--mp-border); border-radius: 20px;
    padding: 1.5rem 1.5rem; box-shadow: 0 2px 12px rgba(0,0,0,0.05); height: 100%;
}
.phase-badge { display: inline-block; padding: 0.28rem 0.75rem; border-radius: 999px; font-size: 0.7rem; font-weight: 700; margin-bottom: 0.75rem; }
.phase-done   { background: #e6faf5; color: #007a5a; border: 1px solid #a7f3d8; }
.phase-next   { background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe; }
.phase-future { background: #fdf4ff; color: #7e22ce; border: 1px solid #e9d5ff; }
.phase-title  { font-size: 1.15rem; font-weight: 800; margin-bottom: 0.25rem; color: var(--mp-text); }
.phase-sub    { font-size: 0.8rem; color: var(--mp-muted); margin-bottom: 1rem; }
.roadmap-item { font-size: 0.84rem; padding: 0.32rem 0; border-bottom: 1px solid #f1f5f9; color: #374151; }
.roadmap-item:last-child { border-bottom: none; }

/* ── Buttons ── */
.stButton > button {
    border-radius: 999px; padding: 0.52rem 1.3rem;
    font-weight: 700; border: 1px solid #d1d5db;
    transition: all 0.15s ease;
}
.stButton > button:hover { border-color: var(--mp-accent); color: var(--mp-accent); }
div[data-testid="stMetric"] {
    background: #ffffff; border: 1px solid var(--mp-border);
    border-top: 3px solid var(--mp-accent);
    border-radius: 14px; padding: 0.85rem 1rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}

/* ── Sidebar toggle button — make it obvious when sidebar is collapsed ── */
[data-testid="collapsedControl"] {
    background: var(--mp-accent) !important;
    border-radius: 0 8px 8px 0 !important;
    opacity: 0.9;
}
[data-testid="collapsedControl"]:hover { opacity: 1 !important; }
[data-testid="collapsedControl"] svg { fill: #ffffff !important; }

/* ── Sidebar controls — dark theme overrides ── */
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
    background: #1e2d47 !important; border: 1px solid #334155 !important;
    border-radius: 10px !important; color: #f1f5f9 !important;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] svg { fill: #94a3b8 !important; }
[data-testid="stSidebar"] [data-testid="stSelectbox"] span,
[data-testid="stSidebar"] [data-testid="stSelectbox"] p { color: #f1f5f9 !important; }
[data-testid="stSelectboxVirtualDropdown"] { background: #1e2d47 !important; border: 1px solid #334155 !important; }
[data-testid="stSelectboxVirtualDropdown"] li { color: #f1f5f9 !important; }
[data-testid="stSelectboxVirtualDropdown"] li:hover,
[data-testid="stSelectboxVirtualDropdown"] li[aria-selected="true"] { background: #00c896 !important; color: #0d1b2a !important; }
[data-testid="stSidebar"] .stButton > button { background: #1e2d47 !important; border: 1px solid #334155 !important; color: #f1f5f9 !important; }
[data-testid="stSidebar"] .stButton > button:hover { background: #00c896 !important; border-color: #00c896 !important; color: #0d1b2a !important; }
[data-testid="stSidebar"] [data-testid="stAlert"] {
    background: rgba(30,45,71,0.85) !important; border-color: #334155 !important;
    color: #f1f5f9 !important; border-radius: 10px !important;
}
[data-testid="stSidebar"] [data-testid="stAlert"] p { color: #f1f5f9 !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Helper functions ───────────────────────────────────────────────────────────


def sanitize_url(url: str) -> str:
    """Allow only http/https URLs to prevent XSS via unsafe_allow_html."""
    if not url or not isinstance(url, str):
        return ""
    url = url.strip()
    return url if re.match(r"^https?://", url) else ""


def sentiment_color(sentiment: str) -> str:
    return {"Bullish": "#00c896", "Bearish": "#ef4444"}.get(sentiment, "#d97706")


def metric_card(label: str, value: str, helper: str = "", badge: str = "", badge_type: str = "ok"):
    badge_class = {"ok": "metric-badge", "live": "metric-badge metric-badge-live", "warn": "metric-badge metric-badge-warn"}.get(badge_type, "metric-badge")
    badge_html = f'<span class="{badge_class}">{badge}</span>' if badge else ""
    # No blank lines inside the HTML — Streamlit's Markdown parser breaks HTML on blank lines
    st.markdown(
        f'<div class="metric-card">{badge_html}'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-help">{helper}</div></div>',
        unsafe_allow_html=True,
    )


def theme_pills(values, css_class: str = "theme-pill"):
    html = "".join(f'<span class="pill {css_class}">{v}</span>' for v in values if pd.notna(v))
    st.markdown(html or '<span class="small-muted">No signals detected.</span>', unsafe_allow_html=True)


def _parse_evidence_doc(doc: str) -> dict:
    """Parse a pipe-delimited ChromaDB document string into structured fields."""
    parts = [p.strip() for p in doc.split("|")]
    result = {"sector": "—", "source": "—", "headline": "—", "theme": "", "risk": ""}
    if len(parts) > 0:
        result["sector"] = parts[0]
    if len(parts) > 1:
        result["source"] = parts[1]
    if len(parts) > 2:
        result["headline"] = parts[2]
    for part in parts[3:]:
        if part.startswith("Theme:"):
            result["theme"] = part.replace("Theme:", "").strip()
        elif part.startswith("Risk:"):
            result["risk"] = part.replace("Risk:", "").strip()
    return result


def _style_sentiment(val: str) -> str:
    return {
        "Bullish": "background-color:#e6faf5; color:#007a5a; font-weight:700",
        "Bearish": "background-color:#fff1f2; color:#dc2626; font-weight:700",
    }.get(val, "background-color:#fef3c7; color:#d97706; font-weight:700")


def _style_score(val: float) -> str:
    if val > 0.15:
        return "color:#007a5a; font-weight:700"
    if val < -0.15:
        return "color:#dc2626; font-weight:700"
    return "color:#d97706; font-weight:700"


def _color_signal_row(row) -> list[str]:
    """Return full-row CSS based on the Sentiment column value."""
    sentiment = row.get("Sentiment", "")
    if sentiment == "Bullish":
        base = "background-color:#dcfce7; color:#0d1b2a"
        accent = "background-color:#dcfce7; color:#007a5a; font-weight:700"
    elif sentiment == "Bearish":
        base = "background-color:#fee2e2; color:#0d1b2a"
        accent = "background-color:#fee2e2; color:#dc2626; font-weight:700"
    else:
        base = "background-color:#fef9ee; color:#0d1b2a"
        accent = "background-color:#fef9ee; color:#d97706; font-weight:700"
    return [
        accent if col in ("Sentiment", "Score") else base
        for col in row.index
    ]


def make_signal_gauge(avg_score: float, sentiment: str):
    bar_color = {"Bullish": "#00c896", "Bearish": "#ef4444"}.get(sentiment, "#d97706")
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=float(avg_score),
            delta={"reference": 0, "valueformat": ".2f", "increasing": {"color": "#00c896"}, "decreasing": {"color": "#ef4444"}},
            number={"valueformat": ".2f", "font": {"size": 36, "color": "#0d1b2a"}},
            gauge={
                "axis": {
                    "range": [-1, 1], "tickwidth": 1,
                    "tickvals": [-1, -0.5, 0, 0.5, 1],
                    "ticktext": ["−1", "−0.5", "0", "+0.5", "+1"],
                    "tickcolor": "#6b7280",
                },
                "bar": {"color": bar_color, "thickness": 0.28},
                "bgcolor": "#ffffff",
                "borderwidth": 0,
                "steps": [
                    {"range": [-1,  -0.15], "color": "#fff1f2"},
                    {"range": [-0.15, 0.15], "color": "#f9fafb"},
                    {"range": [0.15,  1],   "color": "#e6faf5"},
                ],
                "threshold": {
                    "line": {"color": "#0d1b2a", "width": 2},
                    "thickness": 0.8,
                    "value": float(avg_score),
                },
            },
            title={"text": f"Signal Strength · <b>{sentiment}</b>", "font": {"size": 13, "color": "#6b7280"}},
        )
    )
    fig.update_layout(
        height=270,
        margin=dict(l=20, r=20, t=65, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "-apple-system, BlinkMacSystemFont, Inter, sans-serif"},
    )
    return fig


def page_header(title: str, subtitle: str, persona: str):
    st.markdown(
        f"""<div class="page-header">
            <div>
                <div class="page-header-title">{title}</div>
                <div class="page-header-sub">{subtitle}</div>
            </div>
            <div style="margin-left:auto;">
                <span class="persona-tag">👤 {persona}</span>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )


def enrich_news_to_sector(news_df: pd.DataFrame, sector: str) -> pd.DataFrame:
    if news_df.empty:
        return news_df
    rows = []
    for _, row in news_df.iterrows():
        headline = row.get("headline", "") or row.get("summary", "")
        sentiment, score = score_text(headline)
        rows.append({
            "sector": sector, "source": row.get("source", "Live News"),
            "company": "Live Source", "ticker": "NA",
            "headline": headline, "sentiment": sentiment, "score": score,
            "theme": "Live market signal", "risk": "Requires analyst validation",
            "url": sanitize_url(str(row.get("url", ""))),
        })
    return pd.DataFrame(rows)


# ── Session state bootstrap ────────────────────────────────────────────────────

for key in ("summaries", "indexed_sectors", "exec_briefs", "live_cache"):
    if key not in st.session_state:
        st.session_state[key] = {}

# ── Data ───────────────────────────────────────────────────────────────────────


@st.cache_data
def load_data():
    return load_sample_data()


base_df = load_data()

# ── Sidebar ────────────────────────────────────────────────────────────────────

st.sidebar.markdown("## MarketPulse AI")
st.sidebar.caption("GenAI · Alternative Data · Capital Markets")
st.sidebar.markdown("---")

mode = st.sidebar.toggle(
    "Live news mode",
    value=False,
    help="Fetches real-time news from NewsAPI when enabled.",
)

sector = st.sidebar.selectbox("Sector focus", sorted(base_df["sector"].unique()))

# Live news — cached per sector, refreshed on demand
if mode:
    cache_key = f"live_{sector}"
    if cache_key not in st.session_state.live_cache:
        with st.spinner(f"Fetching live news for {sector}…"):
            live_news = fetch_news(sector, page_size=6)
            st.session_state.live_cache[cache_key] = live_news
        if not st.session_state.live_cache[cache_key].empty:
            n = len(st.session_state.live_cache[cache_key])
            st.toast(f"Loaded {n} live articles for {sector}", icon="📡")

    live_df = st.session_state.live_cache.get(cache_key, pd.DataFrame())
    live_count = len(live_df)

    if live_count:
        st.sidebar.success(f"📡 {live_count} live articles loaded")
    else:
        st.sidebar.warning("📡 Live mode active — no articles found for this sector")

    if st.sidebar.button("↺ Refresh live data"):
        del st.session_state.live_cache[cache_key]
        st.rerun()

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Workspace",
    [
        "Command Center",
        "Signals Explorer",
        "Market Data",
        "Live Intelligence",
        "Research Copilot",
        "Executive Brief",
        "Data Health",
        "Product Roadmap",
    ],
    captions=[
        "C-Suite & Portfolio Managers",
        "Analysts & Research Associates",
        "Traders & Quantitative Analysts",
        "Market Intelligence Teams",
        "Strategists & Sales Teams",
        "Client Coverage & Management",
        "Data & Governance",
        "Platform Vision",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption("Design principle: executive-first · source-grounded · human-in-the-loop")

# API status strip
api_items = [
    ("OpenAI", bool(OPENAI_API_KEY)),
    ("News", bool(NEWS_API_KEY)),
    ("AlphaV", bool(ALPHA_VANTAGE_API_KEY)),
]
status_html = " &nbsp; ".join(
    f'<span class="dot {"dot-green" if ok else "dot-red"}"></span>'
    f'<span style="font-size:0.72rem;color:#94a3b8;">{name}</span>'
    for name, ok in api_items
)
st.sidebar.markdown(status_html, unsafe_allow_html=True)

# ── Build active dataframe ─────────────────────────────────────────────────────

sector_df = base_df[base_df["sector"] == sector].copy()
live_article_count = 0

if mode:
    live_df = st.session_state.live_cache.get(f"live_{sector}", pd.DataFrame())
    live_sector_df = enrich_news_to_sector(live_df, sector)
    if not live_sector_df.empty:
        live_article_count = len(live_sector_df)
        sector_df = pd.concat([sector_df, live_sector_df], ignore_index=True)

avg_score = sector_df["score"].mean()
overall_sentiment = sentiment_label(avg_score)

# ── Page: Command Center ───────────────────────────────────────────────────────

if page == "Command Center":
    st.markdown(
        f"""<div class="hero">
            <div class="eyebrow">GenAI · Alternative Data Intelligence Platform</div>
            <div class="hero-title">MarketPulse AI</div>
            <div class="hero-subtitle">
                Turns fragmented alternative data — news, social signals, market prices, filings —
                into explainable, source-grounded market intelligence for capital markets teams.
                Built for analysts, advisors, and executives who need clarity in under 30 seconds.
            </div>
            <div class="hero-chips">
                <span class="hero-chip">🔍 Alternative Data</span>
                <span class="hero-chip">🤖 RAG Copilot</span>
                <span class="hero-chip">📊 Sentiment Signals</span>
                <span class="hero-chip">🔒 Source Grounded</span>
                <span class="hero-chip">🏦 Capital Markets</span>
                <span class="hero-chip">👤 Human-in-the-Loop</span>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='section-title'>Command Center</div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    bullish_n = (sector_df["sentiment"] == "Bullish").sum()
    bearish_n = (sector_df["sentiment"] == "Bearish").sum()

    with c1:
        metric_card("Sector", sector, "Intelligence lens", badge="Active", badge_type="ok")
    with c2:
        posture_badge = "LIVE" if mode and live_article_count else ""
        metric_card("Market posture", overall_sentiment, f"Avg score: {avg_score:.2f}",
                    badge=posture_badge, badge_type="live")
    with c3:
        metric_card("Signal score", f"{avg_score:.2f}", "−1 bearish → +1 bullish")
    with c4:
        metric_card("Evidence items", str(len(sector_df)),
                    f"↑ {bullish_n} bullish  ·  ↓ {bearish_n} bearish")

    left, right = st.columns([1.5, 1])

    with left:
        st.markdown("**Sentiment landscape by source**")
        fig = px.bar(
            sector_df, x="source", y="score", color="sentiment",
            hover_data=["headline", "theme", "risk"],
            color_discrete_map={"Bullish": "#00c896", "Bearish": "#ef4444", "Neutral": "#d97706"},
            barmode="overlay",
        )
        fig.update_layout(
            height=340, margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            legend_title_text="", yaxis_title="Signal score", xaxis_title="",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis=dict(gridcolor="#f0f0f0", zerolinecolor="#d1d5db", zerolinewidth=1.5),
            xaxis=dict(tickangle=-20),
            font=dict(family="-apple-system, BlinkMacSystemFont, Inter, sans-serif", size=12),
        )
        fig.add_hline(y=0, line_dash="dot", line_color="#9ca3af", line_width=1)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.plotly_chart(make_signal_gauge(avg_score, overall_sentiment), use_container_width=True)
        st.markdown("**Emerging themes**")
        theme_pills(detect_top_themes(sector_df, top_n=6), "theme-pill")
        st.markdown("**Watchlist risks**")
        theme_pills(detect_risks(sector_df, top_n=6), "risk-pill")

    # Executive readout — cached per sector to avoid redundant API calls
    summary_key = f"summary_{sector}_{len(sector_df)}"
    st.markdown("---")
    col_a, col_b = st.columns([6, 1])
    with col_a:
        st.markdown("**Executive readout**")
    with col_b:
        if st.button("↺ Refresh", key="refresh_summary", help="Re-generate AI readout"):
            if summary_key in st.session_state.summaries:
                del st.session_state.summaries[summary_key]
            st.rerun()

    if summary_key not in st.session_state.summaries:
        with st.spinner("Generating AI market readout…"):
            st.session_state.summaries[summary_key] = generate_summary(sector_df, sector)

    st.markdown(add_ai_disclaimer(st.session_state.summaries[summary_key]))
    st.caption(f"Generated: {datetime.now().strftime('%d %b %Y · %H:%M')}  ·  Model: GPT-4o-mini  ·  Source-grounded")

# ── Page: Signals Explorer ─────────────────────────────────────────────────────

elif page == "Signals Explorer":
    page_header(
        "Signals Explorer",
        "Inspect the evidence behind the executive view — every signal traced to its source.",
        "Analysts & Research Associates",
    )

    col_filter, col_sort = st.columns([3, 1])
    with col_filter:
        source_filter = st.multiselect(
            "Filter sources",
            sorted(sector_df["source"].dropna().unique()),
            default=sorted(sector_df["source"].dropna().unique()),
        )
    with col_sort:
        sort_by = st.selectbox("Sort by", ["Score (high→low)", "Score (low→high)", "Source"])

    filtered = sector_df[sector_df["source"].isin(source_filter)] if source_filter else sector_df

    sort_map = {
        "Score (high→low)": ("score", False),
        "Score (low→high)": ("score", True),
        "Source": ("source", True),
    }
    col_s, asc_s = sort_map[sort_by]
    filtered = filtered.sort_values(col_s, ascending=asc_s)

    bullish_c = (filtered["sentiment"] == "Bullish").sum()
    bearish_c = (filtered["sentiment"] == "Bearish").sum()
    neutral_c = (filtered["sentiment"] == "Neutral").sum()
    st.caption(f"Showing {len(filtered)} signals · ↑ {bullish_c} Bullish · ↓ {bearish_c} Bearish · → {neutral_c} Neutral")

    for _, row in filtered.iterrows():
        color = sentiment_color(row.get("sentiment", "Neutral"))
        url = sanitize_url(str(row.get("url", "")))
        link = f'<a href="{url}" target="_blank" style="font-size:0.8rem;color:var(--mp-accent);">Open source ↗</a>' if url else ""
        st.markdown(
            f"""<div class="source-row">
                <div class="small-muted">{row.get('source','')} · {row.get('company','')} · {row.get('ticker','')}</div>
                <div style="font-weight:700;font-size:1rem;margin:0.22rem 0;">{row.get('headline','')}</div>
                <span class="pill" style="color:{color};">{row.get('sentiment','Neutral')} &nbsp; {row.get('score',0):.2f}</span>
                <span class="pill theme-pill">{row.get('theme','')}</span>
                <span class="pill risk-pill">{row.get('risk','')}</span>
                <span style="float:right;margin-top:-1.6rem;">{link}</span>
            </div>""",
            unsafe_allow_html=True,
        )

    st.caption("ⓘ Sample dataset links are illustrative. Enable Live news mode in the sidebar to fetch real-time article URLs.")

    st.markdown("---")
    st.markdown("**📋 Structured signal table**")
    tbl = filtered[["source", "company", "ticker", "headline", "sentiment", "score", "theme", "risk", "url"]].copy()
    tbl["score"] = tbl["score"].round(3)
    styled_tbl = (
        tbl.rename(columns={
            "source": "Source", "company": "Company", "ticker": "Ticker",
            "headline": "Headline", "sentiment": "Sentiment", "score": "Score",
            "theme": "Theme", "risk": "Risk", "url": "Link",
        })
        .style
        .apply(_color_signal_row, axis=1)
        .format({"Score": "{:+.2f}"})
    )
    st.dataframe(
        styled_tbl,
        use_container_width=True,
        hide_index=True,
        column_config={"Link": st.column_config.LinkColumn("Source link", display_text="Open ↗")},
    )

# ── Page: Market Data ──────────────────────────────────────────────────────────

elif page == "Market Data":
    page_header(
        "Market Data Agent",
        "Live OHLCV price data pulled from Alpha Vantage for sector-relevant tickers.",
        "Traders & Quantitative Analysts",
    )

    tickers = sorted([t for t in sector_df["ticker"].dropna().unique() if t != "NA"])

    tab1, tab2 = st.tabs(["Sector Tickers", "Custom Search"])

    with tab1:
        if tickers:
            st.markdown(f"**Detected tickers:** " + "  ".join(f"`{t}`" for t in tickers))
        else:
            st.info("No tickers detected for this sector in the sample dataset.")
        if tickers and st.button("Fetch 5-day price movement", key="sector_fetch"):
            with st.spinner("Fetching market data from Alpha Vantage…"):
                price_df = fetch_market_prices(tickers, days=5)
            if price_df.empty:
                st.warning("No price data available. Live market data requires an active data connection.")
            else:
                n_tickers = price_df["symbol"].nunique()
                st.success(f"Fetched data for {n_tickers} ticker(s)")

                # ── Close price trend (multi-ticker) ──────────────────────────
                st.markdown("**Close price — 5 day trend**")
                palette = ["#00c896", "#ef4444", "#d97706", "#6366f1", "#ec4899",
                           "#0ea5e9", "#84cc16", "#f97316"]
                fig_close = go.Figure()
                for i, sym in enumerate(price_df["symbol"].unique()):
                    sym_df = price_df[price_df["symbol"] == sym].sort_values("date")
                    fig_close.add_trace(go.Scatter(
                        x=sym_df["date"], y=sym_df["close"],
                        mode="lines+markers",
                        name=sym,
                        line=dict(color=palette[i % len(palette)], width=2.5),
                        marker=dict(size=6),
                        hovertemplate=f"<b>{sym}</b><br>Date: %{{x}}<br>Close: $%{{y:.2f}}<extra></extra>",
                    ))
                fig_close.update_layout(
                    height=340, margin=dict(l=10, r=10, t=20, b=10),
                    hovermode="x unified", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
                    yaxis=dict(gridcolor="#f0f0f0", tickprefix="$"),
                    xaxis=dict(showgrid=False),
                    font=dict(family="-apple-system, BlinkMacSystemFont, Inter, sans-serif", size=12),
                )
                st.plotly_chart(fig_close, use_container_width=True)

                # ── Daily high-low range chart ────────────────────────────────
                st.markdown("**Daily High / Low range**")
                fig_range = go.Figure()
                for i, sym in enumerate(price_df["symbol"].unique()):
                    sym_df = price_df[price_df["symbol"] == sym].sort_values("date")
                    color = palette[i % len(palette)]
                    fig_range.add_trace(go.Bar(
                        x=sym_df["date"],
                        y=sym_df["high"] - sym_df["low"],
                        name=f"{sym} range",
                        base=sym_df["low"],
                        marker_color=color,
                        opacity=0.75,
                        hovertemplate=f"<b>{sym}</b><br>Date: %{{x}}<br>Low: $%{{base:.2f}}<br>High: $%{{y:.2f}}<extra></extra>",
                    ))
                fig_range.update_layout(
                    height=300, margin=dict(l=10, r=10, t=20, b=10),
                    barmode="overlay", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
                    yaxis=dict(gridcolor="#f0f0f0", title="Price ($)"),
                    xaxis=dict(showgrid=False),
                    font=dict(family="-apple-system, BlinkMacSystemFont, Inter, sans-serif", size=12),
                )
                st.plotly_chart(fig_range, use_container_width=True)

                # ── Volatility ────────────────────────────────────────────────
                if "volatility_pct" in price_df.columns:
                    st.markdown("**Annualised volatility (%) · trailing window**")
                    latest_vol = (
                        price_df.dropna(subset=["volatility_pct"])
                        .groupby("symbol")["volatility_pct"].last()
                        .reset_index()
                        .rename(columns={"symbol": "Ticker", "volatility_pct": "Volatility %"})
                    )
                    if not latest_vol.empty:
                        vol_cols = st.columns(min(len(latest_vol), 4))
                        for vi, (_, vrow) in enumerate(latest_vol.iterrows()):
                            with vol_cols[vi % len(vol_cols)]:
                                metric_card(vrow["Ticker"], f"{vrow['Volatility %']:.1f}%", "Annualised vol")

                with st.expander("Raw price data table"):
                    display_cols = ["symbol", "date", "open", "high", "low", "close", "volume",
                                    "volatility_pct", "daily_range_pct"]
                    available = [c for c in display_cols if c in price_df.columns]
                    st.dataframe(price_df[available].sort_values(["symbol", "date"], ascending=[True, False]),
                                 use_container_width=True, hide_index=True)

    with tab2:
        st.write("**Search any ticker for current market data**")
        col1, col2 = st.columns([3, 1])
        with col1:
            custom_ticker = st.text_input(
                "Ticker symbol", placeholder="e.g. AAPL, RY.TO, MSFT, GOOGL",
                label_visibility="collapsed",
            ).upper()
        with col2:
            search_btn = st.button("Search", key="custom_fetch")

        if search_btn and custom_ticker:
            with st.spinner(f"Fetching data for {custom_ticker}…"):
                price_df = fetch_market_prices([custom_ticker], days=30)
            if price_df.empty:
                st.error(f"No data found for **{custom_ticker}**. Verify the symbol or check API limits.")
            else:
                st.success(f"Fetched {len(price_df)} records for **{custom_ticker}**")
                latest = price_df.iloc[-1]
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Latest Close", f"${latest['close']:.2f}")
                m2.metric("30d High",     f"${price_df['high'].max():.2f}")
                m3.metric("30d Low",      f"${price_df['low'].min():.2f}")
                m4.metric("Avg Volume",   f"{price_df['volume'].mean():,.0f}")

                st.markdown(f"**30-Day Price Chart — {custom_ticker}**")
                sorted_df = price_df.sort_values("date")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=sorted_df["date"], y=sorted_df["close"],
                    mode="lines",
                    line=dict(color="#00c896", width=2.5),
                    fill="tozeroy",
                    fillcolor="rgba(0,200,150,0.08)",
                    hovertemplate="<b>%{x}</b><br>Close: $%{y:.2f}<extra></extra>",
                ))
                fig.add_trace(go.Scatter(
                    x=sorted_df["date"], y=sorted_df["high"],
                    mode="lines", line=dict(color="#a7f3d8", width=1, dash="dot"),
                    name="30d High", hovertemplate="High: $%{y:.2f}<extra></extra>",
                ))
                fig.add_trace(go.Scatter(
                    x=sorted_df["date"], y=sorted_df["low"],
                    mode="lines", line=dict(color="#fecdd3", width=1, dash="dot"),
                    name="30d Low", hovertemplate="Low: $%{y:.2f}<extra></extra>",
                ))
                fig.update_layout(
                    height=400, margin=dict(l=10, r=10, t=20, b=10),
                    hovermode="x unified", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
                    yaxis=dict(gridcolor="#f0f0f0", tickprefix="$"),
                    xaxis=dict(showgrid=False),
                    font=dict(family="-apple-system, BlinkMacSystemFont, Inter, sans-serif", size=12),
                )
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("Price history table"):
                    st.dataframe(
                        price_df[["date", "open", "high", "low", "close", "volume"]].sort_values("date", ascending=False),
                        use_container_width=True,
                    )
        elif search_btn and not custom_ticker:
            st.warning("Please enter a ticker symbol.")

# ── Page: Live Intelligence ────────────────────────────────────────────────────

elif page == "Live Intelligence":
    page_header(
        "Live Intelligence Workflow",
        "Agentic pipeline: fetch live news → index into ChromaDB → generate source-grounded market summary.",
        "Market Intelligence Teams",
    )

    topic = st.text_input("Market topic or sector", value=sector, placeholder="e.g. AI Infrastructure, Canadian Banking")

    col_run, col_info = st.columns([2, 5])
    with col_run:
        run_btn = st.button("▶ Run Intelligence Workflow", type="primary")
    with col_info:
        st.caption("Agentic pipeline: fetch live news → index into vector store → generate source-grounded summary")

    if run_btn:
        progress = st.progress(0, text="Step 1 of 3 · Fetching live news…")
        result = None
        with st.spinner("Running MarketPulse agents…"):
            progress.progress(33, text="Step 2 of 3 · Indexing articles into vector store…")
            result = run_market_intelligence_workflow(topic)
            progress.progress(100, text="Step 3 of 3 · Complete")

        progress.empty()
        st.success("Live intelligence workflow completed.")

        c1, c2, c3 = st.columns(3)
        c1.metric("Topic", result.get("topic", topic))
        c2.metric("Articles loaded", result.get("articles_loaded", 0))
        c3.metric("Records indexed", result.get("records_indexed", 0))

        st.markdown("---")
        st.markdown("**AI Market Summary**")
        st.markdown(add_ai_disclaimer(result.get("summary", "No summary generated.")))

        sources = result.get("sources", [])
        if sources:
            st.markdown("**Source evidence**")
            for src in sources:
                title = src.get("title", "Untitled")
                publisher = src.get("source", "Unknown")
                url = sanitize_url(str(src.get("url", "")))
                if url:
                    st.markdown(f"- [{title}]({url}) — *{publisher}*")
                else:
                    st.markdown(f"- {title} — *{publisher}*")
        else:
            st.info("No sources retrieved for this topic. Try a different query or switch to a broader market theme.")

# ── Page: Research Copilot ─────────────────────────────────────────────────────

elif page == "Research Copilot":
    page_header(
        "Research Copilot",
        "Ask questions grounded in indexed evidence. Every answer is source-cited and confidence-rated.",
        "Strategists & Sales Teams",
    )

    # Index current sector — once per sector per session
    if sector not in st.session_state.indexed_sectors:
        with st.spinner(f"Indexing {sector} signals into vector store…"):
            count = index_dataframe(sector_df)
            st.session_state.indexed_sectors[sector] = count
        st.success(f"Indexed {st.session_state.indexed_sectors[sector]} records for {sector}.")
    else:
        st.caption(
            f"Vector store: {st.session_state.indexed_sectors[sector]} records indexed for **{sector}** "
            f"· Local sentence-transformer embeddings · GPT-4o-mini generation"
        )

    examples = [
        "What are the top risks in this sector?",
        "What are the strongest bullish signals?",
        "Create a client-ready market summary.",
        "What evidence supports the current sentiment?",
        "What opportunities should we be tracking?",
    ]

    example = st.selectbox("Try a question", ["— select an example —"] + examples)
    question = st.text_input(
        "Or ask your own question",
        value=example if example != "— select an example —" else "",
        placeholder="e.g. What changed in banking sector sentiment this week?",
    )

    if question:
        fallback = generate_summary(sector_df, sector)
        with st.spinner("Searching evidence and generating response…"):
            answer, evidence = answer_question(question, sector, fallback)

        st.markdown("**Copilot response**")
        st.markdown(add_ai_disclaimer(answer))

        if evidence:
            st.markdown("---")
            st.markdown(f"**Sources used to generate this answer** · {len(evidence)} document(s) retrieved")
            ev_cols = st.columns(min(len(evidence), 2))
            for i, doc in enumerate(evidence):
                parsed = _parse_evidence_doc(doc)
                col = ev_cols[i % len(ev_cols)]
                with col:
                    theme_badge = f'<span class="pill theme-pill">{parsed["theme"]}</span>' if parsed["theme"] else ""
                    risk_badge  = f'<span class="pill risk-pill">{parsed["risk"]}</span>'   if parsed["risk"]  else ""
                    st.markdown(
                        f'<div class="source-row">'
                        f'<div class="small-muted">📂 {parsed["sector"]} &nbsp;·&nbsp; {parsed["source"]}</div>'
                        f'<div style="font-weight:600;font-size:0.9rem;margin:0.3rem 0 0.4rem;">{parsed["headline"]}</div>'
                        f'{theme_badge}{risk_badge}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

# ── Page: Executive Brief ──────────────────────────────────────────────────────

elif page == "Executive Brief":
    page_header(
        "Executive Brief Generator",
        "Converts analyst-grade signals into a boardroom-ready client note.",
        "Client Coverage & Management",
    )

    col_gen, col_note = st.columns([2, 5])
    with col_gen:
        gen_btn = st.button("Generate Executive Note", type="primary")
    with col_note:
        brief_key = f"brief_{sector}_{len(sector_df)}"
        if brief_key in st.session_state.exec_briefs:
            st.caption(f"Showing cached brief for {sector} · Regenerate to refresh")
        else:
            st.caption("Generates a boardroom-ready narrative for the selected sector")

    if gen_btn:
        with st.spinner("Drafting executive note…"):
            brief = generate_summary(sector_df, sector)
            st.session_state.exec_briefs[brief_key] = brief

    if brief_key in st.session_state.exec_briefs:
        st.markdown(add_ai_disclaimer(st.session_state.exec_briefs[brief_key]))
        st.markdown("**Opportunity / Risk matrix**")
        matrix = (
            sector_df[["theme", "risk", "sentiment", "score", "source", "company"]]
            .drop_duplicates(subset=["theme", "risk", "sentiment"])
            .sort_values("score", ascending=False)
            .rename(columns={"theme": "Theme", "risk": "Risk", "sentiment": "Sentiment",
                             "score": "Score", "source": "Source", "company": "Company"})
        )
        styled_matrix = (
            matrix.style
            .applymap(_style_sentiment, subset=["Sentiment"])
            .applymap(_style_score, subset=["Score"])
            .format({"Score": "{:+.2f}"})
            .bar(subset=["Score"], vmin=-1, vmax=1,
                 color=["#fff1f2", "#e6faf5"])
        )
        st.dataframe(styled_matrix, use_container_width=True, hide_index=True)

        csv_export = matrix.to_csv(index=False)
        st.download_button(
            "Download matrix (CSV)", data=csv_export,
            file_name=f"marketpulse_{sector.lower().replace(' ','_')}_matrix.csv",
            mime="text/csv",
        )
    else:
        st.info("Click **Generate Executive Note** to produce a client-ready brief for the selected sector.")

# ── Page: Data Health ──────────────────────────────────────────────────────────

elif page == "Data Health":
    page_header(
        "Data Health",
        "Monitor data coverage, API connectivity, and vector store status across the platform.",
        "Data & Governance",
    )

    # API connectivity
    st.markdown("**API Connectivity**")
    apis = [
        ("OpenAI GPT-4o-mini", OPENAI_API_KEY, "Powers AI summaries and RAG answer generation — embeddings are local"),
        ("NewsAPI", NEWS_API_KEY, "Real-time news ingestion for live market signals"),
        ("Alpha Vantage", ALPHA_VANTAGE_API_KEY, "OHLCV market price data for ticker lookup"),
        ("ChromaDB Vector Store", True, "Local persistent vector database — always available"),
    ]

    a1, a2 = st.columns(2)
    for i, (name, configured, description) in enumerate(apis):
        col = a1 if i % 2 == 0 else a2
        with col:
            dot = f'<span class="dot {"dot-green" if configured else "dot-red"}"></span>'
            status = "Configured" if configured else "Not configured"
            st.markdown(
                f"""<div class="health-card">
                    <div class="health-label">{dot}{name}</div>
                    <div class="health-value" style="font-size:0.9rem;color:{'#007a5a' if configured else '#dc2626'};">{status}</div>
                    <div class="small-muted" style="margin-top:0.3rem;">{description}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # Vector store stats
    st.markdown("**Vector Store**")
    try:
        collection = get_collection()
        doc_count = collection.count()
        v1, v2, v3 = st.columns(3)
        with v1:
            st.markdown(
                f"""<div class="health-card">
                    <div class="health-label">Documents indexed</div>
                    <div class="health-value">{doc_count:,}</div>
                </div>""", unsafe_allow_html=True,
            )
        with v2:
            indexed_sectors = list(st.session_state.indexed_sectors.keys())
            st.markdown(
                f"""<div class="health-card">
                    <div class="health-label">Sectors indexed this session</div>
                    <div class="health-value">{len(indexed_sectors) or '—'}</div>
                    <div class="small-muted">{', '.join(indexed_sectors) if indexed_sectors else 'None yet'}</div>
                </div>""", unsafe_allow_html=True,
            )
        with v3:
            st.markdown(
                '<div class="health-card">'
                '<div class="health-label">Embedding model</div>'
                '<div class="health-value" style="font-size:0.88rem;">all-MiniLM-L6-v2 (local)</div>'
                '<div class="small-muted" style="margin-top:0.3rem;">sentence-transformers · runs fully on-device</div>'
                '</div>',
                unsafe_allow_html=True,
            )
    except Exception as e:
        st.error(f"Could not connect to ChromaDB: {e}")

    st.markdown("---")

    # Dataset coverage
    st.markdown("**Sample Dataset Coverage**")
    coverage = base_df.groupby("sector").agg(
        records=("headline", "count"),
        bullish=("sentiment", lambda x: (x == "Bullish").sum()),
        bearish=("sentiment", lambda x: (x == "Bearish").sum()),
        avg_score=("score", "mean"),
    ).reset_index()
    coverage["avg_score"] = coverage["avg_score"].round(2)
    st.dataframe(coverage, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("**Re-index Actions**")
    col_idx, col_clear, col_reset = st.columns(3)
    with col_idx:
        if st.button("Index current sector into vector store"):
            with st.spinner(f"Indexing {sector}…"):
                n = index_dataframe(sector_df)
                st.session_state.indexed_sectors[sector] = n
            st.success(f"Indexed {n} records for {sector}.")
    with col_clear:
        if st.button("Clear session index cache"):
            st.session_state.indexed_sectors = {}
            st.success("Session index cache cleared. Sectors will re-index on next Copilot visit.")
    with col_reset:
        if st.button("Reset vector store", type="secondary"):
            with st.spinner("Resetting ChromaDB collection…"):
                delete_collection()
                st.session_state.indexed_sectors = {}
            st.success("Vector store reset. All embeddings cleared — re-index to restore.")

# ── Page: Product Roadmap ──────────────────────────────────────────────────────

elif page == "Product Roadmap":
    page_header(
        "Product Roadmap",
        "Platform evolution from MVP to enterprise-grade alternative data intelligence.",
        "Platform Vision",
    )

    st.markdown("**Platform phases**")
    p1, p2, p3 = st.columns(3)

    with p1:
        st.markdown(
            """<div class="timeline-card">
                <span class="phase-badge phase-done">✓ MVP 1 · Live Now</span>
                <div class="phase-title">Foundation</div>
                <div class="phase-sub">Core intelligence + explainability layer</div>
                <div class="roadmap-item">✓ Multi-source ingestion (News, Price, RSS)</div>
                <div class="roadmap-item">✓ Keyword-based sentiment scoring</div>
                <div class="roadmap-item">✓ ChromaDB RAG vector store</div>
                <div class="roadmap-item">✓ GPT-4o-mini executive summaries</div>
                <div class="roadmap-item">✓ Source-grounded Research Copilot</div>
                <div class="roadmap-item">✓ Compliance + governance disclaimers</div>
                <div class="roadmap-item">✓ Executive brief generator</div>
                <div class="roadmap-item">✓ Role-oriented navigation (6 personas)</div>
            </div>""",
            unsafe_allow_html=True,
        )

    with p2:
        st.markdown(
            """<div class="timeline-card">
                <span class="phase-badge phase-next">→ MVP 2 · Next Quarter</span>
                <div class="phase-title">Depth</div>
                <div class="phase-sub">Richer signals + analyst workflow integration</div>
                <div class="roadmap-item">○ Reddit &amp; social sentiment ingestion</div>
                <div class="roadmap-item">○ SEC / SEDAR filings parser</div>
                <div class="roadmap-item">○ Earnings call transcript summarization</div>
                <div class="roadmap-item">○ ML-based sentiment model (FinBERT)</div>
                <div class="roadmap-item">○ Signal anomaly detection &amp; alerts</div>
                <div class="roadmap-item">○ Watchlist &amp; sector subscription</div>
                <div class="roadmap-item">○ Analyst feedback loop (thumbs up/down)</div>
                <div class="roadmap-item">○ PDF/Excel export of executive briefs</div>
            </div>""",
            unsafe_allow_html=True,
        )

    with p3:
        st.markdown(
            """<div class="timeline-card">
                <span class="phase-badge phase-future">◇ MVP 3 · H2 Vision</span>
                <div class="phase-title">Enterprise</div>
                <div class="phase-sub">Multi-agent · governed · bank-grade</div>
                <div class="roadmap-item">◇ Multi-agent orchestration (LangGraph)</div>
                <div class="roadmap-item">◇ Real-time streaming data pipeline</div>
                <div class="roadmap-item">◇ Model evaluation &amp; explainability scoring</div>
                <div class="roadmap-item">◇ Enterprise SSO + RBAC permissions</div>
                <div class="roadmap-item">◇ Audit trail &amp; model risk governance</div>
                <div class="roadmap-item">◇ Integration: Bloomberg, Refinitiv, MSCI</div>
                <div class="roadmap-item">◇ Client-facing API + embedded widget</div>
                <div class="roadmap-item">◇ On-premise / private cloud deployment</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("**Alternative data landscape**")
    st.caption("True alternative data goes far beyond news — here is the full signal universe MarketPulse AI is designed to ingest.")

    alt_data = pd.DataFrame([
        ["News & Media",        "NewsAPI, Reuters, Bloomberg",       "Sentiment, event detection, macro narrative",  "✅ Live"],
        ["Market Prices",       "Alpha Vantage, Yahoo Finance",      "Price signals, volatility, momentum",          "✅ Live"],
        ["RSS Feeds",           "Reuters, FT, Globe & Mail",         "No-key news ingestion, topic tracking",        "✅ Live"],
        ["Social Media",        "Reddit (WSB, investing), Twitter/X","Retail sentiment, crowdsourced signals",        "📋 MVP 2"],
        ["Regulatory Filings",  "SEC EDGAR, SEDAR+",                 "Material events, insider activity",            "📋 MVP 2"],
        ["Earnings Transcripts","FactSet, Refinitiv",                "Management tone, forward guidance signals",    "📋 MVP 2"],
        ["Job Postings",        "LinkedIn, Indeed, Glassdoor",       "Hiring signals → revenue / expansion proxy",   "🔭 MVP 3"],
        ["Satellite Imagery",   "Orbital Insight, Descartes Labs",   "Physical activity: retail foot traffic, oil",  "🔭 MVP 3"],
        ["Credit Card Data",    "Yodlee, Plaid, Mastercard SpendingPulse", "Consumer spending by category",         "🔭 MVP 3"],
        ["Web Traffic",         "SimilarWeb, Semrush",               "Digital growth, product adoption signals",     "🔭 MVP 3"],
        ["Supply Chain",        "Panjiva, ImportGenius",             "Trade flows, inventory stress, logistics",     "🔭 MVP 3"],
    ], columns=["Data Type", "Example Sources", "Capital Markets Application", "Status"])

    st.dataframe(alt_data, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("**Success metrics vision**")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        metric_card("Insight generation", "< 30 sec", "vs. 2–4 hrs manual research", badge="⚡ Speed", badge_type="ok")
    with kpi2:
        metric_card("Signal source types", "9 types", "News · Price · RSS · ESG · Earnings · Alt Data…", badge="↑ Coverage", badge_type="ok")
    with kpi3:
        metric_card("Analyst time saved", "~40%", "estimated per intelligence workflow", badge="↑ Efficiency", badge_type="ok")
    with kpi4:
        metric_card("Evidence grounding", "100%", "every GPT answer cited to source document", badge="✓ Quality", badge_type="ok")

    kpi5, kpi6, kpi7, kpi8 = st.columns(4)
    with kpi5:
        metric_card("Sectors monitored", "8 live", "Banking · AI · FinTech · Energy · Healthcare…", badge="◎ Breadth", badge_type="ok")
    with kpi6:
        metric_card("RAG search latency", "< 2 sec", "ChromaDB vector retrieval at scale", badge="⚡ Speed", badge_type="live")
    with kpi7:
        metric_card("Compliance layer", "Built-in", "AI disclaimer on every generated output", badge="🔒 Governance", badge_type="ok")
    with kpi8:
        metric_card("Deployment target", "Cloud-ready", "Streamlit · containerised · RBC-deployable", badge="◇ Scalable", badge_type="live")

# ── Global footer — rendered on every page ────────────────────────────────────
# Spacer so the fixed footer never overlaps page content
st.markdown('<div style="height:2.8rem;"></div>', unsafe_allow_html=True)

st.markdown(
    '<div style="'
    'position:fixed;bottom:0;left:0;right:0;'
    'background:linear-gradient(90deg,#0f172a 0%,#1a2540 100%);'
    'color:#94a3b8;text-align:center;'
    'padding:0.55rem 1rem;font-size:0.76rem;'
    'border-top:1px solid #1e2d47;z-index:9999;'
    'letter-spacing:0.02em;'
    '">'
    'Built with <span style="color:#ef4444;font-size:0.9rem;">♥</span> by <strong style="color:#f1f5f9;">Mahesh</strong>'
    '&nbsp;&nbsp;·&nbsp;&nbsp;MarketPulse AI'
    '&nbsp;&nbsp;·&nbsp;&nbsp;GenAI-powered Capital Markets Intelligence'
    '</div>',
    unsafe_allow_html=True,
)
