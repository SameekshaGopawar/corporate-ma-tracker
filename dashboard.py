import os
import re
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Corporate M&A Tracker",
    page_icon="briefcase",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0f1117; color: #e0e0e0;
}
[data-testid="stHeader"]  { background-color: #0f1117; }
[data-testid="stSidebar"] { background-color: #161b22; }

h1 { font-size:2rem!important; font-weight:700!important;
     letter-spacing:-0.5px; color:#ffffff!important; }
h2 { font-size:1.25rem!important; font-weight:600!important; color:#c9d1d9!important; }
h3 { font-size:0.8rem!important; font-weight:600!important; color:#8b949e!important;
     text-transform:uppercase; letter-spacing:1px; }

.kpi-card {
    background: linear-gradient(135deg,#161b22 0%,#1c2128 100%);
    border:1px solid #30363d; border-radius:12px;
    padding:20px 24px; text-align:center; transition:border-color 0.2s;
}
.kpi-card:hover { border-color:#58a6ff; }
.kpi-value { font-size:2rem; font-weight:700; color:#58a6ff;
             line-height:1.1; margin-bottom:4px; }
.kpi-label { font-size:0.72rem; color:#8b949e;
             text-transform:uppercase; letter-spacing:1.2px; }
.kpi-sub   { font-size:0.68rem; color:#58a6ff; margin-top:3px; opacity:0.7; }

.badge { display:inline-block; padding:2px 9px; border-radius:20px;
         font-size:0.72rem; font-weight:600; letter-spacing:0.3px; }
.badge-high   { background:#1a3a1a; color:#3fb950; border:1px solid #3fb950; }
.badge-medium { background:#2d2a0f; color:#d29922; border:1px solid #d29922; }

.deal-card {
    background:#1c2128; border:1px solid #30363d;
    border-left:3px solid #58a6ff; border-radius:8px;
    padding:10px 14px; margin-bottom:8px;
    display:flex; justify-content:space-between; align-items:center;
}
.deal-buyer  { font-weight:600; color:#e6edf3; font-size:0.88rem; }
.deal-target { color:#8b949e; font-size:0.78rem; margin-top:1px; }
.deal-value  { color:#3fb950; font-weight:600; font-size:0.82rem; white-space:nowrap; }
.deal-unknown { color:#484f58; font-style:italic; }
.deal-right  { display:flex; align-items:center; gap:10px; flex-shrink:0; }

[data-testid="stTextInput"] input,
[data-testid="stSelectbox"] > div {
    background-color:#1c2128!important; border:1px solid #30363d!important;
    color:#e0e0e0!important; border-radius:8px!important;
}
[data-testid="stButton"] > button {
    background:linear-gradient(135deg,#1f6feb,#388bfd);
    color:white; border:none; border-radius:8px;
    padding:8px 20px; font-weight:600; letter-spacing:0.3px;
    transition:opacity 0.2s;
}
[data-testid="stButton"] > button:hover { opacity:0.85; }
hr { border-color:#21262d!important; }
::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-track { background:#0f1117; }
::-webkit-scrollbar-thumb { background:#30363d; border-radius:3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# API KEYS  (Streamlit secrets → env fallback)
# ─────────────────────────────────────────────

def _secret(key):
    try:
        return st.secrets[key]
    except Exception:
        return os.environ.get(key, "")

NEWS_API_KEY = _secret("NEWS_API_KEY")
HF_API_TOKEN = _secret("HF_API_TOKEN")

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

CSV_PATH = Path(__file__).parent / "ma_deals.csv"

HF_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
HF_LABELS  = ["Acquisition", "Partnership", "Investment", "Other"]

NEWS_QUERIES = ["company acquisition", "acquires", "merger deal", "buyout", "acquired"]

KNOWN_DOMAINS = {
    "google": "google.com", "microsoft": "microsoft.com",
    "amazon": "amazon.com", "apple": "apple.com",
    "meta": "meta.com", "salesforce": "salesforce.com",
    "oracle": "oracle.com", "sap": "sap.com",
    "ibm": "ibm.com", "cisco": "cisco.com",
    "adobe": "adobe.com", "teva": "tevapharm.com",
}

BAD_NAME_PATTERNS = re.compile(
    r"(acquisition|merger|buyout|purchase|takeover|deal|"
    r"billion|million|\$|\d+bn|\d+m\b)",
    re.IGNORECASE,
)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def clean_entity(name):
    if not name or str(name).strip().lower() in ("unknown", "none", "nan", ""):
        return "Unknown"
    if BAD_NAME_PATTERNS.search(str(name)):
        return "Unknown"
    if str(name)[0].isdigit():
        return "Unknown"
    return str(name).strip()


def parse_deal_value(value_str):
    s = str(value_str).lower().replace(",", "")
    match = re.search(r"\$?([\d.]+)\s*(b|bn|billion|m|mn|million)?", s)
    if not match:
        return None
    number = float(match.group(1))
    unit   = match.group(2) or ""
    if unit.startswith("b"): return number
    if unit.startswith("m"): return round(number / 1000, 4)
    return number


def find_largest_deal(df):
    best_val, best_str = -1, "N/A"
    for v in df["Deal_Value"]:
        if v == "Unknown":
            continue
        parsed = parse_deal_value(v)
        if parsed is not None and parsed > best_val:
            best_val, best_str = parsed, v
    return best_str


def confidence_badge(score):
    score = float(score)
    if score >= 85:
        return f'<span class="badge badge-high">{score:.0f}% High</span>'
    return f'<span class="badge badge-medium">{score:.0f}% Med</span>'


def avatar_html(buyer_name):
    key    = buyer_name.lower().split()[0] if buyer_name != "Unknown" else ""
    domain = KNOWN_DOMAINS.get(key)
    initials = "".join(w[0].upper() for w in buyer_name.split()[:2]) \
               if buyer_name != "Unknown" else "?"
    if domain:
        return (
            f'<img src="https://logo.clearbit.com/{domain}" width="28" height="28" '
            f'style="border-radius:5px;vertical-align:middle;background:#fff;padding:2px;" '
            f'onerror="this.style.display=\'none\'">'
        )
    colours = ["#1f6feb", "#388bfd", "#58a6ff", "#2ea043", "#d29922"]
    colour  = colours[sum(ord(c) for c in initials) % len(colours)]
    return (
        f'<span style="display:inline-flex;align-items:center;justify-content:center;'
        f'width:28px;height:28px;border-radius:5px;background:{colour};color:#fff;'
        f'font-size:0.7rem;font-weight:700;vertical-align:middle;">{initials}</span>'
    )

# ─────────────────────────────────────────────
# LIVE PIPELINE (no spaCy — regex + HF API)
# ─────────────────────────────────────────────

# Patterns: (buyer_group, target_group) from headline text
_PATTERNS = [
    # "X acquires / acquired / buys / bought Y"
    re.compile(
        r"^(?P<buyer>[A-Z][A-Za-z0-9& .,']+?)\s+"
        r"(?:acquires?|acquired|buys?|bought|purchases?|purchased)\s+"
        r"(?P<target>[A-Z][A-Za-z0-9& .,']+)",
        re.IGNORECASE,
    ),
    # "X to acquire Y" / "X plans to acquire Y"
    re.compile(
        r"^(?P<buyer>[A-Z][A-Za-z0-9& .,']+?)\s+"
        r"(?:plans?\s+to|agrees?\s+to|set\s+to|moves?\s+to)?\s*"
        r"acquire\s+(?P<target>[A-Z][A-Za-z0-9& .,']+)",
        re.IGNORECASE,
    ),
    # "X closes / completes / announces acquisition of Y"
    re.compile(
        r"^(?P<buyer>[A-Z][A-Za-z0-9& .,']+?)\s+"
        r"(?:closes?|completes?|announces?|finalizes?|seals?)\s+"
        r"(?:\w+\s+)?acquisition\s+of\s+(?P<target>[A-Z][A-Za-z0-9& .,']+)",
        re.IGNORECASE,
    ),
    # "acquisition of Y by X"
    re.compile(
        r"acquisition\s+of\s+(?P<target>[A-Z][A-Za-z0-9& .,']+?)"
        r"\s+by\s+(?P<buyer>[A-Z][A-Za-z0-9& .,']+)",
        re.IGNORECASE,
    ),
]

_VALUE_RE = re.compile(
    r"\$[\d.,]+\s*(?:billion|million|bn|mn|[bm])\b"
    r"|\b[\d.,]+\s*(?:billion|million|bn|mn)\b",
    re.IGNORECASE,
)

_STOP_WORDS = {
    "the", "a", "an", "its", "their", "this", "that", "and", "or",
    "in", "of", "to", "for", "on", "at", "by", "with", "from",
    "deal", "shares", "stake", "unit", "inc", "corp", "ltd",
}


def _trim(text):
    """Remove trailing filler words from extracted entity."""
    text = re.sub(r"\s+(for|in|to|of|and|with|at|from|as|a|an|the)$",
                  "", text.strip(), flags=re.IGNORECASE)
    words = text.split()
    filtered = [w for w in words if w.lower() not in _STOP_WORDS or len(words) == 1]
    return " ".join(filtered).strip() or text.strip()


def extract_entities(headline):
    buyer, target, deal_value = "Unknown", "Unknown", "Unknown"

    for pat in _PATTERNS:
        m = pat.search(headline)
        if m:
            try:    buyer  = _trim(m.group("buyer"))
            except Exception: pass
            try:    target = _trim(m.group("target"))
            except Exception: pass
            break

    vm = _VALUE_RE.search(headline)
    if vm:
        deal_value = vm.group(0).strip()

    return clean_entity(buyer), clean_entity(target), deal_value


def classify_headline(headline):
    """Call HF zero-shot API. Returns (label, score_pct) or (None, 0)."""
    if not HF_API_TOKEN:
        return None, 0
    try:
        resp = requests.post(
            HF_API_URL,
            headers={"Authorization": f"Bearer {HF_API_TOKEN}"},
            json={"inputs": headline, "parameters": {"candidate_labels": HF_LABELS}},
            timeout=20,
        )
        data = resp.json()
        if isinstance(data, list):
            data = data[0]
        if "labels" in data and "scores" in data:
            top_idx   = data["scores"].index(max(data["scores"]))
            top_label = data["labels"][top_idx]
            scores    = {l: s * 100 for l, s in zip(data["labels"], data["scores"])}
            return top_label, scores
    except Exception:
        pass
    return None, {}


def fetch_headlines():
    """Pull headlines from NewsAPI across multiple queries."""
    if not NEWS_API_KEY:
        return []
    seen, results = set(), []
    for q in NEWS_QUERIES:
        try:
            resp = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": q, "language": "en", "pageSize": 20,
                    "sortBy": "publishedAt", "apiKey": NEWS_API_KEY,
                },
                timeout=10,
            )
            for art in resp.json().get("articles", []):
                title = (art.get("title") or "").strip()
                if title and title not in seen:
                    seen.add(title)
                    results.append(title)
        except Exception:
            pass
    return results


def run_live_pipeline():
    """Fetch headlines, classify, extract — return DataFrame."""
    headlines = fetch_headlines()
    if not headlines:
        return pd.DataFrame()

    rows = []
    progress = st.progress(0, text="Fetching & classifying headlines…")

    for i, headline in enumerate(headlines):
        progress.progress((i + 1) / len(headlines),
                          text=f"Processing {i+1}/{len(headlines)}…")

        label, scores = classify_headline(headline)
        if label != "Acquisition":
            continue
        acq_score = scores.get("Acquisition", 0)
        if acq_score < 70:
            continue

        buyer, target, deal_value = extract_entities(headline)
        rows.append({
            "Buyer":      buyer,
            "Target":     target,
            "Deal_Value": deal_value,
            "Category":   "Acquisition",
            "Acq_%":      round(acq_score, 1),
            "Partner_%":  round(scores.get("Partnership", 0), 1),
            "Invest_%":   round(scores.get("Investment", 0), 1),
            "Other_%":    round(scores.get("Other", 0), 1),
            "Headline":   headline,
        })
        time.sleep(0.3)   # avoid HF rate-limit

    progress.empty()
    return pd.DataFrame(rows) if rows else pd.DataFrame()

# ─────────────────────────────────────────────
# DATA LOADER  (CSV fallback)
# ─────────────────────────────────────────────

@st.cache_data
def load_csv():
    if not CSV_PATH.exists():
        return pd.DataFrame(columns=[
            "Buyer", "Target", "Deal_Value", "Category",
            "Acq_%", "Partner_%", "Invest_%", "Other_%"
        ])
    df = pd.read_csv(CSV_PATH)
    for col in ("Buyer", "Target", "Deal_Value"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    df["Buyer"]  = df["Buyer"].apply(clean_entity)
    df["Target"] = df["Target"].apply(clean_entity)
    df["Deal_Value_Num"] = df["Deal_Value"].apply(
        lambda v: parse_deal_value(v) if v != "Unknown" else None)
    if "Acq_%" not in df.columns:
        df["Acq_%"] = 0.0
    return df

# ─────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────

DARK = dict(
    paper_bgcolor="#161b22", plot_bgcolor="#161b22",
    font=dict(color="#c9d1d9", family="Inter,sans-serif", size=12),
    margin=dict(l=20, r=20, t=40, b=20),
    hoverlabel=dict(bgcolor="#1c2128", bordercolor="#30363d",
                    font=dict(color="#e6edf3")),
)


def chart_by_buyer(df):
    known = df[df["Buyer"] != "Unknown"]
    if known.empty: return None
    counts = (known.groupby("Buyer").size()
                   .reset_index(name="Deals")
                   .sort_values("Deals", ascending=True))
    fig = px.bar(counts, x="Deals", y="Buyer", orientation="h",
                 title="Acquisition Volume by Buyer",
                 color="Deals",
                 color_continuous_scale=[[0, "#1f6feb"], [1, "#58a6ff"]])
    fig.update_layout(**DARK, showlegend=False, coloraxis_showscale=False)
    fig.update_traces(marker_line_width=0,
                      hovertemplate="<b>%{y}</b><br>Deals: %{x}<extra></extra>")
    fig.update_yaxes(gridcolor="#21262d", tickfont=dict(size=11))
    fig.update_xaxes(gridcolor="#21262d", dtick=1)
    return fig


def chart_donut(df):
    if df.empty or "Acq_%" not in df.columns:
        return go.Figure()
    bins, labels = [70, 80, 90, 101], ["70-79%", "80-89%", "90-100%"]
    counts = pd.cut(df["Acq_%"], bins=bins, labels=labels,
                    right=False).value_counts().sort_index()
    fig = go.Figure(go.Pie(
        labels=counts.index, values=counts.values, hole=0.6,
        marker=dict(colors=["#d29922", "#1f6feb", "#3fb950"],
                    line=dict(color="#0f1117", width=2)),
        textfont=dict(color="#e6edf3", size=12),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<extra></extra>",
    ))
    fig.update_layout(
        **DARK, title="Confidence Distribution",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25,
                    font=dict(color="#c9d1d9")),
        annotations=[dict(text=f"<b>{len(df)}</b><br>deals",
                          x=0.5, y=0.5, font=dict(size=14, color="#e6edf3"),
                          showarrow=False)],
    )
    return fig


def chart_treemap(df):
    known = df[df["Buyer"] != "Unknown"]
    if known.empty: return None
    counts = known.groupby("Buyer").size().reset_index(name="Deals")
    fig = px.treemap(counts, path=["Buyer"], values="Deals",
                     title="Top Buyers Treemap",
                     color="Deals",
                     color_continuous_scale=[[0, "#1f6feb"], [1, "#3fb950"]])
    fig.update_layout(**DARK, coloraxis_showscale=False)
    fig.update_traces(
        textfont=dict(color="#ffffff", size=13),
        hovertemplate="<b>%{label}</b><br>Deals: %{value}<extra></extra>",
        marker=dict(line=dict(color="#0f1117", width=2)),
    )
    return fig

# ─────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────

# ── Header ───────────────────────────────────

hL, hR = st.columns([3, 1])
with hL:
    st.markdown("## Corporate M&A Tracker")
    st.markdown(
        '<p style="color:#8b949e;font-size:0.82rem;margin-top:-10px;">'
        'Real-time acquisition intelligence &nbsp;|&nbsp; '
        'NewsAPI &bull; spaCy NLP &bull; Hugging Face AI</p>',
        unsafe_allow_html=True,
    )
with hR:
    st.markdown("<br>", unsafe_allow_html=True)
    btn_cols = st.columns(2)
    fetch_clicked  = btn_cols[0].button("Fetch Live Data", use_container_width=True, type="primary")
    reload_clicked = btn_cols[1].button("Reload CSV",      use_container_width=True)

st.divider()

# ── Session state ─────────────────────────────

if "live_df" not in st.session_state:
    st.session_state.live_df = None

# ── Fetch live data ───────────────────────────

if fetch_clicked:
    keys_ok = bool(NEWS_API_KEY and HF_API_TOKEN)
    if not keys_ok:
        st.error(
            "API keys not found. Add NEWS_API_KEY and HF_API_TOKEN "
            "in Streamlit Cloud → App Settings → Secrets."
        )
    else:
        with st.spinner("Fetching live headlines and running AI classification… (~60s)"):
            live = run_live_pipeline()
        if live.empty:
            st.warning("No acquisition deals found in today's headlines. Try again later.")
        else:
            st.session_state.live_df = live
            st.success(f"Found {len(live)} acquisition deals from live news!")

if reload_clicked:
    st.cache_data.clear()
    st.session_state.live_df = None
    st.rerun()

# ── Choose data source ────────────────────────

if st.session_state.live_df is not None:
    df = st.session_state.live_df.copy()
    st.markdown(
        '<p style="color:#3fb950;font-size:0.8rem;margin-bottom:8px;">'
        'Showing live data fetched this session.</p>',
        unsafe_allow_html=True,
    )
else:
    df = load_csv()

if df.empty:
    st.warning("No deal data available. Click **Fetch Live Data** to pull today's headlines.")
    st.stop()

# ── KPI cards ────────────────────────────────────

total_deals   = len(df)
largest_deal  = find_largest_deal(df)
unique_buyers = df["Buyer"][df["Buyer"] != "Unknown"].nunique()
avg_conf      = df["Acq_%"].mean() if "Acq_%" in df.columns else 0

kpi_html = """
<div class="kpi-card">
  <div class="kpi-value">{value}</div>
  <div class="kpi-label">{label}</div>
  <div class="kpi-sub">{sub}</div>
</div>
"""

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(kpi_html.format(
        value=total_deals, label="Total Deals Found",
        sub="classified as Acquisition"), unsafe_allow_html=True)
with k2:
    st.markdown(kpi_html.format(
        value=largest_deal, label="Largest Deal Value",
        sub="highest single transaction"), unsafe_allow_html=True)
with k3:
    st.markdown(kpi_html.format(
        value=unique_buyers, label="Unique Buyers",
        sub="distinct acquiring companies"), unsafe_allow_html=True)
with k4:
    st.markdown(kpi_html.format(
        value=f"{avg_conf:.0f}%", label="Avg Confidence",
        sub="across all Acquisition labels"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Charts ─────────────────────────────────────

cL, cR = st.columns([3, 2])
with cL:
    fig = chart_by_buyer(df)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No named buyers in current data.")
with cR:
    st.plotly_chart(chart_donut(df), use_container_width=True)

fig_tree = chart_treemap(df)
if fig_tree:
    st.plotly_chart(fig_tree, use_container_width=True)

st.divider()

# ── Filters ────────────────────────────────────

st.markdown("### Filters & Search")
st.markdown("<br>", unsafe_allow_html=True)

fc1, fc2, fc3 = st.columns([2, 2, 3])
with fc1:
    buyer_opts = ["All"] + sorted(
        df["Buyer"][df["Buyer"] != "Unknown"].dropna().unique().tolist())
    sel_buyer = st.selectbox("Buyer", buyer_opts)
with fc2:
    target_opts = ["All"] + sorted(
        df["Target"][df["Target"] != "Unknown"].dropna().unique().tolist())
    sel_target = st.selectbox("Target", target_opts)
with fc3:
    search = st.text_input("Search", placeholder="Search buyer, target, deal value...")

filtered = df.copy()
if sel_buyer  != "All": filtered = filtered[filtered["Buyer"]  == sel_buyer]
if sel_target != "All": filtered = filtered[filtered["Target"] == sel_target]
if search:
    mask = filtered.apply(
        lambda row: row.astype(str).str.contains(search, case=False, na=False).any(),
        axis=1)
    filtered = filtered[mask]

st.markdown("<br>", unsafe_allow_html=True)

# ── Deal cards ─────────────────────────────────

count_label = f'{len(filtered)} result{"s" if len(filtered) != 1 else ""}'
st.markdown(
    f'### Recent Acquisitions &nbsp;'
    f'<span style="color:#58a6ff;font-size:0.85rem;font-weight:400;'
    f'text-transform:none;">{count_label}</span>',
    unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

if filtered.empty:
    st.markdown(
        '<p style="color:#8b949e;text-align:center;padding:30px 0;">'
        'No deals match your filters.</p>', unsafe_allow_html=True)
else:
    rows = list(filtered.iterrows())
    for i in range(0, len(rows), 2):
        col_a, col_b = st.columns(2)
        for col, idx in zip([col_a, col_b], [i, i + 1]):
            if idx >= len(rows): break
            _, row = rows[idx]
            buyer  = row.get("Buyer",      "Unknown")
            target = row.get("Target",     "Unknown")
            value  = row.get("Deal_Value", "Unknown")
            score  = row.get("Acq_%", 0)

            buyer_html  = f'<span class="deal-buyer">{buyer}</span>' \
                          if buyer  != "Unknown" else \
                          '<span class="deal-buyer deal-unknown">Buyer unknown</span>'
            target_html = f'<span class="deal-target">Target: {target}</span>' \
                          if target != "Unknown" else \
                          '<span class="deal-target deal-unknown">Target unknown</span>'
            value_html  = f'<span class="deal-value">{value}</span>' \
                          if value  != "Unknown" else \
                          '<span class="deal-value deal-unknown">Undisclosed</span>'

            # show headline if available (live data)
            headline_html = ""
            if "Headline" in row and pd.notna(row["Headline"]):
                hl = str(row["Headline"])[:100] + ("…" if len(str(row["Headline"])) > 100 else "")
                headline_html = (
                    f'<div style="color:#484f58;font-size:0.7rem;'
                    f'margin-top:4px;font-style:italic;">{hl}</div>'
                )

            with col:
                st.markdown(f"""
                <div class="deal-card">
                  <div style="flex:1;min-width:0;">
                    <div style="display:flex;align-items:center;gap:10px;">
                      {avatar_html(buyer)}
                      <div>{buyer_html}{target_html}</div>
                    </div>
                    {headline_html}
                  </div>
                  <div class="deal-right">
                    {value_html}
                    {confidence_badge(score)}
                  </div>
                </div>
                """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.divider()

# ── Raw data expander ──────────────────────────

with st.expander("View raw data table"):
    show_cols = [c for c in
                 ["Buyer", "Target", "Deal_Value", "Acq_%",
                  "Partner_%", "Invest_%", "Other_%", "Headline"]
                 if c in filtered.columns]
    st.dataframe(filtered[show_cols], use_container_width=True, hide_index=True)

# ── Footer ─────────────────────────────────────

st.markdown(
    '<p style="text-align:center;color:#484f58;font-size:0.72rem;padding:8px 0 2px;">'
    'Corporate M&A Tracker &nbsp;&bull;&nbsp; NewsAPI &bull; Hugging Face AI '
    '&nbsp;&bull;&nbsp; Acquisition >= 70% confidence only</p>',
    unsafe_allow_html=True)
