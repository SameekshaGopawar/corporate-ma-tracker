import re
import subprocess
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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

/* Compact deal card */
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
# CONSTANTS
# ─────────────────────────────────────────────

CSV_PATH = "ma_deals.csv"
PIPELINE = "ma_tracker.py"

KNOWN_DOMAINS = {
    "google": "google.com", "microsoft": "microsoft.com",
    "amazon": "amazon.com", "apple": "apple.com",
    "meta": "meta.com", "salesforce": "salesforce.com",
    "oracle": "oracle.com", "sap": "sap.com",
    "ibm": "ibm.com", "cisco": "cisco.com",
    "adobe": "adobe.com", "teva": "tevapharm.com",
    "trucordia": "trucordia.com", "nagase": "nagase.com",
}

# Words that indicate the NLP grabbed a deal descriptor instead of a company name
BAD_NAME_PATTERNS = re.compile(
    r"(acquisition|merger|buyout|purchase|takeover|deal|"
    r"billion|million|\$|\d+bn|\d+m\b)",
    re.IGNORECASE,
)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def clean_entity(name):
    """
    Return the name as-is if it looks like a real company.
    Return 'Unknown' if it looks like a price, a deal noun,
    or starts with a digit — meaning the NLP made a mistake.
    Examples rejected: '55bn acquisition', '$32 billion', '10 Boeing'
    """
    if not name or name.strip().lower() in ("unknown", "none", "nan", ""):
        return "Unknown"
    if BAD_NAME_PATTERNS.search(name):
        return "Unknown"
    if name[0].isdigit():
        return "Unknown"
    return name.strip()


def parse_deal_value(value_str):
    s = str(value_str).lower().replace(",", "")
    match = re.search(r"\$?([\d.]+)\s*(b|bn|billion|m|mn|million)?", s)
    if not match:
        return None
    number = float(match.group(1))
    unit   = match.group(2) or ""
    if unit.startswith("b"):   return number
    if unit.startswith("m"):   return round(number / 1000, 4)
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
    return     f'<span class="badge badge-medium">{score:.0f}% Med</span>'


def logo_url(buyer_name):
    key = buyer_name.lower().split()[0] if buyer_name != "Unknown" else ""
    domain = KNOWN_DOMAINS.get(key)
    return f"https://logo.clearbit.com/{domain}" if domain else None


def avatar_html(buyer_name):
    url      = logo_url(buyer_name)
    initials = "".join(w[0].upper() for w in buyer_name.split()[:2]) \
               if buyer_name != "Unknown" else "?"
    if url:
        return (
            f'<img src="{url}" width="28" height="28" '
            f'style="border-radius:5px;vertical-align:middle;'
            f'background:#fff;padding:2px;" '
            f'onerror="this.style.display=\'none\'">'
        )
    colours = ["#1f6feb","#388bfd","#58a6ff","#2ea043","#d29922"]
    colour  = colours[sum(ord(c) for c in initials) % len(colours)]
    return (
        f'<span style="display:inline-flex;align-items:center;'
        f'justify-content:center;width:28px;height:28px;border-radius:5px;'
        f'background:{colour};color:#fff;font-size:0.7rem;font-weight:700;'
        f'vertical-align:middle;">{initials}</span>'
    )

# ─────────────────────────────────────────────
# DATA LOADER
# ─────────────────────────────────────────────

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        st.error(f"Could not find {CSV_PATH}. Run ma_tracker.py first.")
        st.stop()

    for col in ("Buyer", "Target", "Deal_Value"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # ── Fix bad NLP extractions ──────────────────
    # If the NLP grabbed a deal descriptor (e.g. "55bn acquisition")
    # instead of a real company name, replace it with "Unknown".
    df["Buyer"]  = df["Buyer"].apply(clean_entity)
    df["Target"] = df["Target"].apply(clean_entity)

    df["Deal_Value_Num"] = df["Deal_Value"].apply(
        lambda v: parse_deal_value(v) if v != "Unknown" else None
    )
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
    if known.empty:
        return None
    counts = (known.groupby("Buyer").size()
                   .reset_index(name="Deals")
                   .sort_values("Deals", ascending=True))
    fig = px.bar(counts, x="Deals", y="Buyer", orientation="h",
                 title="Acquisition Volume by Buyer",
                 color="Deals",
                 color_continuous_scale=[[0,"#1f6feb"],[1,"#58a6ff"]])
    fig.update_layout(**DARK, showlegend=False, coloraxis_showscale=False)
    fig.update_traces(marker_line_width=0,
                      hovertemplate="<b>%{y}</b><br>Deals: %{x}<extra></extra>")
    fig.update_yaxes(gridcolor="#21262d", tickfont=dict(size=11))
    fig.update_xaxes(gridcolor="#21262d", dtick=1)
    return fig


def chart_donut(df):
    bins, labels = [70,80,90,101], ["70-79%","80-89%","90-100%"]
    counts = pd.cut(df["Acq_%"], bins=bins, labels=labels,
                    right=False).value_counts().sort_index()
    fig = go.Figure(go.Pie(
        labels=counts.index, values=counts.values, hole=0.6,
        marker=dict(colors=["#d29922","#1f6feb","#3fb950"],
                    line=dict(color="#0f1117", width=2)),
        textfont=dict(color="#e6edf3", size=12),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<extra></extra>",
    ))
    fig.update_layout(
        **DARK, title="Confidence Distribution",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25,
                    font=dict(color="#c9d1d9")),
        annotations=[dict(text=f"<b>{len(df)}</b><br>deals",
                          x=0.5, y=0.5, font=dict(size=14,color="#e6edf3"),
                          showarrow=False)],
    )
    return fig


def chart_treemap(df):
    known = df[df["Buyer"] != "Unknown"]
    if known.empty:
        return None
    counts = known.groupby("Buyer").size().reset_index(name="Deals")
    fig = px.treemap(counts, path=["Buyer"], values="Deals",
                     title="Top Buyers Treemap",
                     color="Deals",
                     color_continuous_scale=[[0,"#1f6feb"],[1,"#3fb950"]])
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
    # Two buttons side by side
    b1, b2 = st.columns(2)
    with b1:
        fetch = st.button("Fetch New Data", use_container_width=True, type="primary")
    with b2:
        reload = st.button("Reload CSV", use_container_width=True)

st.divider()

# ── Button actions ───────────────────────────

if fetch:
    # Run the full pipeline (NewsAPI → NLP → CSV) then reload
    with st.spinner("Fetching live headlines and running NLP pipeline... (this takes ~60s)"):
        result = subprocess.run(
            ["python", PIPELINE],
            capture_output=True, text=True,
            cwd="C:/Users/Sameeksha/Desktop/MA_tracker"
        )
    if result.returncode == 0:
        st.success("Pipeline complete! New deals loaded.")
    else:
        st.error(f"Pipeline error: {result.stderr[-500:] if result.stderr else 'unknown'}")
    st.cache_data.clear()
    st.rerun()

if reload:
    # Just re-read the existing CSV without re-running the pipeline
    st.cache_data.clear()
    st.rerun()

# ── Load & clean data ─────────────────────────

df = load_data()

# ── KPI cards ────────────────────────────────

total_deals   = len(df)
largest_deal  = find_largest_deal(df)
unique_buyers = df["Buyer"][df["Buyer"] != "Unknown"].nunique()
avg_conf      = df["Acq_%"].mean() if not df.empty else 0

kpi_html = """
<div class="kpi-card">
  <div class="kpi-value">{value}</div>
  <div class="kpi-label">{label}</div>
  <div class="kpi-sub">{sub}</div>
</div>
"""

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(kpi_html.format(value=total_deals,
        label="Total Deals Found", sub="classified as Acquisition"),
        unsafe_allow_html=True)
with k2:
    st.markdown(kpi_html.format(value=largest_deal,
        label="Largest Deal Value", sub="highest single transaction"),
        unsafe_allow_html=True)
with k3:
    st.markdown(kpi_html.format(value=unique_buyers,
        label="Unique Buyers", sub="distinct acquiring companies"),
        unsafe_allow_html=True)
with k4:
    st.markdown(kpi_html.format(value=f"{avg_conf:.0f}%",
        label="Avg Confidence", sub="across all Acquisition labels"),
        unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Charts ────────────────────────────────────

cL, cR = st.columns([3, 2])
with cL:
    fig = chart_by_buyer(df)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No named buyers yet — run Fetch New Data.")
with cR:
    st.plotly_chart(chart_donut(df), use_container_width=True)

fig_tree = chart_treemap(df)
if fig_tree:
    st.plotly_chart(fig_tree, use_container_width=True)

st.divider()

# ── Filters ──────────────────────────────────

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

# Apply filters
filtered = df.copy()
if sel_buyer  != "All": filtered = filtered[filtered["Buyer"]  == sel_buyer]
if sel_target != "All": filtered = filtered[filtered["Target"] == sel_target]
if search:
    mask = filtered.apply(
        lambda row: row.astype(str).str.contains(search, case=False, na=False).any(),
        axis=1)
    filtered = filtered[mask]

st.markdown("<br>", unsafe_allow_html=True)

# ── Recent acquisitions — 2-column compact cards ──

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
    # Render in 2 columns
    for i in range(0, len(rows), 2):
        col_a, col_b = st.columns(2)
        for col, idx in zip([col_a, col_b], [i, i+1]):
            if idx >= len(rows):
                break
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

            with col:
                st.markdown(f"""
                <div class="deal-card">
                  <div style="display:flex;align-items:center;gap:10px;">
                    {avatar_html(buyer)}
                    <div>
                      {buyer_html}
                      {target_html}
                    </div>
                  </div>
                  <div class="deal-right">
                    {value_html}
                    {confidence_badge(score)}
                  </div>
                </div>
                """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.divider()

# ── Raw data expander ─────────────────────────

with st.expander("View raw data table"):
    show_cols = [c for c in
                 ["Buyer","Target","Deal_Value","Acq_%","Partner_%","Invest_%","Other_%"]
                 if c in filtered.columns]
    st.dataframe(filtered[show_cols], use_container_width=True, hide_index=True)

# ── Footer ─────────────────────────────────────

st.markdown(
    '<p style="text-align:center;color:#484f58;font-size:0.72rem;padding:8px 0 2px;">'
    'Corporate M&A Tracker &nbsp;&bull;&nbsp; NewsAPI &bull; spaCy &bull; '
    'Hugging Face &nbsp;&bull;&nbsp; Acquisition >= 70% confidence only</p>',
    unsafe_allow_html=True)
