import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, date, timedelta
import math

st.set_page_config(
    page_title="JetLearn MIS – Enrolments (MTD & Cohort) + Conversion%",
    page_icon="📊",
    layout="wide",
)

# ---------- Global UI styling ----------
st.markdown(
    """
    <style>
      .stAltairChart {
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 14px;
        background: #ffffff;
        box-shadow: 0 1px 3px rgba(15,23,42,.08);
      }
      .legend-pill {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        margin-right: 10px;
        font-weight: 600;
        font-size: 0.9rem;
        color: #111827;
      }
      .pill-total { background: #e5e7eb; }
      .pill-ai    { background: #bfdbfe; }
      .pill-math  { background: #bbf7d0; }

      .kpi-card {
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 12px 14px;
        background: #fafafa;
      }
      .kpi-title { color:#6b7280; font-size:.9rem; margin-bottom:6px; }
      .kpi-value { font-weight:700; font-size:1.4rem; color:#111827; }
      .kpi-sub   { color:#6b7280; font-size:.85rem; }
      .section-title {
        font-weight: 700;
        font-size: 1.05rem;
        margin-top: .25rem;
        margin-bottom: .25rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Color palette ----------
PALETTE = {
    "Total": "#6b7280",      # gray-500
    "AI Coding": "#2563eb",  # blue-600
    "Math": "#16a34a",       # green-600
    "ThresholdLow": "#f3f4f6",   # gray-100
    "ThresholdMid": "#e5e7eb",   # gray-200
    "ThresholdHigh": "#d1d5db",  # gray-300
}

# ----------------------------
# Helpers
# ----------------------------
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df

def find_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    low = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None

def coerce_datetime(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce", infer_datetime_format=True, dayfirst=True)
    if s.notna().sum() == 0:
        try:
            s = pd.to_datetime(series, errors="coerce", unit="s")
        except Exception:
            try:
                s = pd.to_datetime(series, errors="coerce", unit="ms")
            except Exception:
                pass
    return s

def month_bounds(d: date):
    start = date(d.year, d.month, 1)
    if d.month == 12:
        end = date(d.year + 1, 1, 1) - timedelta(days=1)
    else:
        end = date(d.year, d.month + 1, 1) - timedelta(days=1)
    return start, end

def last_month_bounds(today: date):
    first_this = date(today.year, today.month, 1)
    last_of_prev = first_this - timedelta(days=1)
    return month_bounds(last_of_prev)

def normalize_pipeline(value: str) -> str:
    if not isinstance(value, str):
        return "Other"
    v = value.strip().lower()
    if "math" in v:
        return "Math"
    if "ai" in v or "coding" in v or "ai-coding" in v or "ai coding" in v:
        return "AI Coding"
    return "Other"

def apply_filters(
    df: pd.DataFrame,
    counsellor_col: str | None,
    country_col: str | None,
    source_col: str | None,
    sel_counsellors: list[str],
    sel_countries: list[str],
    sel_sources: list[str],
) -> pd.DataFrame:
    f = df.copy()
    if counsellor_col and len(sel_counsellors) > 0 and "All" not in sel_counsellors:
        f = f[f[counsellor_col].astype(str).isin(sel_counsellors)]
    if country_col and len(sel_countries) > 0 and "All" not in sel_countries:
        f = f[f[country_col].astype(str).isin(sel_countries)]
    if source_col and len(sel_sources) > 0 and "All" not in sel_sources:
        f = f[f[source_col].astype(str).isin(sel_sources)]
    return f

# ---------- COUNT LOGIC ----------
def prepare_counts_for_range(
    df: pd.DataFrame,
    start_d: date,
    end_d: date,
    month_for_mtd: date,
    create_col: str,
    pay_col: str,
    pipeline_col: str | None
):
    """(mtd_counts, cohort_counts)"""
    df = df.copy()
    df["_create_dt"] = coerce_datetime(df[create_col])
    df["_pay_dt"] = coerce_datetime(df[pay_col])

    in_range_pay = df["_pay_dt"].dt.date.between(start_d, end_d)
    cohort_df = df.loc[in_range_pay]

    m_start, m_end = month_bounds(month_for_mtd)
    in_month_create = df["_create_dt"].dt.date.between(m_start, m_end)
    mtd_df = df.loc[in_range_pay & in_month_create]

    if pipeline_col and pipeline_col in df.columns:
        cohort_split = cohort_df[pipeline_col].map(normalize_pipeline).fillna("Other")
        mtd_split = mtd_df[pipeline_col].map(normalize_pipeline).fillna("Other")
    else:
        cohort_split = pd.Series([], dtype=object)
        mtd_split = pd.Series([], dtype=object)

    cohort_counts = {
        "Total": int(len(cohort_df)),
        "AI Coding": int((pd.Series(cohort_split) == "AI Coding").sum()),
        "Math": int((pd.Series(cohort_split) == "Math").sum()),
    }
    mtd_counts = {
        "Total": int(len(mtd_df)),
        "AI Coding": int((pd.Series(mtd_split) == "AI Coding").sum()),
        "Math": int((pd.Series(mtd_split) == "Math").sum()),
    }
    return mtd_counts, cohort_counts

# ---------- CONVERSION% LOGIC ----------
def deals_created_in_anchor_month(df: pd.DataFrame, running_month_any_date: date, create_col: str) -> int:
    df = df.copy()
    df["_create_dt"] = coerce_datetime(df[create_col])
    m_start, m_end = month_bounds(running_month_any_date)
    return int(df["_create_dt"].dt.date.between(m_start, m_end).sum())

def deals_created_in_range(df: pd.DataFrame, denom_start: date, denom_end: date, create_col: str) -> int:
    df = df.copy()
    df["_create_dt"] = coerce_datetime(df[create_col])
    return int(df["_create_dt"].dt.date.between(denom_start, denom_end).sum())

def prepare_conversion_for_range(
    df: pd.DataFrame,
    start_d: date,
    end_d: date,
    create_col: str,
    pay_col: str,
    pipeline_col: str | None,
    *,
    denom_mode: str = "anchor",                  # "anchor" or "range"
    running_month_anchor: date | None = None,    # required if denom_mode="anchor"
    denom_start: date | None = None,             # required if denom_mode="range"
    denom_end: date | None = None
):
    """Returns (mtd_pct, coh_pct, denom, numerators)"""
    df = df.copy()
    df["_create_dt"] = coerce_datetime(df[create_col])
    df["_pay_dt"] = coerce_datetime(df[pay_col])

    # Denominator
    if denom_mode == "range":
        if denom_start is None or denom_end is None:
            return {"Total":0.0,"AI Coding":0.0,"Math":0.0}, {"Total":0.0,"AI Coding":0.0,"Math":0.0}, 0, {"mtd":{}, "cohort":{}}
        denom = deals_created_in_range(df, denom_start, denom_end, create_col)
        in_mtd_create = df["_create_dt"].dt.date.between(denom_start, denom_end)
    else:
        if running_month_anchor is None:
            return {"Total":0.0,"AI Coding":0.0,"Math":0.0}, {"Total":0.0,"AI Coding":0.0,"Math":0.0}, 0, {"mtd":{}, "cohort":{}}
        denom = deals_created_in_anchor_month(df, running_month_anchor, create_col)
        m_start, m_end = month_bounds(running_month_anchor)
        in_mtd_create = df["_create_dt"].dt.date.between(m_start, m_end)

    # Numerators
    in_range_pay = df["_pay_dt"].dt.date.between(start_d, end_d)
    mtd_df = df.loc[in_range_pay & in_mtd_create]
    cohort_df = df.loc[in_range_pay]

    if pipeline_col and pipeline_col in df.columns:
        mtd_ai = int((mtd_df[pipeline_col].map(normalize_pipeline).fillna("Other") == "AI Coding").sum())
        mtd_math = int((mtd_df[pipeline_col].map(normalize_pipeline).fillna("Other") == "Math").sum())
        coh_ai = int((cohort_df[pipeline_col].map(normalize_pipeline).fillna("Other") == "AI Coding").sum())
        coh_math = int((cohort_df[pipeline_col].map(normalize_pipeline).fillna("Other") == "Math").sum())
    else:
        mtd_ai = mtd_math = coh_ai = coh_math = 0

    mtd_total = int(len(mtd_df))
    coh_total = int(len(cohort_df))

    cap = lambda x: max(0.0, min(100.0, x))
    pct1 = lambda v, d: 0.0 if d == 0 else cap(round(100.0 * v / d, 1))

    mtd_pct = {"Total": pct1(mtd_total, denom), "AI Coding": pct1(mtd_ai, denom), "Math": pct1(mtd_math, denom)}
    coh_pct = {"Total": pct1(coh_total, denom), "AI Coding": pct1(coh_ai, denom), "Math": pct1(coh_math, denom)}

    numerators = {
        "mtd": {"Total": mtd_total, "AI Coding": mtd_ai, "Math": mtd_math},
        "cohort": {"Total": coh_total, "AI Coding": coh_ai, "Math": coh_math},
    }
    return mtd_pct, coh_pct, denom, numerators

# ---------- BULLET GAUGE (professional horizontal) ----------
def bullet_gauge(percent: float, title: str, series_color: str, numerator: int, denominator: int,
                 thresholds=(10, 20)):
    """
    Professional horizontal gauge with thresholds:
      - Background bands: [0, low], (low, mid], (mid, 100]
      - Foreground bar: 0 → percent (series color)
      - Thin marker (needle) at percent
    """
    p = float(max(0.0, min(100.0, percent)))
    low, mid = thresholds

    # Background bands
    bg = pd.DataFrame([
        {"band": "low",  "start": 0,   "end": low, "color": PALETTE["ThresholdLow"]},
        {"band": "mid",  "start": low, "end": mid, "color": PALETTE["ThresholdMid"]},
        {"band": "high", "start": mid, "end": 100, "color": PALETTE["ThresholdHigh"]},
    ])

    # Foreground (value)
    fg = pd.DataFrame([{"start": 0, "end": p, "title": title,
                        "percent": f"{p:.1f}%", "num": numerator, "den": denominator}])

    base = alt.Chart(bg).mark_bar(height=18).encode(
        x=alt.X("start:Q", axis=None, scale=alt.Scale(domain=[0,100])),
        x2="end:Q",
        color=alt.Color("band:N", scale=alt.Scale(
            domain=["low","mid","high"],
            range=[PALETTE["ThresholdLow"], PALETTE["ThresholdMid"], PALETTE["ThresholdHigh"]]),
            legend=None),
    ).properties(height=32, width=360)

    value_bar = alt.Chart(fg).mark_bar(height=18, cornerRadius=4).encode(
        x=alt.X("start:Q", axis=None, scale=alt.Scale(domain=[0,100])),
        x2="end:Q",
        color=alt.value(series_color),
        tooltip=[
            alt.Tooltip("title:N", title="Series"),
            alt.Tooltip("percent:N", title="Conversion %"),
            alt.Tooltip("num:Q", title="Numerator"),
            alt.Tooltip("den:Q", title="Denominator"),
        ],
    )

    needle = alt.Chart(pd.DataFrame({"val":[p], "title":[title]})).mark_rule(strokeWidth=2).encode(
        x=alt.X("val:Q", scale=alt.Scale(domain=[0,100])),
        color=alt.value("#111827"),
    )

    label_left = alt.Chart(pd.DataFrame({"t":[title]})).mark_text(
        align="left", baseline="middle", dx=-6, color="#374151", fontSize=12
    ).encode(
        text="t:N"
    ).properties(width=0)  # just used in hconcat

    label_right = alt.Chart(pd.DataFrame({"p":[f"{p:.1f}%"]})).mark_text(
        align="left", baseline="middle", dx=8, color="#111827", fontSize=12, fontWeight="bold"
    ).encode(
        text="p:N"
    ).properties(width=0)

    # Compose: [left label] [gauge] [right % label]
    gauge = alt.hconcat(label_left, (base + value_bar + needle), label_right).resolve_scale(x="shared")
    return gauge

def bullet_group(title: str, pcts: dict, nums: dict, denom: int):
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)

    # KPI chips above gauges
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Total</div>"
                    f"<div class='kpi-value'>{pcts['Total']:.1f}%</div>"
                    f"<div class='kpi-sub'>Den: {denom:,} • Num: {nums.get('Total',0):,}</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>AI-Coding</div>"
                    f"<div class='kpi-value' style='color:{PALETTE['AI Coding']}'>{pcts['AI Coding']:.1f}%</div>"
                    f"<div class='kpi-sub'>Den: {denom:,} • Num: {nums.get('AI Coding',0):,}</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Math</div>"
                    f"<div class='kpi-value' style='color:{PALETTE['Math']}'>{pcts['Math']:.1f}%</div>"
                    f"<div class='kpi-sub'>Den: {denom:,} • Num: {nums.get('Math',0):,}</div></div>", unsafe_allow_html=True)

    g1 = bullet_gauge(pcts["Total"], "Total", PALETTE["Total"], nums.get("Total",0), denom)
    g2 = bullet_gauge(pcts["AI Coding"], "AI-Coding", PALETTE["AI Coding"], nums.get("AI Coding",0), denom)
    g3 = bullet_gauge(pcts["Math"], "Math", PALETTE["Math"], nums.get("Math",0), denom)

    st.altair_chart(g1, use_container_width=True)
    st.altair_chart(g2, use_container_width=True)
    st.altair_chart(g3, use_container_width=True)

# ---------- BUBBLES FOR COUNTS ----------
def bubble_chart_counts(title: str, total: int, ai_cnt: int, math_cnt: int):
    data = pd.DataFrame({
        "Label": ["Total", "AI Coding", "Math"],
        "Value": [total, ai_cnt, math_cnt],
        "Row": [0, 1, 1],
        "Col": [0.5, 0.33, 0.66],
    })
    color_domain = ["Total", "AI Coding", "Math"]
    color_range  = [PALETTE["Total"], PALETTE["AI Coding"], PALETTE["Math"]]
    base = alt.Chart(data).encode(
        x=alt.X("Col:Q", axis=None, scale=alt.Scale(domain=(0, 1))),
        y=alt.Y("Row:Q", axis=None, scale=alt.Scale(domain=(-0.2, 1.2))),
        tooltip=[alt.Tooltip("Label:N"), alt.Tooltip("Value:Q")],
    )
    circles = base.mark_circle(opacity=0.85).encode(
        size=alt.Size("Value:Q", scale=alt.Scale(range=[400, 8000]), legend=None),
        color=alt.Color("Label:N", scale=alt.Scale(domain=color_domain, range=color_range), legend=None),
    )
    text = base.mark_text(fontWeight="bold", dy=0, color="#111827").encode(text=alt.Text("Value:Q"))
    return (circles + text).properties(height=360, title=title)

# ---------- TREND (bar + two lines, dual y) ----------
def trend_timeseries(
    df: pd.DataFrame,
    payments_start: date,
    payments_end: date,
    *,
    denom_mode: str = "anchor",
    running_month_anchor: date | None = None,
    denom_start: date | None = None,
    denom_end: date | None = None,
    create_col: str,
    pay_col: str
):
    df = df.copy()
    df["_create_dt"] = coerce_datetime(df[create_col]).dt.date
    df["_pay_dt"] = coerce_datetime(df[pay_col]).dt.date

    base_start = payments_start
    base_end = payments_end

    if denom_mode == "range" and denom_start and denom_end:
        base_start = min(base_start, denom_start)
        base_end = max(base_end, denom_end)
        denom_mask = df["_create_dt"].between(denom_start, denom_end)
    else:
        if not running_month_anchor:
            running_month_anchor = payments_start
        m_start, m_end = month_bounds(running_month_anchor)
        base_start = min(base_start, m_start)
        base_end = max(base_end, m_end)
        denom_mask = df["_create_dt"].between(m_start, m_end)

    all_days = pd.date_range(base_start, base_end, freq="D").date

    leads = (
        df.loc[denom_mask]
          .groupby("_create_dt")
          .size()
          .reindex(all_days, fill_value=0)
          .rename("Leads")
    )
    pay_mask = df["_pay_dt"].between(payments_start, payments_end)
    cohort = (
        df.loc[pay_mask]
          .groupby("_pay_dt")
          .size()
          .reindex(all_days, fill_value=0)
          .rename("Cohort")
    )
    mtd = (
        df.loc[pay_mask & denom_mask]
          .groupby("_pay_dt")
          .size()
          .reindex(all_days, fill_value=0)
          .rename("MTD")
    )

    ts = pd.concat([leads, mtd, cohort], axis=1).fillna(0).reset_index()
    ts = ts.rename(columns={"index": "Date"})
    return ts

def trend_chart(ts: pd.DataFrame, title: str):
    base = alt.Chart(ts).encode(x=alt.X("Date:T", axis=alt.Axis(title=None)))

    bars = base.mark_bar(opacity=0.75).encode(
        y=alt.Y("Leads:Q", axis=alt.Axis(title="Leads (deals created)")),
        tooltip=[alt.Tooltip("Date:T"), alt.Tooltip("Leads:Q")]
    ).properties(height=260)

    line_mtd = base.mark_line(point=True).encode(
        y=alt.Y("MTD:Q", axis=alt.Axis(title="Enrolments"), scale=alt.Scale(zero=True)),
        color=alt.value(PALETTE["AI Coding"]),
        tooltip=[alt.Tooltip("Date:T"), alt.Tooltip("MTD:Q", title="MTD Enrolments")]
    )

    line_coh = base.mark_line(point=True).encode(
        y=alt.Y("Cohort:Q", axis=alt.Axis(title="Enrolments"), scale=alt.Scale(zero=True)),
        color=alt.value(PALETTE["Math"]),
        tooltip=[alt.Tooltip("Date:T"), alt.Tooltip("Cohort:Q", title="Cohort Enrolments")]
    )

    return alt.layer(bars, line_mtd, line_coh).resolve_scale(y='independent').properties(title=title)

# ----------------------------
# UI
# ----------------------------
with st.sidebar:
    st.header("JetLearn • Navigation")
    view = st.radio("Go to", ["MIS"], index=0)
    st.caption("Use the quick periods, filters, or the Custom tab.")

st.title("📊 JetLearn MIS")
st.markdown(
    """
    <div>
      <span class="legend-pill pill-ai">AI-Coding</span>
      <span class="legend-pill pill-math">Math</span>
      <span class="legend-pill pill-total">Total (Both)</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write(
    "Visualizes **Enrolments (Payments)**, **Conversion%** (bullet gauges), and **Trend** (Leads vs Enrolments). "
    "Conversion% uses a **single shared denominator** for Total/AI/Math."
)

# --- Load data
default_path = "Master_sheet_DB.csv"
data_src = st.text_input("Data file path", value=default_path, help="CSV path (pre-uploaded in the repo).")
df = load_data(data_src)

# --- Resolve columns
create_col = find_col(df, ["Create Date", "Create date", "Create_Date", "Created At"])
pay_col = find_col(df, ["Payment Received Date", "Payment Received date", "Payment_Received_Date", "Payment Date", "Paid At"])
pipeline_col = find_col(df, ["Pipeline"])

# Filters: Academic Counsellor, Country, JetLearn Deal Source
counsellor_col = find_col(df, ["Student/Academic Counsellor", "Academic Counsellor", "Student/Academic Counselor", "Counsellor", "Counselor"])
country_col = find_col(df, ["Country"])
source_col = find_col(df, ["JetLearn Deal Source", "Deal Source", "Source"])

if not create_col or not pay_col:
    st.error("Could not find required date columns. Ensure the CSV has 'Create Date' and 'Payment Received Date' (or close variants).")
    st.stop()

# --- Period presets
today = date.today()
yday = today - timedelta(days=1)
last_m_start, last_m_end = last_month_bounds(today)
this_m_start, this_m_end = month_bounds(today)

# --- Filters UI
with st.expander("Filters", expanded=True):
    def prep_options(series: pd.Series):
        vals = sorted([str(v) for v in series.dropna().unique()])
        return ["All"] + vals

    if counsellor_col:
        counsellor_opts = prep_options(df[counsellor_col])
        sel_counsellors = st.multiselect("Academic Counsellor", options=counsellor_opts, default=["All"])
    else:
        sel_counsellors = []
        st.info("Academic Counsellor column not found. Skipping this filter.")

    if country_col:
        country_opts = prep_options(df[country_col])
        sel_countries = st.multiselect("Country", options=country_opts, default=["All"])
    else:
        sel_countries = []
        st.info("Country column not found. Skipping this filter.")

    if source_col:
        source_opts = prep_options(df[source_col])
        sel_sources = st.multiselect("JetLearn Deal Source", options=source_opts, default=["All"])
    else:
        sel_sources = []
        st.info("JetLearn Deal Source column not found. Skipping this filter.")

# Apply filters
df_f = apply_filters(df, counsellor_col, country_col, source_col, sel_counsellors, sel_countries, sel_sources)
st.caption(f"Rows in scope after filters: **{len(df_f):,}**")

# Show-all toggle
show_all = st.checkbox("Show all preset periods (Yesterday • Today • Last Month • This Month)", value=False)

# ----------------------------
# Period sections (Counts + Conversion% + Trend)
# ----------------------------
def render_period_block(title: str, range_start: date, range_end: date, running_month_anchor: date):
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)

    # Counts
    mtd_counts, coh_counts = prepare_counts_for_range(
        df_f, range_start, range_end, running_month_anchor,
        create_col, pay_col, pipeline_col
    )
    c1, c2 = st.columns(2)
    with c1:
        st.altair_chart(
            bubble_chart_counts("MTD Enrolments (counts)", mtd_counts["Total"], mtd_counts["AI Coding"], mtd_counts["Math"]),
            use_container_width=True
        )
    with c2:
        st.altair_chart(
            bubble_chart_counts("Cohort Enrolments (counts)", coh_counts["Total"], coh_counts["AI Coding"], coh_counts["Math"]),
            use_container_width=True
        )

    # Conversion% (shared denominator) → Bullet gauges
    mtd_pct, coh_pct, denom, nums = prepare_conversion_for_range(
        df_f, range_start, range_end, create_col, pay_col, pipeline_col,
        denom_mode="anchor", running_month_anchor=running_month_anchor
    )
    st.caption(f"Conversion% denominator (deals created in running month): **{denom:,}**")
    bullet_group("MTD Conversion %", mtd_pct, nums["mtd"], denom)
    bullet_group("Cohort Conversion %", coh_pct, nums["cohort"], denom)

    # Trend (combined)
    ts = trend_timeseries(
        df_f, range_start, range_end,
        denom_mode="anchor", running_month_anchor=running_month_anchor,
        create_col=create_col, pay_col=pay_col
    )
    st.altair_chart(trend_chart(ts, "Trend: Leads (bars) vs Enrolments (lines)"), use_container_width=True)

if view == "MIS":
    if show_all:
        st.subheader("Preset Periods")
        colA, colB = st.columns(2)
        with colA:
            render_period_block("Yesterday", yday, yday, yday)
            st.divider()
            render_period_block("Last Month", last_m_start, last_m_end, last_m_start)
        with colB:
            render_period_block("Today", today, today, today)
            st.divider()
            render_period_block("This Month", this_m_start, this_m_end, this_m_start)
    else:
        tabs = st.tabs(["Yesterday", "Today", "Last Month", "This Month", "Custom"])

        with tabs[0]:
            render_period_block("Yesterday", yday, yday, yday)

        with tabs[1]:
            render_period_block("Today", today, today, today)

        with tabs[2]:
            render_period_block("Last Month", last_m_start, last_m_end, last_m_start)

        with tabs[3]:
            render_period_block("This Month", this_m_start, this_m_end, this_m_start)

        # ------------- Custom tab -------------
        with tabs[4]:
            st.markdown("Select a **payments period** and choose the **Conversion% denominator** mode.")
            colc1, colc2 = st.columns(2)
            with colc1:
                custom_start = st.date_input("Payments period start", value=this_m_start)
            with colc2:
                custom_end = st.date_input("Payments period end (inclusive)", value=this_m_end)
            if custom_end < custom_start:
                st.error("Payments period end cannot be before start.")
            else:
                denom_mode = st.radio("Denominator for Conversion%", ["Anchor month", "Custom range"], index=0, horizontal=True)

                if denom_mode == "Anchor month":
                    anchor = st.date_input("Running-month anchor (denominator month)", value=custom_start)

                    # Counts
                    mtd_counts, coh_counts = prepare_counts_for_range(df_f, custom_start, custom_end, anchor, create_col, pay_col, pipeline_col)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.altair_chart(bubble_chart_counts("MTD Enrolments (counts)", mtd_counts["Total"], mtd_counts["AI Coding"], mtd_counts["Math"]), use_container_width=True)
                    with c2:
                        st.altair_chart(bubble_chart_counts("Cohort Enrolments (counts)", coh_counts["Total"], coh_counts["AI Coding"], coh_counts["Math"]), use_container_width=True)

                    # Conversion → bullet gauges (anchor)
                    mtd_pct, coh_pct, denom, nums = prepare_conversion_for_range(
                        df_f, custom_start, custom_end, create_col, pay_col, pipeline_col,
                        denom_mode="anchor", running_month_anchor=anchor
                    )
                    st.caption(f"Conversion% denominator (deals created in anchor month): **{denom:,}**")
                    bullet_group("MTD Conversion %", mtd_pct, nums["mtd"], denom)
                    bullet_group("Cohort Conversion %", coh_pct, nums["cohort"], denom)

                    # Trend using anchor denom
                    ts = trend_timeseries(
                        df_f, custom_start, custom_end,
                        denom_mode="anchor", running_month_anchor=anchor,
                        create_col=create_col, pay_col=pay_col
                    )
                    st.altair_chart(trend_chart(ts, "Trend: Leads (bars) vs Enrolments (lines)"), use_container_width=True)

                else:
                    cold1, cold2 = st.columns(2)
                    with cold1:
                        denom_start = st.date_input("Denominator start (deals created from)", value=custom_start, key="denom_start")
                    with cold2:
                        denom_end = st.date_input("Denominator end (deals created to)", value=custom_end, key="denom_end")

                    if denom_end < denom_start:
                        st.error("Denominator end cannot be before start.")
                    else:
                        # Counts anchored to custom_start month for MTD count view
                        anchor_for_counts = custom_start
                        mtd_counts, coh_counts = prepare_counts_for_range(df_f, custom_start, custom_end, anchor_for_counts, create_col, pay_col, pipeline_col)
                        c1, c2 = st.columns(2)
                        with c1:
                            st.altair_chart(bubble_chart_counts("MTD Enrolments (counts)", mtd_counts["Total"], mtd_counts["AI Coding"], mtd_counts["Math"]), use_container_width=True)
                        with c2:
                            st.altair_chart(bubble_chart_counts("Cohort Enrolments (counts)", coh_counts["Total"], coh_counts["AI Coding"], coh_counts["Math"]), use_container_width=True)

                        # Conversion → bullet gauges (custom range denom)
                        mtd_pct, coh_pct, denom, nums = prepare_conversion_for_range(
                            df_f, custom_start, custom_end, create_col, pay_col, pipeline_col,
                            denom_mode="range", denom_start=denom_start, denom_end=denom_end
                        )
                        st.caption(f"Conversion% denominator (deals created in custom range): **{denom:,}**")
                        bullet_group("MTD Conversion %", mtd_pct, nums["mtd"], denom)
                        bullet_group("Cohort Conversion %", coh_pct, nums["cohort"], denom)

                        # Trend using custom denom range
                        ts = trend_timeseries(
                            df_f, custom_start, custom_end,
                            denom_mode="range", denom_start=denom_start, denom_end=denom_end,
                            create_col=create_col, pay_col=pay_col
                        )
                        st.altair_chart(trend_chart(ts, "Trend: Leads (bars) vs Enrolments (lines)"), use_container_width=True)

# Optional: data preview
with st.expander("Data preview & column mapping", expanded=False):
    st.write({
        "Create Date": create_col,
        "Payment Received Date": pay_col,
        "Pipeline (split)": pipeline_col or "Not found → using heuristic",
        "Academic Counsellor": counsellor_col or "Not found",
        "Country": country_col or "Not found",
        "JetLearn Deal Source": source_col or "Not found",
    })
    st.dataframe(df.head(20))
