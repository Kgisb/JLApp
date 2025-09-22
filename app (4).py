import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, date, timedelta
import re
from calendar import monthrange

st.set_page_config(
    page_title="JetLearn MIS – Enrolments (MTD & Cohort) + Conversion% + Predictibility",
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
    "Total": "#6b7280",
    "AI Coding": "#2563eb",
    "Math": "#16a34a",
    "ThresholdLow": "#f3f4f6",
    "ThresholdMid": "#e5e7eb",
    "ThresholdHigh": "#d1d5db",

    # Predictibility
    "A_actual": "#2563eb",          # blue
    "Rem_prev": "#6b7280",          # gray
    "Rem_same": "#16a34a",          # green
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
        for unit in ["s", "ms"]:
            try:
                s = pd.to_datetime(series, errors="coerce", unit=unit)
                break
            except Exception:
                pass
    return s

def month_bounds(d: date):
    start = date(d.year, d.month, 1)
    end = date(d.year, d.month, monthrange(d.year, d.month)[1])
    return start, end

def last_month_bounds(today: date):
    first_this = date(today.year, today.month, 1)
    last_of_prev = first_this - timedelta(days=1)
    return month_bounds(last_of_prev)

def month_days(period: pd.Period) -> int:
    y, m = period.year, period.month
    return monthrange(y, m)[1]

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

# ---------- Always-on exclusion for Invalid Deals ----------
INVALID_RE = re.compile(r"^\s*1\.2\s*invalid\s*deal[s]?\s*$", flags=re.IGNORECASE)

def exclude_invalid_deals(df: pd.DataFrame, dealstage_col: str | None) -> tuple[pd.DataFrame, int]:
    if not dealstage_col:
        return df, 0
    col = df[dealstage_col].astype(str)
    mask_keep = ~col.apply(lambda x: bool(INVALID_RE.match(x)))
    removed = int((~mask_keep).sum())
    return df.loc[mask_keep].copy(), removed

# ----------------------------
# Session state for data path (default)
# ----------------------------
DEFAULT_DATA_PATH = "Master_sheet_DB.csv"
if "data_src" not in st.session_state:
    st.session_state["data_src"] = DEFAULT_DATA_PATH

# ----------------------------
# UI – Sidebar and Header
# ----------------------------
with st.sidebar:
    st.header("JetLearn • Navigation")
    view = st.radio("Go to", ["MIS", "Predictibility", "Trend & Analysis"], index=0)
    track = st.radio("Track", ["Both", "AI Coding", "Math"], index=0)  # Track toggle
    st.caption("Use MIS for status; Predictibility for month-end forecast (A/B/C) & accuracy; Trend & Analysis for grouped drilldowns.")

st.title("📊 JetLearn MIS")

# Legend pills show only active series
def active_labels(track: str) -> list[str]:
    if track == "AI Coding":
        return ["Total", "AI Coding"]
    if track == "Math":
        return ["Total", "Math"]
    return ["Total", "AI Coding", "Math"]

legend_labels = active_labels(track)
pill_map = {
    "Total": "<span class='legend-pill pill-total'>Total (Both)</span>",
    "AI Coding": "<span class='legend-pill pill-ai'>AI-Coding</span>",
    "Math": "<span class='legend-pill pill-math'>Math</span>",
}
st.markdown(
    "<div>" + "".join(pill_map[l] for l in legend_labels) + "</div>",
    unsafe_allow_html=True,
)
st.write("Visualizes **Enrolments (Payments)**, **Conversion%**, **Trend**, **Predictibility**, and **Drilldowns** with **Model Accuracy**.")

# ----------------------------
# Load & clean data
# ----------------------------
data_src = st.session_state["data_src"]
df = load_data(data_src)

dealstage_col = find_col(df, ["Deal Stage", "Deal stage", "Stage", "Deal Status", "Stage Name", "Deal Stage Name"])
df, _removed = exclude_invalid_deals(df, dealstage_col)
if dealstage_col:
    st.caption(f"Excluded “1.2 Invalid deal(s)”: **{_removed:,}** rows (column: **{dealstage_col}**).")
else:
    st.info("Deal Stage column not found — cannot exclude “1.2 Invalid deal(s)”. Check your file.")

create_col = find_col(df, ["Create Date", "Create date", "Create_Date", "Created At"])
pay_col = find_col(df, ["Payment Received Date", "Payment Received date", "Payment_Received_Date", "Payment Date", "Paid At"])
pipeline_col = find_col(df, ["Pipeline"])

counsellor_col = find_col(df, ["Student/Academic Counsellor", "Academic Counsellor", "Student/Academic Counselor", "Counsellor", "Counselor"])
country_col = find_col(df, ["Country"])
source_col = find_col(df, ["JetLearn Deal Source", "Deal Source", "Source"])

if not create_col or not pay_col:
    st.error("Could not find required date columns. Need 'Create Date' and 'Payment Received Date' (or close variants).")
    st.stop()

tmp_create = coerce_datetime(df[create_col])
missing_create = int(tmp_create.isna().sum())
if missing_create > 0:
    df = df.loc[tmp_create.notna()].copy()
    st.caption(f"Removed rows with missing/invalid **Create Date**: **{missing_create:,}**")

# --- Period presets
today = date.today()
yday = today - timedelta(days=1)
last_m_start, last_m_end = last_month_bounds(today)
this_m_start, this_m_end = month_bounds(today)

# ----------------------------
# Filters expander (collapsed) with data path bottom-left
# ----------------------------
def _update_data_src():
    st.session_state["data_src"] = st.session_state.get("data_src_input", DEFAULT_DATA_PATH)
    st.rerun()

with st.expander("Filters", expanded=False):
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

    st.caption("Auto-excludes ‘1.2 Invalid deal(s)’ • Drops rows with missing **Create Date**")

    col_bottom_left, col_bottom_right = st.columns([3, 2])
    with col_bottom_left:
        st.text_input(
            "Data file path",
            key="data_src_input",
            value=st.session_state.get("data_src", DEFAULT_DATA_PATH),
            help="CSV path (pre-uploaded in the repo).",
            on_change=_update_data_src,
        )
    with col_bottom_right:
        st.empty()

# Apply filters and Track filter
df_f = apply_filters(df, counsellor_col, country_col, source_col, sel_counsellors, sel_countries, sel_sources)

if track != "Both":
    if pipeline_col and pipeline_col in df_f.columns:
        _norm = df_f[pipeline_col].map(normalize_pipeline).fillna("Other")
        df_f = df_f.loc[_norm == track].copy()
    else:
        st.warning("Pipeline column not found — the Track filter can’t be applied.", icon="⚠️")

st.caption(f"Rows in scope after filters: **{len(df_f):,}**")
st.caption(f"Track filter: **{track}**")

# ----------------------------
# COUNT & CONVERSION LOGIC
# ----------------------------
def prepare_counts_for_range(
    df: pd.DataFrame,
    start_d: date,
    end_d: date,
    month_for_mtd: date,
    create_col: str,
    pay_col: str,
    pipeline_col: str | None
):
    d = df.copy()
    d["_create_dt"] = coerce_datetime(d[create_col])
    d["_pay_dt"] = coerce_datetime(d[pay_col])

    in_range_pay = d["_pay_dt"].dt.date.between(start_d, end_d)
    m_start, m_end = month_bounds(month_for_mtd)
    in_month_create = d["_create_dt"].dt.date.between(m_start, m_end)

    cohort_df = d.loc[in_range_pay]
    mtd_df = d.loc[in_range_pay & in_month_create]

    if pipeline_col and pipeline_col in d.columns:
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

def deals_created_mask_anchor(df: pd.DataFrame, running_month_any_date: date, create_col: str) -> pd.Series:
    d = df.copy()
    d["_create_dt"] = coerce_datetime(d[create_col]).dt.date
    m_start, m_end = month_bounds(running_month_any_date)
    return d["_create_dt"].between(m_start, m_end)

def deals_created_mask_range(df: pd.DataFrame, denom_start: date, denom_end: date, create_col: str) -> pd.Series:
    d = df.copy()
    d["_create_dt"] = coerce_datetime(d[create_col]).dt.date
    return d["_create_dt"].between(denom_start, denom_end)

def prepare_conversion_for_range(
    df: pd.DataFrame,
    start_d: date,
    end_d: date,
    create_col: str,
    pay_col: str,
    pipeline_col: str | None,
    *,
    denom_mode: str = "anchor",
    running_month_anchor: date | None = None,
    denom_start: date | None = None,
    denom_end: date | None = None
):
    d = df.copy()
    d["_create_dt"] = coerce_datetime(d[create_col]).dt.date
    d["_pay_dt"] = coerce_datetime(d[pay_col]).dt.date

    if denom_mode == "range":
        if denom_start is None or denom_end is None:
            zero = {"Total":0.0,"AI Coding":0.0,"Math":0.0}
            return zero, zero, {"Total":0,"AI Coding":0,"Math":0}, {"mtd": {}, "cohort": {}}
        denom_mask = deals_created_mask_range(d, denom_start, denom_end, create_col)
    else:
        if running_month_anchor is None:
            zero = {"Total":0.0,"AI Coding":0.0,"Math":0.0}
            return zero, zero, {"Total":0,"AI Coding":0,"Math":0}, {"mtd": {}, "cohort": {}}
        denom_mask = deals_created_mask_anchor(d, running_month_anchor, create_col)

    if pipeline_col and pipeline_col in d.columns:
        pl = d[pipeline_col].map(normalize_pipeline).fillna("Other")
    else:
        pl = pd.Series(["Other"] * len(d), index=d.index)

    den_total = int(denom_mask.sum())
    den_ai    = int((denom_mask & (pl == "AI Coding")).sum())
    den_math  = int((denom_mask & (pl == "Math")).sum())
    denoms = {"Total": den_total, "AI Coding": den_ai, "Math": den_math}

    pay_mask = d["_pay_dt"].between(start_d, end_d)

    mtd_mask = pay_mask & denom_mask
    mtd_total = int(mtd_mask.sum())
    mtd_ai    = int((mtd_mask & (pl == "AI Coding")).sum())
    mtd_math  = int((mtd_mask & (pl == "Math")).sum())

    coh_mask = pay_mask
    coh_total = int(coh_mask.sum())
    coh_ai    = int((coh_mask & (pl == "AI Coding")).sum())
    coh_math  = int((coh_mask & (pl == "Math")).sum())

    def pct(n, d):
        if d == 0:
            return 0.0
        return max(0.0, min(100.0, round(100.0 * n / d, 1)))

    mtd_pct = {"Total": pct(mtd_total, den_total), "AI Coding": pct(mtd_ai, den_ai), "Math": pct(mtd_math, den_math)}
    coh_pct = {"Total": pct(coh_total, den_total), "AI Coding": pct(coh_ai, den_ai), "Math": pct(coh_math, den_math)}
    numerators = {"mtd": {"Total": mtd_total, "AI Coding": mtd_ai, "Math": mtd_math},
                  "cohort": {"Total": coh_total, "AI Coding": coh_ai, "Math": coh_math}}
    return mtd_pct, coh_pct, denoms, numerators

# ----------------------------
# MIS visuals (track-aware)
# ----------------------------
def bubble_chart_counts(title: str, total: int, ai_cnt: int, math_cnt: int, labels: list[str] = None):
    all_rows = [
        {"Label": "Total",     "Value": total,   "Row": 0, "Col": 0.5},
        {"Label": "AI Coding", "Value": ai_cnt,  "Row": 1, "Col": 0.33},
        {"Label": "Math",      "Value": math_cnt,"Row": 1, "Col": 0.66},
    ]
    if labels is None:
        labels = ["Total", "AI Coding", "Math"]
    data = pd.DataFrame([r for r in all_rows if r["Label"] in labels])

    color_domain = labels
    color_range_map = {
        "Total": PALETTE["Total"],
        "AI Coding": PALETTE["AI Coding"],
        "Math": PALETTE["Math"],
    }
    color_range = [color_range_map[l] for l in labels]

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

def bullet_gauge(percent: float, title: str, series_color: str, numerator: int, denominator: int,
                 thresholds=(10, 20)):
    p = float(max(0.0, min(100.0, percent)))
    low, mid = thresholds
    bg = pd.DataFrame([
        {"band": "low",  "start": 0,   "end": low},
        {"band": "mid",  "start": low, "end": mid},
        {"band": "high", "start": mid, "end": 100},
    ])
    fg = pd.DataFrame([{"start": 0, "end": p, "title": title, "percent": f"{p:.1f}%", "num": numerator, "den": denominator}])

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

    needle = alt.Chart(pd.DataFrame({"val":[p]})).mark_rule(strokeWidth=2).encode(
        x=alt.X("val:Q", scale=alt.Scale(domain=[0,100])),
        color=alt.value("#111827"),
    )

    label_left = alt.Chart(pd.DataFrame({"t":[title]})).mark_text(
        align="left", baseline="middle", dx=-6, color="#374151", fontSize=12
    ).encode(text="t:N").properties(width=0)

    label_right = alt.Chart(pd.DataFrame({"p":[f"{p:.1f}%"]})).mark_text(
        align="left", baseline="middle", dx=8, color="#111827", fontSize=12, fontWeight="bold"
    ).encode(text="p:N").properties(width=0)

    return alt.hconcat(label_left, (base + value_bar + needle), label_right).resolve_scale(x="shared")

def bullet_group(title: str, pcts: dict, nums: dict, denoms: dict, labels: list[str]):
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
    order = [l for l in ["Total", "AI Coding", "Math"] if l in labels]

    cols = st.columns(len(order))
    for i, label in enumerate(order):
        color = {"Total":"#111827","AI Coding":PALETTE["AI Coding"],"Math":PALETTE["Math"]}[label]
        with cols[i]:
            st.markdown(
                f"<div class='kpi-card'><div class='kpi-title'>{label}</div>"
                f"<div class='kpi-value' style='color:{color}'>{pcts[label]:.1f}%</div>"
                f"<div class='kpi-sub'>Den: {denoms.get(label,0):,} • Num: {nums.get(label,0):,}</div></div>",
                unsafe_allow_html=True,
            )

    for label in order:
        series_color = {"Total": PALETTE["Total"], "AI Coding": PALETTE["AI Coding"], "Math": PALETTE["Math"]}[label]
        st.altair_chart(
            bullet_gauge(pcts[label], label, series_color, nums.get(label,0), denoms.get(label,0)),
            use_container_width=True
        )

def trend_timeseries(
    df: pd.DataFrame,
    payments_start: date,
    payments_end: date,
    *,
    denom_mode: str = "anchor",
    running_month_anchor: date | None = None,
    denom_start: date | None = None,
    denom_end: date | None = None,
    create_col: str = "",
    pay_col: str = ""
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
# Predictibility core (A/B/C via lookback daily rates) + Accuracy
# ----------------------------
def add_month_cols(df: pd.DataFrame, create_col: str, pay_col: str) -> pd.DataFrame:
    d = df.copy()
    d["_create_dt"] = coerce_datetime(df[create_col])
    d["_pay_dt"]    = coerce_datetime(df[pay_col])
    d["_create_m"]  = d["_create_dt"].dt.to_period("M")
    d["_pay_m"]     = d["_pay_dt"].dt.to_period("M")
    d["_same_month"] = (d["_create_m"] == d["_pay_m"])
    return d

def per_source_monthly_counts(d_hist: pd.DataFrame, source_col: str):
    if d_hist.empty:
        return pd.DataFrame(columns=["_pay_m", source_col, "cnt_same", "cnt_prev", "days_in_month"])
    g = d_hist.groupby(["_pay_m", source_col])
    by = g["_same_month"].agg(
        cnt_same=lambda s: int(s.sum()),
        cnt_prev=lambda s: int((~s).sum())
    ).reset_index()
    by["days_in_month"] = by["_pay_m"].apply(month_days)
    return by

def daily_rates_from_lookback(d_hist: pd.DataFrame, source_col: str, lookback: int, weighted: bool):
    if d_hist.empty:
        return {}, {}, 0.0, 0.0

    months = sorted(d_hist["_pay_m"].unique())
    months = months[-lookback:] if len(months) > lookback else months
    d_hist = d_hist[d_hist["_pay_m"].isin(months)].copy()

    by = per_source_monthly_counts(d_hist, source_col)
    month_to_w = {m: (i+1 if weighted else 1.0) for i, m in enumerate(sorted(months))}

    rates_same, rates_prev = {}, {}
    for src, sub in by.groupby(source_col):
        w = sub["_pay_m"].map(month_to_w)
        num_same = (sub["cnt_same"] / sub["days_in_month"] * w).sum()
        num_prev = (sub["cnt_prev"] / sub["days_in_month"] * w).sum()
        den = w.sum()
        rates_same[str(src)] = float(num_same/den) if den > 0 else 0.0
        rates_prev[str(src)] = float(num_prev/den) if den > 0 else 0.0

    by_overall = d_hist.groupby("_pay_m")["_same_month"].agg(
        cnt_same=lambda s: int(s.sum()),
        cnt_prev=lambda s: int((~s).sum())
    ).reset_index()
    by_overall["days_in_month"] = by_overall["_pay_m"].apply(month_days)
    w_all = by_overall["_pay_m"].map(month_to_w)
    num_same_o = (by_overall["cnt_same"] / by_overall["days_in_month"] * w_all).sum()
    num_prev_o = (by_overall["cnt_prev"] / by_overall["days_in_month"] * w_all).sum()
    den_o = w_all.sum()
    overall_same_rate = float(num_same_o/den_o) if den_o > 0 else 0.0
    overall_prev_rate = float(num_prev_o/den_o) if den_o > 0 else 0.0

    return rates_same, rates_prev, overall_same_rate, overall_prev_rate

def predict_running_month(df_f: pd.DataFrame, create_col: str, pay_col: str, source_col: str,
                          lookback: int, weighted: bool, today: date):
    if source_col is None or source_col not in df_f.columns:
        df_work = df_f.copy()
        source_col = "_Source"
        df_work[source_col] = "All"
    else:
        df_work = df_f.copy()

    d = add_month_cols(df_work, create_col, pay_col)

    cur_start, cur_end = month_bounds(today)
    cur_period = pd.Period(today, freq="M")

    d_cur = d[d["_pay_m"] == cur_period].copy()
    if d_cur.empty:
        realized_by_src = pd.DataFrame(columns=[source_col, "A"])
    else:
        realized_by_src = d_cur.groupby(source_col).size().rename("A").reset_index()

    d_hist = d[d["_pay_m"] < cur_period].copy()
    rates_same, rates_prev, overall_same_rate, overall_prev_rate = daily_rates_from_lookback(
        d_hist, source_col, lookback, weighted
    )

    elapsed_days = (today - cur_start).days + 1
    total_days   = (cur_end - cur_start).days + 1
    remaining_days = max(0, total_days - elapsed_days)

    src_realized = set(d_cur[source_col].dropna().astype(str)) if not d_cur.empty else set()
    src_hist = set(list(rates_same.keys()) + list(rates_prev.keys()))
    all_sources = sorted(src_realized | src_hist | ({"All"} if source_col == "_Source" else set()))

    A_tot = B_tot = C_tot = 0.0
    rows = []
    a_map = dict(zip(realized_by_src[source_col], realized_by_src["A"])) if not realized_by_src.empty else {}

    for src in all_sources:
        a_val = float(a_map.get(src, 0.0))
        rate_same = rates_same.get(src, overall_same_rate)
        rate_prev = rates_prev.get(src, overall_prev_rate)

        b_val = float(rate_same * remaining_days)
        c_val = float(rate_prev * remaining_days)

        rows.append({
            "Source": src,
            "A_Actual_ToDate": a_val,
            "B_Remaining_SameMonth": b_val,
            "C_Remaining_PrevMonths": c_val,
            "Projected_MonthEnd_Total": a_val + b_val + c_val,
            "Rate_Same_Daily": rate_same,
            "Rate_Prev_Daily": rate_prev,
            "Remaining_Days": remaining_days
        })
        A_tot += a_val
        B_tot += b_val
        C_tot += c_val

    tbl = pd.DataFrame(rows).sort_values("Source").reset_index(drop=True)
    totals = {
        "A_Actual_ToDate": A_tot,
        "B_Remaining_SameMonth": B_tot,
        "C_Remaining_PrevMonths": C_tot,
        "Projected_MonthEnd_Total": A_tot + B_tot + C_tot,
        "Remaining_Days": remaining_days
    }
    return tbl, totals

def predict_chart_stacked(tbl: pd.DataFrame):
    if tbl.empty:
        return alt.Chart(pd.DataFrame({"x":[],"y":[]}))
    melt = tbl.melt(
        id_vars=["Source"],
        value_vars=["A_Actual_ToDate","B_Remaining_SameMonth","C_Remaining_PrevMonths"],
        var_name="Component",
        value_name="Value"
    )
    color_map = {
        "A_Actual_ToDate": PALETTE["A_actual"],
        "B_Remaining_SameMonth": PALETTE["Rem_same"],
        "C_Remaining_PrevMonths": PALETTE["Rem_prev"],
    }
    chart = alt.Chart(melt).mark_bar().encode(
        x=alt.X("Source:N", sort=alt.SortField("Source")),
        y=alt.Y("Value:Q", stack=True),
        color=alt.Color("Component:N",
                        scale=alt.Scale(domain=list(color_map.keys()), range=list(color_map.values())),
                        legend=alt.Legend(title="Component", orient="top", labelLimit=240)),
        tooltip=[alt.Tooltip("Source:N"),
                 alt.Tooltip("Component:N"),
                 alt.Tooltip("Value:Q", format=",.1f")]
    ).properties(height=360, title="Predictibility (A + B + C = Projected Month-End)")
    return chart

def month_list_before(period_end: pd.Period, k: int):
    months = []
    p = period_end
    for _ in range(k):
        p = (p - 1)
        months.append(p)
    months.reverse()
    return months

def backtest_accuracy(df_f: pd.DataFrame, create_col: str, pay_col: str, source_col: str,
                      lookback: int, weighted: bool, backtest_months: int, today: date):
    if source_col is None or source_col not in df_f.columns:
        df_work = df_f.copy()
        source_col = "_Source"
        df_work[source_col] = "All"
    else:
        df_work = df_f.copy()

    d = add_month_cols(df_work, create_col, pay_col)
    current_period = pd.Period(today, freq="M")

    months_to_eval = month_list_before(current_period, backtest_months)
    rows = []
    for m in months_to_eval:
        train_months = month_list_before(m, lookback)
        d_train = d[d["_pay_m"].isin(train_months)]
        if d_train.empty:
            same_rates, prev_rates, same_rate_o, prev_rate_o = {}, {}, 0.0, 0.0
        else:
            same_rates, prev_rates, same_rate_o, prev_rate_o = daily_rates_from_lookback(
                d_train, source_col, lookback=len(train_months), weighted=weighted
            )

        d_m = d[d["_pay_m"] == m]
        actual_total = int(len(d_m))
        days_in_m = month_days(m)

        sources = set(list(same_rates.keys()) + list(prev_rates.keys()))
        if not sources and source_col != "_Source":
            sources = set(d_m[source_col].dropna().astype(str).unique().tolist())
        if not sources:
            sources = {"All"}

        forecast = 0.0
        for src in sources:
            r_same = same_rates.get(src, same_rate_o)
            r_prev = prev_rates.get(src, prev_rate_o)
            forecast += (r_same + r_prev) * days_in_m

        err = forecast - actual_total
        rows.append({
            "Month": str(m),
            "Days": days_in_m,
            "Forecast": float(forecast),
            "Actual": float(actual_total),
            "Error": float(err),
            "AbsError": float(abs(err)),
            "SqError": float(err**2),
            "APE": float(abs(err) / actual_total) if actual_total > 0 else np.nan
        })

    bt = pd.DataFrame(rows)
    if bt.empty:
        return bt, {"MAPE": np.nan, "WAPE": np.nan, "MAE": np.nan, "RMSE": np.nan, "R2": np.nan}

    mae = bt["AbsError"].mean()
    rmse = (bt["SqError"].mean())**0.5
    wape = (bt["AbsError"].sum() / bt["Actual"].sum()) if bt["Actual"].sum() > 0 else np.nan
    mape = bt["APE"].dropna().mean() if bt["APE"].notna().any() else np.nan
    ss_res = ((bt["Actual"] - bt["Forecast"])**2).sum()
    ss_tot = ((bt["Actual"] - bt["Actual"].mean())**2).sum()
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan

    metrics = {"MAPE": mape, "WAPE": wape, "MAE": mae, "RMSE": rmse, "R2": r2}
    return bt, metrics

def accuracy_scatter(bt: pd.DataFrame):
    if bt.empty:
        return alt.Chart(pd.DataFrame({"x":[],"y":[]}))
    chart = alt.Chart(bt).mark_circle(size=120, opacity=0.8).encode(
        x=alt.X("Actual:Q", title="Actual (month total)"),
        y=alt.Y("Forecast:Q", title="Forecast (start-of-month)"),
        tooltip=[alt.Tooltip("Month:N"), alt.Tooltip("Actual:Q"), alt.Tooltip("Forecast:Q"), alt.Tooltip("Error:Q")],
    ).properties(height=360, title="Forecast vs Actual (by month)")
    line = alt.Chart(pd.DataFrame({"x":[bt["Actual"].min(), bt["Actual"].max()],
                                   "y":[bt["Actual"].min(), bt["Actual"].max()]})).mark_line()
    return chart + line

# ----------------------------
# MIS section helpers (track-aware rendering)
# ----------------------------
def render_period_block(
    df_scope: pd.DataFrame,
    title: str,
    range_start: date,
    range_end: date,
    running_month_anchor: date,
    create_col: str,
    pay_col: str,
    pipeline_col: str | None,
    track: str
):
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
    labels = active_labels(track)

    mtd_counts, coh_counts = prepare_counts_for_range(
        df_scope, range_start, range_end, running_month_anchor, create_col, pay_col, pipeline_col
    )

    c1, c2 = st.columns(2)
    with c1:
        st.altair_chart(
            bubble_chart_counts(
                "MTD Enrolments (counts)",
                mtd_counts["Total"], mtd_counts["AI Coding"], mtd_counts["Math"],
                labels=labels,
            ),
            use_container_width=True
        )
    with c2:
        st.altair_chart(
            bubble_chart_counts(
                "Cohort Enrolments (counts)",
                coh_counts["Total"], coh_counts["AI Coding"], coh_counts["Math"],
                labels=labels,
            ),
            use_container_width=True
        )

    mtd_pct, coh_pct, denoms, nums = prepare_conversion_for_range(
        df_scope, range_start, range_end, create_col, pay_col, pipeline_col,
        denom_mode="anchor", running_month_anchor=running_month_anchor
    )
    st.caption("Denominators — " + " • ".join([f"{lbl}: {denoms.get(lbl,0):,}" for lbl in labels]))

    bullet_group("MTD Conversion %", mtd_pct, nums["mtd"], denoms, labels=labels)
    bullet_group("Cohort Conversion %", coh_pct, nums["cohort"], denoms, labels=labels)

    ts = trend_timeseries(
        df_scope, range_start, range_end,
        denom_mode="anchor", running_month_anchor=running_month_anchor,
        create_col=create_col, pay_col=pay_col
    )
    st.altair_chart(trend_chart(ts, "Trend: Leads (bars) vs Enrolments (lines)"), use_container_width=True)

# ----------------------------
# MIS
# ----------------------------
if view == "MIS":
    show_all = st.checkbox("Show all preset periods (Yesterday • Today • Last Month • This Month)", value=False)
    if show_all:
        st.subheader("Preset Periods")
        colA, colB = st.columns(2)
        with colA:
            render_period_block(df_f, "Yesterday", yday, yday, yday, create_col, pay_col, pipeline_col, track)
            st.divider()
            render_period_block(df_f, "Last Month", last_m_start, last_m_end, last_m_start, create_col, pay_col, pipeline_col, track)
        with colB:
            render_period_block(df_f, "Today", today, today, today, create_col, pay_col, pipeline_col, track)
            st.divider()
            render_period_block(df_f, "This Month", this_m_start, this_m_end, this_m_start, create_col, pay_col, pipeline_col, track)
    else:
        tabs = st.tabs(["Yesterday", "Today", "Last Month", "This Month", "Custom"])
        with tabs[0]:
            render_period_block(df_f, "Yesterday", yday, yday, yday, create_col, pay_col, pipeline_col, track)
        with tabs[1]:
            render_period_block(df_f, "Today", today, today, today, create_col, pay_col, pipeline_col, track)
        with tabs[2]:
            render_period_block(df_f, "Last Month", last_m_start, last_m_end, last_m_start, create_col, pay_col, pipeline_col, track)
        with tabs[3]:
            render_period_block(df_f, "This Month", this_m_start, this_m_end, this_m_start, create_col, pay_col, pipeline_col, track)
        with tabs[4]:
            st.markdown("Select a **payments period** and choose the **Conversion% denominator** mode.")
            colc1, colc2 = st.columns(2)
            with colc1: custom_start = st.date_input("Payments period start", value=this_m_start)
            with colc2: custom_end   = st.date_input("Payments period end (inclusive)", value=this_m_end)
            if custom_end < custom_start:
                st.error("Payments period end cannot be before start.")
            else:
                denom_mode = st.radio("Denominator for Conversion%", ["Anchor month", "Custom range"], index=0, horizontal=True)
                if denom_mode == "Anchor month":
                    anchor = st.date_input("Running-month anchor (denominator month)", value=custom_start)
                    mtd_counts, coh_counts = prepare_counts_for_range(df_f, custom_start, custom_end, anchor, create_col, pay_col, pipeline_col)
                    c1, c2 = st.columns(2)
                    with c1: st.altair_chart(bubble_chart_counts("MTD Enrolments (counts)", mtd_counts["Total"], mtd_counts["AI Coding"], mtd_counts["Math"], labels=active_labels(track)), use_container_width=True)
                    with c2: st.altair_chart(bubble_chart_counts("Cohort Enrolments (counts)", coh_counts["Total"], coh_counts["AI Coding"], coh_counts["Math"], labels=active_labels(track)), use_container_width=True)
                    mtd_pct, coh_pct, denoms, nums = prepare_conversion_for_range(df_f, custom_start, custom_end, create_col, pay_col, pipeline_col, denom_mode="anchor", running_month_anchor=anchor)
                    st.caption("Denominators — " + " • ".join([f"{lbl}: {denoms.get(lbl,0):,}" for lbl in active_labels(track)]))
                    bullet_group("MTD Conversion %", mtd_pct, nums["mtd"], denoms, labels=active_labels(track))
                    bullet_group("Cohort Conversion %", coh_pct, nums["cohort"], denoms, labels=active_labels(track))
                    ts = trend_timeseries(df_f, custom_start, custom_end, denom_mode="anchor", running_month_anchor=anchor, create_col=create_col, pay_col=pay_col)
                    st.altair_chart(trend_chart(ts, "Trend: Leads (bars) vs Enrolments (lines)"), use_container_width=True)
                else:
                    cold1, cold2 = st.columns(2)
                    with cold1: denom_start = st.date_input("Denominator start (deals created from)", value=custom_start, key="denom_start")
                    with cold2: denom_end   = st.date_input("Denominator end (deals created to)",   value=custom_end,   key="denom_end")
                    if denom_end < denom_start:
                        st.error("Denominator end cannot be before start.")
                    else:
                        anchor_for_counts = custom_start
                        mtd_counts, coh_counts = prepare_counts_for_range(df_f, custom_start, custom_end, anchor_for_counts, create_col, pay_col, pipeline_col)
                        c1, c2 = st.columns(2)
                        with c1: st.altair_chart(bubble_chart_counts("MTD Enrolments (counts)", mtd_counts["Total"], mtd_counts["AI Coding"], mtd_counts["Math"], labels=active_labels(track)), use_container_width=True)
                        with c2: st.altair_chart(bubble_chart_counts("Cohort Enrolments (counts)", coh_counts["Total"], coh_counts["AI Coding"], coh_counts["Math"], labels=active_labels(track)), use_container_width=True)
                        mtd_pct, coh_pct, denoms, nums = prepare_conversion_for_range(df_f, custom_start, custom_end, create_col, pay_col, pipeline_col, denom_mode="range", denom_start=denom_start, denom_end=denom_end)
                        st.caption("Denominators — " + " • ".join([f"{lbl}: {denoms.get(lbl,0):,}" for lbl in active_labels(track)]))
                        bullet_group("MTD Conversion %", mtd_pct, nums["mtd"], denoms, labels=active_labels(track))
                        bullet_group("Cohort Conversion %", coh_pct, nums["cohort"], denoms, labels=active_labels(track))
                        ts = trend_timeseries(df_f, custom_start, custom_end, denom_mode="range", denom_start=denom_start, denom_end=denom_end, create_col=create_col, pay_col=pay_col)
                        st.altair_chart(trend_chart(ts, "Trend: Leads (bars) vs Enrolments (lines)"), use_container_width=True)

# ----------------------------
# NEW: Trend & Analysis
# ----------------------------
def group_trend_analysis(
    df_scope: pd.DataFrame,
    group_col: str,
    level: str,
    pay_start: date,
    pay_end: date,
    create_start: date,
    create_end: date,
    create_col: str,
    pay_col: str,
) -> pd.DataFrame:
    """Return table with Count_Payments, Count_Created, Conversion% per group."""
    d = df_scope.copy()
    d["_create_dt"] = coerce_datetime(d[create_col]).dt.date
    d["_pay_dt"]    = coerce_datetime(d[pay_col]).dt.date

    pay_mask = d["_pay_dt"].between(pay_start, pay_end)
    create_mask = d["_create_dt"].between(create_start, create_end)

    # Numerator definition
    if level == "MTD":
        num_mask = pay_mask & create_mask
    else:  # Cohort
        num_mask = pay_mask

    # Denominator definition
    den_mask = create_mask

    # Produce grouped counts
    g_vals = d[group_col].astype(str).fillna("Unknown")

    payments_by_group = g_vals[num_mask].value_counts().rename("Count_Payments")
    created_by_group  = g_vals[den_mask].value_counts().rename("Count_Created")

    all_groups = sorted(set(payments_by_group.index) | set(created_by_group.index))
    result = pd.DataFrame({"Group": all_groups}).set_index("Group")

    result = result.join(payments_by_group, how="left").join(created_by_group, how="left").fillna(0.0)
    result["Count_Payments"] = result["Count_Payments"].astype(int)
    result["Count_Created"]  = result["Count_Created"].astype(int)

    def pct(n, d):
        if d <= 0:
            return 0.0
        return round(100.0 * n / d, 1)

    result["Conversion%"] = [
        pct(result.loc[g, "Count_Payments"], result.loc[g, "Count_Created"]) for g in result.index
    ]

    result = result.reset_index()
    return result.sort_values(["Count_Payments", "Count_Created"], ascending=[False, False])

if view == "Trend & Analysis":
    st.subheader("Trend & Analysis – Grouped Drilldowns")

    # Page-level filters (scoped to this page, start from df_f)
    df_page = df_f.copy()

    # Build option helpers safely
    def opts(series: pd.Series):
        return ["All"] + sorted([str(v) for v in series.dropna().unique()])

    with st.container():
        col1, col2, col3, col4 = st.columns([1.2, 1.2, 1.2, 1.0])

        with col1:
            group_by_label = st.selectbox("Group by", ["Academic Counsellor", "Country", "Deal Source"], index=0)
        with col2:
            level = st.radio("Level", ["MTD", "Cohort"], index=0, horizontal=True)
        with col3:
            pay_start = st.date_input("Payments period start", value=this_m_start, key="ta_pay_start")
        with col4:
            pay_end   = st.date_input("Payments period end (inclusive)", value=this_m_end, key="ta_pay_end")

        col5, col6 = st.columns(2)
        with col5:
            create_start = st.date_input("Create-date period start", value=this_m_start, key="ta_create_start")
        with col6:
            create_end   = st.date_input("Create-date period end (inclusive)", value=this_m_end, key="ta_create_end")

        st.caption("Below filters apply **only on this page** in addition to the global filters above.")

        f1, f2, f3 = st.columns(3)
        if counsellor_col:
            with f1:
                page_counsellors = st.multiselect("Academic Counsellor (page)", options=opts(df[counsellor_col]), default=["All"])
        else:
            page_counsellors = []
        if country_col:
            with f2:
                page_countries = st.multiselect("Country (page)", options=opts(df[country_col]), default=["All"])
        else:
            page_countries = []
        if source_col:
            with f3:
                page_sources = st.multiselect("JetLearn Deal Source (page)", options=opts(df[source_col]), default=["All"])
        else:
            page_sources = []

    # apply page-level filters
    if counsellor_col and page_counsellors and "All" not in page_counsellors:
        df_page = df_page[df_page[counsellor_col].astype(str).isin(page_counsellors)]
    if country_col and page_countries and "All" not in page_countries:
        df_page = df_page[df_page[country_col].astype(str).isin(page_countries)]
    if source_col and page_sources and "All" not in page_sources:
        df_page = df_page[df_page[source_col].astype(str).isin(page_sources)]

    # validation
    if pay_end < pay_start:
        st.error("Payments period end cannot be before start.")
    elif create_end < create_start:
        st.error("Create-date period end cannot be before start.")
    else:
        # Map label to actual column
        group_col_map = {
            "Academic Counsellor": counsellor_col,
            "Country": country_col,
            "Deal Source": source_col,
        }
        group_col = group_col_map[group_by_label]

        if not group_col:
            st.warning(f"Column for '{group_by_label}' not found in your data.")
        else:
            tbl = group_trend_analysis(
                df_page, group_col, level,
                pay_start, pay_end,
                create_start, create_end,
                create_col, pay_col
            )

            # Output table
            st.markdown("### Output")
            if tbl.empty:
                st.info("No rows match the selected filters and date ranges.")
            else:
                # Nicely format
                show = tbl.copy()
                show["Conversion%"] = show["Conversion%"].map(lambda x: f"{x:.1f}%")
                st.dataframe(show.rename(columns={"Group": group_by_label}), use_container_width=True)

                # Quick charts
                st.markdown("#### Quick charts")
                c1, c2 = st.columns(2)

                with c1:
                    top_pay = tbl.nlargest(15, "Count_Payments")
                    chart1 = alt.Chart(top_pay).mark_bar().encode(
                        x=alt.X("Count_Payments:Q", title="Count of Payment Received Date"),
                        y=alt.Y("Group:N", sort="-x", title=group_by_label),
                        tooltip=[alt.Tooltip("Group:N"), alt.Tooltip("Count_Payments:Q")]
                    ).properties(height=360, title="Top groups by Payments")
                    st.altair_chart(chart1, use_container_width=True)

                with c2:
                    top_conv = tbl.copy()
                    chart2 = alt.Chart(top_conv).mark_bar().encode(
                        x=alt.X("Conversion%:Q", title="Conversion %"),
                        y=alt.Y("Group:N", sort="-x", title=group_by_label),
                        tooltip=[alt.Tooltip("Group:N"), alt.Tooltip("Conversion%:Q")]
                    ).properties(height=360, title="Conversion% by group")
                    st.altair_chart(chart2, use_container_width=True)

                # Download
                csv = tbl.to_csv(index=False).encode("utf-8")
                st.download_button("Download CSV (Trend & Analysis)", data=csv, file_name="trend_analysis.csv", mime="text/csv")

# ----------------------------
# Predictibility (A/B/C) + Model Accuracy (tied to lookback)
# ----------------------------
if view == "Predictibility":
    st.subheader("Predictibility – Running Month Enrolment Forecast")
    st.caption(
        "A = payments received **to date** in the running month. "
        "B = forecast for remaining days from **same-month created** deals (lookback daily rate). "
        "C = forecast for remaining days from **previous-months created** deals (lookback daily rate). "
        "Projected month-end = A + B + C."
    )

    colp1, colp2, colp3 = st.columns([1,1,2])
    with colp1:
        lookback = st.selectbox("Lookback window (months)", [3, 6, 12], index=0)
    with colp2:
        st.markdown("**Averaging:** Recency-weighted")
        st.caption("Uses weights 1..K across the last K pay-months (most recent has highest weight).")
        weighted = True
    with colp3:
        st.info("Daily rates are computed per source from the last K pay-months (excluding current).")

    cur_start, cur_end = month_bounds(today)
    d_preview = add_month_cols(df_f, create_col, pay_col)
    cur_period = pd.Period(today, freq="M")
    in_cur_pay = d_preview["_pay_m"] == cur_period
    st.caption(f"Payments found this month (after filters): **{int(in_cur_pay.sum()):,}**")

    tbl, totals = predict_running_month(df_f, create_col, pay_col, source_col, lookback, weighted, today)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>A · Actual to date</div><div class='kpi-value' style='color:{PALETTE['A_actual']}'>{totals['A_Actual_ToDate']:.1f}</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>B · Remaining (same-month)</div><div class='kpi-value' style='color:{PALETTE['Rem_same']}'>{totals['B_Remaining_SameMonth']:.1f}</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>C · Remaining (prev-months)</div><div class='kpi-value' style='color:{PALETTE['Rem_prev']}'>{totals['C_Remaining_PrevMonths']:.1f}</div></div>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Projected Month-End</div><div class='kpi-value' style='color:{PALETTE['Total']}'>{totals['Projected_MonthEnd_Total']:.1f}</div><div class='kpi-sub'>A + B + C</div></div>", unsafe_allow_html=True)

    st.altair_chart(predict_chart_stacked(tbl), use_container_width=True)

    with st.expander("Detailed table (by source)"):
        show_cols = ["Source","A_Actual_ToDate","B_Remaining_SameMonth","C_Remaining_PrevMonths",
                     "Projected_MonthEnd_Total","Rate_Same_Daily","Rate_Prev_Daily","Remaining_Days"]
        if not tbl.empty:
            view_tbl = tbl[show_cols].copy()
            for c in ["B_Remaining_SameMonth","C_Remaining_PrevMonths","Projected_MonthEnd_Total","Rate_Same_Daily","Rate_Prev_Daily"]:
                view_tbl[c] = view_tbl[c].astype(float).round(3)
            st.dataframe(view_tbl, use_container_width=True)
            csv = view_tbl.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csv, file_name="predictibility_by_source.csv", mime="text/csv")
        else:
            st.info("No data in scope for the running month after filters.")

    st.subheader("Model Accuracy")
    st.caption(f"Accuracy is computed using a rolling backtest over the same lookback window you selected above (**{lookback} months**).")

    bt, metrics = backtest_accuracy(
        df_f, create_col, pay_col, source_col,
        lookback=lookback, weighted=weighted,
        backtest_months=lookback,
        today=today
    )

    acc_pct = np.nan
    if not pd.isna(metrics.get("WAPE", np.nan)):
        acc_pct = max(0.0, min(100.0, (1.0 - metrics["WAPE"]) * 100.0))
    elif not pd.isna(metrics.get("MAPE", np.nan)):
        acc_pct = max(0.0, min(100.0, (1.0 - metrics["MAPE"]) * 100.0))

    st.markdown(
        f"<div class='kpi-card'><div class='kpi-title'>Model Accuracy (100 − WAPE)</div>"
        f"<div class='kpi-value'>{'–' if pd.isna(acc_pct) else f'{acc_pct:.1f}%'}"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    show_details = st.checkbox("Show detailed metrics", value=False)
    if show_details:
        m1, m2, m3, m4, m5 = st.columns(5)
        def fmt(x, pct=False):
            if pd.isna(x): return "–"
            return f"{x*100:.1f}%" if pct else f"{x:.2f}"
        with m1: st.markdown(f"<div class='kpi-card'><div class='kpi-title'>MAPE</div><div class='kpi-value'>{fmt(metrics['MAPE'], pct=True)}</div></div>", unsafe_allow_html=True)
        with m2: st.markdown(f"<div class='kpi-card'><div class='kpi-title'>WAPE</div><div class='kpi-value'>{fmt(metrics['WAPE'], pct=True)}</div></div>", unsafe_allow_html=True)
        with m3: st.markdown(f"<div class='kpi-card'><div class='kpi-title'>MAE</div><div class='kpi-value'>{fmt(metrics['MAE'])}</div></div>", unsafe_allow_html=True)
        with m4: st.markdown(f"<div class='kpi-card'><div class='kpi-title'>RMSE</div><div class='kpi-value'>{fmt(metrics['RMSE'])}</div></div>", unsafe_allow_html=True)
        with m5: st.markdown(f"<div class='kpi-card'><div class='kpi-title'>R²</div><div class='kpi-value'>{fmt(metrics['R2'])}</div></div>", unsafe_allow_html=True)

        if bt.empty:
            st.info("Not enough historical data to backtest with the chosen settings.")
        else:
            st.altair_chart(accuracy_scatter(bt), use_container_width=True)
            with st.expander("Backtest details"):
                show = bt.copy()
                for c in ["Forecast","Actual","Error","AbsError","SqError"]:
                    show[c] = show[c].round(2)
                if show["APE"].notna().any():
                    show["APE%"] = (show["APE"]*100).round(1)
                st.dataframe(show.drop(columns=["APE"]), use_container_width=True)

# Optional: data preview
with st.expander("Data preview & column mapping", expanded=False):
    st.write({
        "Create Date": create_col,
        "Payment Received Date": pay_col,
        "Pipeline (split)": pipeline_col or "Not found → using heuristic",
        "Academic Counsellor": counsellor_col or "Not found",
        "Country": country_col or "Not found",
        "JetLearn Deal Source": source_col or "Not found",
        "Deal Stage": dealstage_col or "Not found",
    })
    st.dataframe(df.head(20))
