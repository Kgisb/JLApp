import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, date, timedelta

# ----------------------------
# App Config
# ----------------------------
st.set_page_config(
    page_title="JetLearn MIS – Enrolments (MTD & Cover)",
    page_icon="📊",
    layout="wide",
)

# ----------------------------
# Helpers
# ----------------------------
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    # Normalize column names (strip spaces for trailing-space issues)
    df.columns = [c.strip() for c in df.columns]
    return df

def find_col(df: pd.DataFrame, candidates):
    # Try exact then case-insensitive
    for c in candidates:
        if c in df.columns:
            return c
    low = {c.lower(): c for c in df.columns}
    for c in candidates:
        lc = c.lower()
        if lc in low:
            return low[lc]
    return None

def coerce_datetime(series: pd.Series) -> pd.Series:
    # Try multiple common formats, defaulting to dayfirst True for EU/IN style
    s = pd.to_datetime(series, errors="coerce", infer_datetime_format=True, dayfirst=True)
    # If entire column is NaT and there are numeric-like timestamps, try unit='s' then 'ms'
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

# Use Pipeline column explicitly to split AI Coding vs Math
def normalize_pipeline(value: str) -> str:
    if not isinstance(value, str):
        return "Other"
    v = value.strip().lower()
    if "math" in v:
        return "Math"
    if "ai" in v or "coding" in v or "ai-coding" in v or "ai coding" in v:
        return "AI Coding"
    return "Other"

def prepare_counts(df: pd.DataFrame, month_date: date,
                   create_col: str, pay_col: str, pipeline_col: str | None):

    m_start, m_end = month_bounds(month_date)
    df = df.copy()

    df["_create_dt"] = coerce_datetime(df[create_col])
    df["_pay_dt"] = coerce_datetime(df[pay_col])

    # Filter by payment-month
    in_month_pay = df["_pay_dt"].dt.date.between(m_start, m_end)

    # COVER: all payments in month
    cover_df = df.loc[in_month_pay]

    # MTD: payments in month with create date also in month
    in_month_create = df["_create_dt"].dt.date.between(m_start, m_end)
    mtd_df = df.loc[in_month_pay & in_month_create]

    # Pipeline split
    if pipeline_col and pipeline_col in df.columns:
        cover_split = cover_df[pipeline_col].map(normalize_pipeline).fillna("Other")
        mtd_split = mtd_df[pipeline_col].map(normalize_pipeline).fillna("Other")
    else:
        cover_split = pd.Series([], dtype=object)
        mtd_split = pd.Series([], dtype=object)

    cover_counts = {
        "Total": int(len(cover_df)),
        "AI Coding": int((pd.Series(cover_split) == "AI Coding").sum()),
        "Math": int((pd.Series(cover_split) == "Math").sum()),
    }
    mtd_counts = {
        "Total": int(len(mtd_df)),
        "AI Coding": int((pd.Series(mtd_split) == "AI Coding").sum()),
        "Math": int((pd.Series(mtd_split) == "Math").sum()),
    }
    return mtd_counts, cover_counts

def bubble_chart(title: str, total: int, ai_cnt: int, math_cnt: int):
    data = pd.DataFrame({
        "Label": ["Total", "AI Coding", "Math"],
        "Value": [total, ai_cnt, math_cnt],
        "Row": [0, 1, 1],
        "Col": [0.5, 0.33, 0.66],
    })

    base = alt.Chart(data).encode(
        x=alt.X("Col:Q", axis=None, scale=alt.Scale(domain=(0, 1))),
        y=alt.Y("Row:Q", axis=None, scale=alt.Scale(domain=(-0.2, 1.2))),
        tooltip=[alt.Tooltip("Label:N"), alt.Tooltip("Value:Q")],
    )

    circles = base.mark_circle(opacity=0.65).encode(
        size=alt.Size("Value:Q", scale=alt.Scale(range=[300, 6500]), legend=None),
        color=alt.Color("Label:N", legend=None),
    )

    text = base.mark_text(fontWeight="bold", dy=0).encode(
        text=alt.Text("Value:Q")
    )

    return (circles + text).properties(height=340, title=title)

# ----------------------------
# UI – Drawer-ish splash -> MIS
# ----------------------------
with st.sidebar:
    st.header("JetLearn • Navigation")
    view = st.radio("Go to", ["MIS"], index=0)
    st.caption("Tip: Use the month selector on the main panel.")

st.title("📊 JetLearn MIS")
st.write(
    "This dashboard shows **Enrolments (Payments)** at two levels — "
    "**MTD (same-month created)** and **Cover (all payments in month)** — "
    "with a pipeline split across **AI Coding** and **Math** using the **Pipeline** column."
)

default_path = "Master_sheet_DB.csv"  # pre-uploaded in repo
data_src = st.text_input("Data file path", value=default_path, help="CSV path (kept pre-uploaded in the repo).")
df = load_data(data_src)

# Resolve critical columns (strip handled; also allow variants)
create_col = find_col(df, ["Create Date", "Create date", "Create_Date", "Created At"])
pay_col = find_col(df, ["Payment Received Date", "Payment Received date", "Payment_Received_Date", "Payment Date", "Paid At"])
pipeline_col = find_col(df, ["Pipeline"])  # explicit

if not create_col or not pay_col:
    st.error("Could not find required date columns. Ensure the CSV has 'Create Date' and 'Payment Received Date' (or close variants).")
    st.stop()

# Month selector (defaults to current month start)
today = date.today()
selected_month = st.date_input(
    "Select month (any day within that month)",
    value=date(today.year, today.month, 1)
)

# Compute counts
mtd_counts, cover_counts = prepare_counts(df, selected_month, create_col, pay_col, pipeline_col)

if view == "MIS":
    st.subheader("MIS – Enrolment Bubbles")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### MTD (Same-Month Created & Paid)")
        chart_mtd = bubble_chart(
            "MTD Enrolments",
            total=mtd_counts["Total"],
            ai_cnt=mtd_counts["AI Coding"],
            math_cnt=mtd_counts["Math"],
        )
        st.altair_chart(chart_mtd, use_container_width=True)

    with c2:
        st.markdown("### Cover (All Payments in Month)")
        chart_cover = bubble_chart(
            "Cover Enrolments",
            total=cover_counts["Total"],
            ai_cnt=cover_counts["AI Coding"],
            math_cnt=cover_counts["Math"],
        )
        st.altair_chart(chart_cover, use_container_width=True)

    st.markdown("---")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("MTD – Total", mtd_counts["Total"])
    kpi2.metric("MTD – AI Coding", mtd_counts["AI Coding"])
    kpi3.metric("MTD – Math", mtd_counts["Math"])
    kpi4.metric("Cover – Total", cover_counts["Total"])

    with st.expander("Data preview & column mapping"):
        st.write("Detected columns:")
        st.write({
            "Create Date": create_col,
            "Payment Received Date": pay_col,
            "Pipeline (split)": pipeline_col or "Not found → falling back to heuristic",
        })
        st.dataframe(df.head(20))