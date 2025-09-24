# app.py
# JetLearn Unified App: MIS + Predictibility + Trend & Analysis + 80-20

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import date, timedelta
from calendar import monthrange
import re

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(
    page_title="JetLearn MIS – Enrolments (MTD & Cohort) + Conversion% + Predictibility + 80-20",
    page_icon="📊",
    layout="wide",
)

# ----------------------------
# Global Styling
# ----------------------------
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
      .kpi-card {
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 12px 14px;
        background: #fafafa;
      }
      .kpi-title {
        color:#6b7280;
        font-size:.9rem;
        margin-bottom:6px;
      }
      .kpi-value {
        font-weight:700;
        font-size:1.4rem;
        color:#111827;
      }
      .kpi-sub {
        color:#6b7280;
        font-size:.85rem;
      }
      .section-title {
        font-weight: 700;
        font-size: 1.05rem;
        margin-top: .25rem;
        margin-bottom: .5rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

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
    return date(d.year, d.month, 1), date(d.year, d.month, monthrange(d.year, d.month)[1])

def last_month_bounds(today: date):
    first_this = date(today.year, today.month, 1)
    last_of_prev = first_this - timedelta(days=1)
    return month_bounds(last_of_prev)

# Invalid deals filter
INVALID_RE = re.compile(r"^\s*1\.2\s*invalid\s*deal[s]?\s*$", re.IGNORECASE)
def exclude_invalid(df: pd.DataFrame, dealstage_col: str | None) -> tuple[pd.DataFrame, int]:
    if not dealstage_col:
        return df, 0
    col = df[dealstage_col].astype(str)
    keep = ~col.apply(lambda x: bool(INVALID_RE.match(x)))
    return df.loc[keep].copy(), int((~keep).sum())

# ----------------------------
# Load data + mappings
# ----------------------------
DATA_PATH = "Master_sheet-DB.csv"
df = load_data(DATA_PATH)

dealstage_col = find_col(df, ["Deal Stage","Stage","Deal Status","Deal Stage Name"])
df, removed_invalid = exclude_invalid(df, dealstage_col)

create_col = find_col(df, ["Create Date","Created At"])
pay_col = find_col(df, ["Payment Received Date","Payment Date","Paid At"])
pipeline_col = find_col(df, ["Pipeline"])
counsellor_col = find_col(df, ["Academic Counsellor","Counsellor"])
country_col = find_col(df, ["Country"])
source_col = find_col(df, ["JetLearn Deal Source","Deal Source","Source"])

# ----------------------------
# Sidebar (global)
# ----------------------------
st.sidebar.header("JetLearn • Navigation")
view = st.sidebar.radio("Go to", ["MIS", "Predictibility", "Trend & Analysis", "80-20"], index=0)
track = st.sidebar.radio("Track", ["Both", "AI Coding", "Math"], index=0)

today = date.today()
yday = today - timedelta(days=1)
last_m_start, last_m_end = last_month_bounds(today)
this_m_start, this_m_end = month_bounds(today)

# ----------------------------
# Header
# ----------------------------
st.title("📊 JetLearn MIS")

# ----------------------------
# VIEWS
# ----------------------------

# --- MIS ---
if view == "MIS":
    st.markdown("### MIS Section")

    df[create_col] = coerce_datetime(df[create_col])
    df[pay_col] = coerce_datetime(df[pay_col])

    filt_df = df.copy()
    if track != "Both" and pipeline_col:
        filt_df = filt_df[filt_df[pipeline_col].astype(str).str.contains(track, case=False, na=False)]

    total_deals = filt_df[create_col].notna().sum()
    total_payments = filt_df[pay_col].notna().sum()
    conv_rate = (total_payments / total_deals * 100) if total_deals > 0 else 0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Total Deals</div><div class="kpi-value">{total_deals}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Total Payments</div><div class="kpi-value">{total_payments}</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Conversion %</div><div class="kpi-value">{conv_rate:.1f}%</div></div>', unsafe_allow_html=True)

    # MTD Conversion
    mtd_start, mtd_end = month_bounds(today)
    mtd_df = filt_df[(filt_df[create_col] >= pd.to_datetime(mtd_start)) & (filt_df[create_col] <= pd.to_datetime(mtd_end))]
    mtd_deals = mtd_df.shape[0]
    mtd_pays = mtd_df[pay_col].notna().sum()
    mtd_conv = (mtd_pays / mtd_deals * 100) if mtd_deals > 0 else 0

    st.markdown("#### 📈 MTD Conversion")
    col1, col2, col3 = st.columns(3)
    col1.metric("Deals Created (MTD)", mtd_deals)
    col2.metric("Payments (MTD)", mtd_pays)
    col3.metric("Conversion % (MTD)", f"{mtd_conv:.1f}%")

    # Cohort Conversion
    st.markdown("#### 👥 Cohort Conversion")
    cohort = (
        filt_df.assign(Month=filt_df[create_col].dt.to_period("M"))
        .groupby("Month")
        .agg(Deals=(create_col, "count"), Payments=(pay_col, lambda x: x.notna().sum()))
        .reset_index()
    )
    cohort["Conversion %"] = (cohort["Payments"] / cohort["Deals"] * 100).round(1)

    chart = alt.Chart(cohort).mark_bar().encode(
        x="Month:T",
        y="Deals:Q",
        tooltip=["Deals", "Payments", "Conversion %"]
    )
    st.altair_chart(chart, use_container_width=True)

# --- Predictibility ---
elif view == "Predictibility":
    st.markdown("### Predictibility Section")

    df[create_col] = coerce_datetime(df[create_col])
    df[pay_col] = coerce_datetime(df[pay_col])
    filt_df = df.copy()

    # Current month
    mtd_start, mtd_end = month_bounds(today)
    mtd_df = filt_df[(filt_df[create_col] >= pd.to_datetime(mtd_start)) & (filt_df[create_col] <= pd.to_datetime(mtd_end))]

    # A: Payments already received this month
    A = mtd_df[pay_col].notna().sum()

    # B: Forecast from same-month created deals
    recent_days = max((today - mtd_start).days, 1)
    daily_rate_same = A / recent_days
    days_remaining = (mtd_end - today).days
    B = int(daily_rate_same * days_remaining)

    # C: Forecast from previous-month deals
    prev_df = filt_df[(filt_df[create_col] < pd.to_datetime(mtd_start))]
    prev_pay = prev_df[pay_col].notna().sum()
    daily_rate_prev = prev_pay / max(len(prev_df), 1)
    C = int(daily_rate_prev * days_remaining)

    projected = A + B + C

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Payments Received (A)", A)
    col2.metric("Forecast Same-Month (B)", B)
    col3.metric("Forecast Prev-Months (C)", C)
    col4.metric("Projected Month-End", projected)

# --- Trend & Analysis ---
elif view == "Trend & Analysis":
    st.markdown("### Trend & Analysis Section")

    df[create_col] = coerce_datetime(df[create_col])
    df[pay_col] = coerce_datetime(df[pay_col])
    filt_df = df.copy()

    trend = (
        filt_df.assign(Date=filt_df[create_col].dt.date)
        .groupby("Date")
        .agg(Deals=(create_col, "count"), Payments=(pay_col, lambda x: x.notna().sum()))
        .reset_index()
    )

    st.markdown("#### 📊 Daily Trend")
    chart = alt.Chart(trend).mark_bar().encode(
        x="Date:T",
        y="Deals:Q",
        color=alt.value("#2563eb"),
        tooltip=["Date", "Deals", "Payments"]
    )
    line = alt.Chart(trend).mark_line(color="red").encode(
        x="Date:T", y="Payments:Q"
    )
    st.altair_chart(chart + line, use_container_width=True)

# --- 80-20 ---
elif view == "80-20":
    st.markdown("### 80-20 Section")

    df[create_col] = coerce_datetime(df[create_col])
    df[pay_col] = coerce_datetime(df[pay_col])
    filt_df = df.copy()

    by_source = (
        filt_df.groupby(source_col)
        .agg(Deals=(create_col, "count"), Payments=(pay_col, lambda x: x.notna().sum()))
        .reset_index()
    )
    by_source["Conversion %"] = (by_source["Payments"] / by_source["Deals"] * 100).round(1)

    st.markdown("#### 📊 Deal Source Pareto")
    chart = alt.Chart(by_source).mark_bar().encode(
        x=alt.X("Deals:Q", title="Deals"),
        y=alt.Y(source_col, sort="-x"),
        tooltip=["Deals", "Payments", "Conversion %"]
    )
    st.altair_chart(chart, use_container_width=True)

    st.dataframe(by_source)
