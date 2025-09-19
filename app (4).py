import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, date, timedelta

st.set_page_config(
    page_title="JetLearn MIS – Enrolments (MTD & Cohort) + Conversion%",
    page_icon="📊",
    layout="wide",
)

# ---------- Global UI styling ----------
st.markdown(
    """
    <style>
      /* Card-like look for Altair charts */
      .stAltairChart {
        border: 1px solid #e5e7eb;            /* gray-200 */
        border-radius: 16px;
        padding: 14px;
        background: #ffffff;
        box-shadow: 0 1px 3px rgba(15,23,42,.08);
      }
      /* Legend pills */
      .legend-pill {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        margin-right: 10px;
        font-weight: 600;
        font-size: 0.9rem;
        color: #111827; /* gray-900 */
      }
      .pill-total { background: #e5e7eb; }    /* gray-200 for Total */
      .pill-ai    { background: #bfdbfe; }    /* blue-200 for AI Coding */
      .pill-math  { background: #bbf7d0; }    /* green-200 for Math */

      /* Section titles */
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

# ---------- Professional color palette ----------
PALETTE = {
    "Total": "#6b7280",      # gray-500
    "AI Coding": "#2563eb",  # blue-600
    "Math": "#16a34a",       # green-600
}

# ----------------------------
# Helpers
# ----------------------------
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]  # strip trailing spaces
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
    if counsellor_col and len(sel_counsellors) > 0:
        if "All" not in sel_counsellors:
            f = f[f[counsellor_col].astype(str).isin(sel_counsellors)]
    if country_col and len(sel_countries) > 0:
        if "All" not in sel_countries:
            f = f[f[country_col].astype(str).isin(sel_countries)]
    if source_col and len(sel_sources) > 0:
        if "All" not in sel_sources:
            f = f[f[source_col].astype(str).isin(sel_sources)]
    return f

# ---------- COUNT LOGIC (existing MIS) ----------
def prepare_counts_for_range(
    df: pd.DataFrame,
    start_d: date,
    end_d: date,
    month_for_mtd: date,
    create_col: str,
    pay_col: str,
    pipeline_col: str | None
):
    """Returns (mtd_counts, cohort_counts) for a given date range.
    - cohort_counts: payments with Payment Received Date between start_d and end_d inclusive.
    - mtd_counts: payments in range AND Create Date in the same calendar month as month_for_mtd.
    """
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
def deals_created_in_running_month(df: pd.DataFrame, running_month_any_date: date, create_col: str) -> int:
    df = df.copy()
    df["_create_dt"] = coerce_datetime(df[create_col])
    m_start, m_end = month_bounds(running_month_any_date)
    return int(df["_create_dt"].dt.date.between(m_start, m_end).sum())

def deals_created_in_window(df: pd.DataFrame, start_d: date, end_d: date, create_col: str) -> int:
    df = df.copy()
    df["_create_dt"] = coerce_datetime(df[create_col])
    return int(df["_create_dt"].dt.date.between(start_d, end_d).sum())

def prepare_conversion_for_range(
    df: pd.DataFrame,
    start_d: date,
    end_d: date,
    create_col: str,
    pay_col: str,
    pipeline_col: str | None,
    denom_mode: str = "running_month",   # "running_month" or "custom_range"
    denom_anchor: date | None = None,    # used when denom_mode == "running_month"
    denom_start: date | None = None,     # used when denom_mode == "custom_range"
    denom_end: date | None = None,       # used when denom_mode == "custom_range"
):
    """
    Returns two dicts with % values (0..100) for MTD% and Cohort% and the denominator used.
      - If denom_mode == "running_month": Denominator = # deals created in month(denom_anchor)
      - If denom_mode == "custom_range":  Denominator = # deals created in [denom_start, denom_end]
      - MTD% numerator: payments in [start_d,end_d] AND created in month(denom_anchor) or in [denom_start, denom_end] respectively
      - Cohort% numerator: payments in [start_d,end_d] (any create date)
    """
    df = df.copy()
    df["_create_dt"] = coerce_datetime(df[create_col])
    df["_pay_dt"] = coerce_datetime(df[pay_col])

    if denom_mode == "custom_range" and denom_start is not None and denom_end is not None:
        denom = deals_created_in_window(df, denom_start, denom_end, create_col)
        in_running_create = df["_create_dt"].dt.date.between(denom_start, denom_end)
    else:
        # default to running month by anchor
        anchor = denom_anchor or date.today()
        denom = deals_created_in_running_month(df, anchor, create_col)
        m_start, m_end = month_bounds(anchor)
        in_running_create = df["_create_dt"].dt.date.between(m_start, m_end)

    if denom == 0:
        zero = {"Total": 0.0, "AI Coding": 0.0, "Math": 0.0}
        return zero, zero, 0

    in_range_pay = df["_pay_dt"].dt.date.between(start_d, end_d)

    # Numerators
    mtd_df = df.loc[in_range_pay & in_running_create]
    cohort_df = df.loc[in_range_pay]

    if pipeline_col and pipeline_col in df.columns:
        mtd_split = mtd_df[pipeline_col].map(normalize_pipeline).fillna("Other")
        cohort_split = cohort_df[pipeline_col].map(normalize_pipeline).fillna("Other")
        mtd_ai = int((pd.Series(mtd_split) == "AI Coding").sum())
        mtd_math = int((pd.Series(mtd_split) == "Math").sum())
        coh_ai = int((pd.Series(cohort_split
