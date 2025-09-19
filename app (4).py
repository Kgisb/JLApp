import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, date, timedelta

st.set_page_config(
    page_title="JetLearn MIS – Enrolments (MTD & Cohort)",
    page_icon="📊",
    layout="wide",
)

# ----------------------------
# Helpers
# ----------------------------
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    # normalize columns (strip spaces)
    df.columns = [c.strip() for c in df.columns]
    return df

def find_col(df: pd.DataFrame, candidates):
    """Find a column by name with case-insensitive / variant matching."""
    for c in candidates:
        if c in df.columns:
            return c
    low = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None

def coerce_datetime(series: pd.Series) -> pd.Series:
    """Parse dates robustly with day-first and unix fallback."""
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
    """Apply 'All' or selected filters safely."""
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

    # Pipeline split
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
# UI
# ----------------------------
with st.sidebar:
    st.header("JetLearn • Navigation")
    view = st.radio("Go to", ["MIS"], index=0)
    st.caption("Use the quick period tabs and filters on the main panel.")

st.title("📊 JetLearn MIS")
st.write(
    "Shows **Enrolments (Payments)** at two levels — **MTD (same-month created)** and **Cohort (payments in period)** — "
    "split by **Pipeline** into **AI Coding** and **Math**."
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
    # Helper to prep options with "All" at top
    def prep_options(series: pd.Series):
        vals = sorted([str(v) for v in series.dropna().unique()])
        return ["All"] + vals

    # Academic Counsellor
    if counsellor_col:
        counsellor_opts = prep_options(df[counsellor_col])
        sel_counsellors = st.multiselect("Academic Counsellor", options=counsellor_opts, default=["All"])
    else:
        sel_counsellors = []
        st.info("Academic Counsellor column not found. Skipping this filter.")

    # Country
    if country_col:
        country_opts = prep_options(df[country_col])
        sel_countries = st.multiselect("Country", options=country_opts, default=["All"])
    else:
        sel_countries = []
        st.info("Country column not found. Skipping this filter.")

    # JetLearn Deal Source
    if source_col:
        source_opts = prep_options(df[source_col])
        sel_sources = st.multiselect("JetLearn Deal Source", options=source_opts, default=["All"])
    else:
        sel_sources = []
        st.info("JetLearn Deal Source column not found. Skipping this filter.")

# --- Apply filters BEFORE calculations
df_f = apply_filters(df, counsellor_col, country_col, source_col, sel_counsellors, sel_countries, sel_sources)

# --- Small header KPI for rows in scope
st.caption(f"Rows in scope after filters: **{len(df_f):,}**")

# --- Show-all toggle
show_all = st.checkbox("Show all periods (Yesterday • Today • Last Month • This Month)", value=False, help="Toggle to view all periods together.")

# ----------------------------
# Period sections
# ----------------------------
if view == "MIS":
    if show_all:
        st.subheader("All Periods")
        colA, colB = st.columns(2)

        # Yesterday + Last Month
        with colA:
            st.markdown("#### Yesterday")
            mtd, coh = prepare_counts_for_range(df_f, yday, yday, yday, create_col, pay_col, pipeline_col)
            c1, c2 = st.columns(2)
            with c1:
                st.altair_chart(bubble_chart("MTD (Yesterday context)", mtd["Total"], mtd["AI Coding"], mtd["Math"]), use_container_width=True)
            with c2:
                st.altair_chart(bubble_chart("Cohort (Yesterday payments)", coh["Total"], coh["AI Coding"], coh["Math"]), use_container_width=True)

            st.divider()

            st.markdown("#### Last Month")
            mtd, coh = prepare_counts_for_range(df_f, last_m_start, last_m_end, last_m_start, create_col, pay_col, pipeline_col)
            c1, c2 = st.columns(2)
            with c1:
                st.altair_chart(bubble_chart("MTD (Last Month)", mtd["Total"], mtd["AI Coding"], mtd["Math"]), use_container_width=True)
            with c2:
                st.altair_chart(bubble_chart("Cohort (Last Month)", coh["Total"], coh["AI Coding"], coh["Math"]), use_container_width=True)

        # Today + This Month
        with colB:
            st.markdown("#### Today")
            mtd, coh = prepare_counts_for_range(df_f, today, today, today, create_col, pay_col, pipeline_col)
            c1, c2 = st.columns(2)
            with c1:
                st.altair_chart(bubble_chart("MTD (Today context)", mtd["Total"], mtd["AI Coding"], mtd["Math"]), use_container_width=True)
            with c2:
                st.altair_chart(bubble_chart("Cohort (Today payments)", coh["Total"], coh["AI Coding"], coh["Math"]), use_container_width=True)

            st.divider()

            st.markdown("#### This Month")
            mtd, coh = prepare_counts_for_range(df_f, this_m_start, this_m_end, this_m_start, create_col, pay_col, pipeline_col)
            c1, c2 = st.columns(2)
            with c1:
                st.altair_chart(bubble_chart("MTD (This Month)", mtd["Total"], mtd["AI Coding"], mtd["Math"]), use_container_width=True)
            with c2:
                st.altair_chart(bubble_chart("Cohort (This Month)", coh["Total"], coh["AI Coding"], coh["Math"]), use_container_width=True)

    else:
        tabs = st.tabs(["Yesterday", "Today", "Last Month", "This Month"])

        with tabs[0]:
            st.markdown("### Yesterday")
            mtd, coh = prepare_counts_for_range(df_f, yday, yday, yday, create_col, pay_col, pipeline_col)
            c1, c2 = st.columns(2)
            with c1:
                st.altair_chart(bubble_chart("MTD (Yesterday context)", mtd["Total"], mtd["AI Coding"], mtd["Math"]), use_container_width=True)
            with c2:
                st.altair_chart(bubble_chart("Cohort (Yesterday payments)", coh["Total"], coh["AI Coding"], coh["Math"]), use_container_width=True)

        with tabs[1]:
            st.markdown("### Today")
            mtd, coh = prepare_counts_for_range(df_f, today, today, today, create_col, pay_col, pipeline_col)
            c1, c2 = st.columns(2)
            with c1:
                st.altair_chart(bubble_chart("MTD (Today context)", mtd["Total"], mtd["AI Coding"], mtd["Math"]), use_container_width=True)
            with c2:
                st.altair_chart(bubble_chart("Cohort (Today payments)", coh["Total"], coh["AI Coding"], coh["Math"]), use_container_width=True)

        with tabs[2]:
            st.markdown("### Last Month")
            mtd, coh = prepare_counts_for_range(df_f, last_m_start, last_m_end, last_m_start, create_col, pay_col, pipeline_col)
            c1, c2 = st.columns(2)
            with c1:
                st.altair_chart(bubble_chart("MTD (Last Month)", mtd["Total"], mtd["AI Coding"], mtd["Math"]), use_container_width=True)
            with c2:
                st.altair_chart(bubble_chart("Cohort (Last Month)", coh["Total"], coh["AI Coding"], coh["Math"]), use_container_width=True)

        with tabs[3]:
            st.markdown("### This Month")
            mtd, coh = prepare_counts_for_range(df_f, this_m_start, this_m_end, this_m_start, create_col, pay_col, pipeline_col)
            c1, c2 = st.columns(2)
            with c1:
                st.altair_chart(bubble_chart("MTD (This Month)", mtd["Total"], mtd["AI Coding"], mtd["Math"]), use_container_width=True)
            with c2:
                st.altair_chart(bubble_chart("Cohort (This Month)", coh["Total"], coh["AI Coding"], coh["Math"]), use_container_width=True)

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
