# app_8020.py
# Standalone Streamlit app: 80-20 Pareto + Trajectory (Top Countries × Referral/PM-Search/PM-Social)

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import date, timedelta
from calendar import monthrange
import re

st.set_page_config(page_title="JetLearn 80-20 + Trajectory", page_icon="📈", layout="wide")

# ----------------------------
# Minimal styling
# ----------------------------
st.markdown(
    """
    <style>
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
        font-weight: 700; font-size: 1.05rem;
        margin-top: .25rem; margin-bottom: .5rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Helpers
# ----------------------------
@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df

def find_col(df: pd.DataFrame, candidates) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    low = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None

def to_datetime(series: pd.Series) -> pd.Series:
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

def months_back_list(end_d: date, k: int):
    """Return list of Period('YYYY-MM') covering the last k months ending in end_d's month (inclusive)."""
    p_end = pd.Period(end_d, freq="M")
    return [p_end - i for i in range(k-1, -1, -1)]

INVALID_RE = re.compile(r"^\s*1\.2\s*invalid\s*deal[s]?\s*$", re.IGNORECASE)

def exclude_invalid(df: pd.DataFrame, dealstage_col: str | None) -> tuple[pd.DataFrame, int]:
    if not dealstage_col:
        return df, 0
    col = df[dealstage_col].astype(str)
    keep = ~col.apply(lambda x: bool(INVALID_RE.match(x)))
    return df.loc[keep].copy(), int((~keep).sum())

def build_pareto(df: pd.DataFrame, group_col: str, label: str) -> pd.DataFrame:
    if group_col is None or group_col not in df.columns:
        return pd.DataFrame(columns=[label, "Count", "CumCount", "CumPct", "Tag"])
    tmp = (
        df.assign(_grp=df[group_col].fillna("Unknown").astype(str))
          .groupby("_grp").size().sort_values(ascending=False).rename("Count").reset_index()
          .rename(columns={"_grp": label})
    )
    if tmp.empty:
        return tmp
    tmp["CumCount"] = tmp["Count"].cumsum()
    total = tmp["Count"].sum()
    tmp["CumPct"] = (tmp["CumCount"] / total) * 100.0
    tmp["Tag"] = np.where(tmp["CumPct"] <= 80.0, "Top 80%", "Bottom 20%")
    return tmp

def pareto_chart(tbl: pd.DataFrame, label: str, title: str):
    if tbl.empty:
        return alt.Chart(pd.DataFrame({"x":[],"y":[]}))
    base = alt.Chart(tbl).encode(x=alt.X(f"{label}:N", sort=list(tbl[label])))
    bars = base.mark_bar(opacity=0.85).encode(
        y=alt.Y("Count:Q", axis=alt.Axis(title="Enrollments (count)")),
        tooltip=[alt.Tooltip(f"{label}:N"), alt.Tooltip("Count:Q")]
    )
    line = base.mark_line(point=True).encode(
        y=alt.Y("CumPct:Q", axis=alt.Axis(title="Cumulative %", orient="right")),
        color=alt.value("#16a34a"),
        tooltip=[alt.Tooltip(f"{label}:N"), alt.Tooltip("CumPct:Q", format=".1f")]
    )
    rule80 = alt.Chart(pd.DataFrame({"y":[80.0]})).mark_rule(strokeDash=[4,4]).encode(y="y:Q")
    return alt.layer(bars, line, rule80).resolve_scale(y='independent').properties(title=title, height=360)

def donut_referral_share(df: pd.DataFrame, source_col: str):
    if source_col is None or source_col not in df.columns or df.empty:
        return alt.Chart(pd.DataFrame({"x":[],"y":[]}))
    s = df[source_col].fillna("Unknown").astype(str)
    is_ref = s.str.contains("referr", case=False, na=False)
    pie = pd.DataFrame({
        "Category": ["Referral", "Non-Referral"],
        "Value": [int(is_ref.sum()), int((~is_ref).sum())]
    })
    return alt.Chart(pie).mark_arc(innerRadius=70).encode(
        theta="Value:Q",
        color=alt.Color("Category:N", legend=alt.Legend(orient="bottom")),
        tooltip=["Category:N", "Value:Q"]
    ).properties(title="Referral vs Non-Referral (cohort)")

# ---- Source normalization for the 3 key sources ----
def normalize_key_source(val: str) -> str:
    if not isinstance(val, str):
        return "Other"
    v = val.strip().lower()
    if "referr" in v:
        return "Referral"
    if "pm" in v and "search" in v:
        return "PM - Search"
    if "pm" in v and "social" in v:
        return "PM - Social"
    return "Other"

# ----------------------------
# Load data
# ----------------------------
st.title("📈 80-20 Pareto + Trajectory (Top Countries × Key Sources)")

df_raw = load_csv("Master_sheet-DB.csv")  # file must be in same folder as this app
dealstage_col = find_col(df_raw, ["Deal Stage","Deal stage","Stage","Deal Status","Stage Name","Deal Stage Name"])
create_col    = find_col(df_raw, ["Create Date","Create date","Create_Date","Created At"])
pay_col       = find_col(df_raw, ["Payment Received Date","Payment Received date","Payment_Received_Date","Payment Date","Paid At"])
source_col    = find_col(df_raw, ["JetLearn Deal Source","Deal Source","Source"])
country_col   = find_col(df_raw, ["Country"])

df_clean, removed_invalid = exclude_invalid(df_raw, dealstage_col)
if removed_invalid > 0:
    st.caption(f"Auto-excluded “1.2 Invalid deal(s)”: **{removed_invalid:,}** rows.")

df_clean["_pay_dt"] = to_datetime(df_clean[pay_col])
df_clean["_create_dt"] = to_datetime(df_clean[create_col]) if create_col else pd.NaT
df_clean["_pay_m"] = df_clean["_pay_dt"].dt.to_period("M")

# ----------------------------
# Filters – Month or custom range (COHORT = Payment in range)
# ----------------------------
st.sidebar.header("Scope (Cohort)")
unique_months = df_clean["_pay_dt"].dropna().dt.to_period("M").drop_duplicates().sort_values()
month_labels = [str(p) for p in unique_months]
use_custom = st.sidebar.toggle("Use custom date range", value=False)

if not use_custom and len(month_labels) > 0:
    month_pick = st.sidebar.selectbox("Cohort month (Payment Received)", month_labels, index=len(month_labels)-1)
    y, m = map(int, month_pick.split("-"))
    start_d = date(y, m, 1)
    end_d = date(y, m, monthrange(y, m)[1])
else:
    default_start = df_clean["_pay_dt"].min().date() if df_clean["_pay_dt"].notna().any() else date.today().replace(day=1)
    default_end   = df_clean["_pay_dt"].max().date() if df_clean["_pay_dt"].notna().any() else date.today()
    start_d = st.sidebar.date_input("Start date", value=default_start)
    end_d   = st.sidebar.date_input("End date", value=default_end)
    if end_d < start_d:
        st.error("End date cannot be before start date.")
        st.stop()

# ----------------------------
# Source filter (for 80-20 view only)
# ----------------------------
if source_col:
    all_sources = sorted(df_clean[source_col].dropna().astype(str).unique())
    excl_ref = st.sidebar.checkbox("Exclude Referral (80-20 view)", value=False)
    sources_for_pick = [s for s in all_sources if not (excl_ref and "referr" in s.lower())]
    picked_sources = st.sidebar.multiselect("Include Deal Sources (80-20 view)", options=sources_for_pick, default=sources_for_pick)
else:
    picked_sources = None

# ----------------------------
# Apply cohort filter
# ----------------------------
scope_mask = df_clean["_pay_dt"].dt.date.between(start_d, end_d)
df_cohort = df_clean.loc[scope_mask].copy()
if picked_sources is not None and source_col:
    df_cohort = df_cohort[df_cohort[source_col].astype(str).isin(picked_sources)]

# ----------------------------
# 80-20 KPIs + Charts (same as earlier)
# ----------------------------
st.markdown("<div class='section-title'>Cohort KPIs</div>", unsafe_allow_html=True)
total_enr = int(len(df_cohort))
if source_col and source_col in df_cohort.columns:
    ref_cnt = int(df_cohort[source_col].fillna("").str.contains("referr", case=False).sum())
else:
    ref_cnt = 0
ref_pct = (ref_cnt/total_enr*100.0) if total_enr > 0 else 0.0

src_tbl = build_pareto(df_cohort, source_col, "Deal Source") if total_enr > 0 else pd.DataFrame()
cty_tbl = build_pareto(df_cohort, country_col, "Country") if total_enr > 0 else pd.DataFrame()
n_sources_80 = int((src_tbl["CumPct"] <= 80).sum()) if not src_tbl.empty else 0
n_countries_80 = int((cty_tbl["CumPct"] <= 80).sum()) if not cty_tbl.empty else 0

k1, k2, k3, k4 = st.columns(4)
with k1: st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Enrollments</div><div class='kpi-value'>{total_enr:,}</div><div class='kpi-sub'>{start_d} → {end_d}</div></div>", unsafe_allow_html=True)
with k2: st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Referral %</div><div class='kpi-value'>{ref_pct:.1f}%</div><div class='kpi-sub'>{ref_cnt:,} of {total_enr:,}</div></div>", unsafe_allow_html=True)
with k3: st.markdown(f"<div class='kpi-card'><div class='kpi-title'># Sources for 80%</div><div class='kpi-value'>{n_sources_80}</div></div>", unsafe_allow_html=True)
with k4: st.markdown(f"<div class='kpi-card'><div class='kpi-title'># Countries for 80%</div><div class='kpi-value'>{n_countries_80}</div></div>", unsafe_allow_html=True)

c1, c2 = st.columns([2,1])
with c1: st.altair_chart(pareto_chart(src_tbl, "Deal Source", "Pareto – Enrolments by Deal Source"), use_container_width=True)
with c2: st.altair_chart(donut_referral_share(df_cohort, source_col), use_container_width=True)
st.altair_chart(pareto_chart(cty_tbl, "Country", "Pareto – Enrolments by Country"), use_container_width=True)

# ----------------------------
# NEW: Trajectory – Top Countries × (Referral, PM-Search, PM-Social)
# ----------------------------
st.markdown("### Trajectory – Top Countries × Referral / PM - Search / PM - Social")

# Controls for trajectory
col_t1, col_t2 = st.columns(2)
with col_t1:
    trailing_k = st.selectbox("Trailing window (months)", [3, 6, 12], index=0,
                              help="Build monthly trajectory ending at your selected end date.")
with col_t2:
    top_k = st.selectbox("Top countries (by cohort enrolments)", [5, 7], index=0)

# Determine trailing months list up to end_d
months_list = months_back_list(end_d, trailing_k)
months_str = [str(p) for p in months_list]

# Build a working frame limited to these months (by pay-month)
df_trail = df_clean[df_clean["_pay_m"].isin(months_list)].copy()
df_trail["_key_source"] = df_trail[source_col].apply(normalize_key_source) if source_col else "Other"

# Determine Top K countries by cohort counts over selected trailing window (all sources)
if country_col:
    cty_counts = (df_trail.groupby(country_col).size().sort_values(ascending=False))
    top_countries = cty_counts.head(top_k).index.astype(str).tolist()
else:
    top_countries = []

# Monthly totals across ALL countries (denominator = total business per month)
monthly_total = df_trail.groupby("_pay_m").size().rename("TotalAll").reset_index()

# Monthly counts for (Top country × key source)
if top_countries and source_col and country_col:
    mcs = (
        df_trail[df_trail[country_col].astype(str).isin(top_countries)]
        .groupby(["_pay_m", country_col, "_key_source"])
        .size()
        .rename("Cnt")
        .reset_index()
    )
else:
    mcs = pd.DataFrame(columns=["_pay_m", country_col if country_col else "Country", "_key_source", "Cnt"])

# Merge to compute % of OVERALL business
if not mcs.empty:
    mcs = mcs.merge(monthly_total, on="_pay_m", how="left")
    mcs["PctOfOverall"] = np.where(mcs["TotalAll"]>0, mcs["Cnt"]/mcs["TotalAll"]*100.0, 0.0)
    # Keep only the 3 key sources
    mcs = mcs[mcs["_key_source"].isin(["Referral", "PM - Search", "PM - Social"])]
    # Ensure month order
    mcs["_pay_m_str"] = mcs["_pay_m"].astype(str)
    mcs["_pay_m_str"] = pd.Categorical(mcs["_pay_m_str"], categories=months_str, ordered=True)

# Chart 1: Faceted stacked bars – each country as a facet, x=month, y=% of overall, color=source
if not mcs.empty:
    facet_chart = alt.Chart(mcs).mark_bar(opacity=0.9).encode(
        x=alt.X("_pay_m_str:N", title="Month", sort=months_str),
        y=alt.Y("PctOfOverall:Q", title="% of overall business", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("_key_source:N", title="Source",
                        sort=["Referral","PM - Search","PM - Social"]),
        tooltip=[
            alt.Tooltip("_pay_m_str:N", title="Month"),
            alt.Tooltip(f"{country_col}:N", title="Country") if country_col else alt.Tooltip("_pay_m_str:N"),
            alt.Tooltip("_key_source:N", title="Source"),
            alt.Tooltip("Cnt:Q", title="Count"),
            alt.Tooltip("PctOfOverall:Q", title="% of overall", format=".1f")
        ]
    ).properties(height=220).facet(
        column=alt.Column(f"{country_col}:N", title="Top Countries", sort=top_countries)
    )
    st.altair_chart(facet_chart, use_container_width=True)
else:
    st.info("No data found for the selected trailing window and columns to build the trajectory.")

# Chart 2: Overall lines across top countries – % of overall by source (summing the top countries)
if not mcs.empty:
    overall = (
        mcs.groupby(["_pay_m_str","_key_source"], as_index=False)
           .agg({"Cnt":"sum", "TotalAll":"first"})
    )
    overall["PctOfOverall"] = np.where(overall["TotalAll"]>0, overall["Cnt"]/overall["TotalAll"]*100.0, 0.0)
    lines = alt.Chart(overall).mark_line(point=True).encode(
        x=alt.X("_pay_m_str:N", title="Month", sort=months_str),
        y=alt.Y("PctOfOverall:Q", title="% of overall business (Top countries)", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("_key_source:N", title="Source",
                        sort=["Referral","PM - Search","PM - Social"]),
        tooltip=[
            alt.Tooltip("_pay_m_str:N", title="Month"),
            alt.Tooltip("_key_source:N", title="Source"),
            alt.Tooltip("PctOfOverall:Q", title="% of overall", format=".1f")
        ]
    ).properties(title="Overall contribution of key sources (restricted to Top countries)", height=320)
    st.altair_chart(lines, use_container_width=True)

# ----------------------------
# Tables + Downloads
# ----------------------------
st.markdown("<div class='section-title'>Tables</div>", unsafe_allow_html=True)
tabs = st.tabs(["Deal Source 80-20", "Country 80-20", "Cohort Rows", "Trajectory table"])

with tabs[0]:
    if src_tbl.empty: st.info("No enrollments in scope.")
    else:
        st.dataframe(src_tbl, use_container_width=True)
        st.download_button("Download CSV – Deal Source Pareto", src_tbl.to_csv(index=False).encode("utf-8"),
                           "pareto_deal_source.csv", "text/csv")

with tabs[1]:
    if cty_tbl.empty: st.info("No enrollments in scope.")
    else:
        st.dataframe(cty_tbl, use_container_width=True)
        st.download_button("Download CSV – Country Pareto", cty_tbl.to_csv(index=False).encode("utf-8"),
                           "pareto_country.csv", "text/csv")

with tabs[2]:
    show_cols = []
    if create_col: show_cols.append(create_col)
    if pay_col: show_cols.append(pay_col)
    if source_col: show_cols.append(source_col)
    if country_col: show_cols.append(country_col)
    preview = df_cohort[show_cols].copy() if show_cols else df_cohort.copy()
    st.dataframe(preview.head(1000), use_container_width=True)
    st.download_button("Download CSV – Cohort subset", preview.to_csv(index=False).encode("utf-8"),
                       "cohort_subset.csv", "text/csv")

with tabs[3]:
    if not mcs.empty:
        tcols = [c for c in [country_col, "_pay_m_str", "_key_source", "Cnt", "TotalAll", "PctOfOverall"] if c in mcs.columns or c in ["_pay_m_str","_key_source","Cnt","TotalAll","PctOfOverall"]]
        show = mcs.rename(columns={country_col: "Country"})[["Country","_pay_m_str","_key_source","Cnt","TotalAll","PctOfOverall"]]
        show = show.sort_values(["Country","_pay_m_str","_key_source"])
        st.dataframe(show, use_container_width=True)
        st.download_button("Download CSV – Trajectory", show.to_csv(index=False).encode("utf-8"),
                           "trajectory_top_countries_sources.csv", "text/csv")
    else:
        st.info("No trajectory table for the current selection.")
