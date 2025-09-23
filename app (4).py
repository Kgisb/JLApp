# app_8020.py
# Streamlit demo: Pareto (80-20) analysis of enrollments by Deal Source and Country

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import date
from calendar import monthrange
import re

st.set_page_config(page_title="JetLearn 80-20 Pareto – Enrolments", page_icon="📈", layout="wide")

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
        # try unix seconds/ms fallback
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

INVALID_RE = re.compile(r"^\s*1\.2\s*invalid\s*deal[s]?\s*$", re.IGNORECASE)

def exclude_invalid(df: pd.DataFrame, dealstage_col: str | None) -> tuple[pd.DataFrame, int]:
    if not dealstage_col:
        return df, 0
    col = df[dealstage_col].astype(str)
    keep = ~col.apply(lambda x: bool(INVALID_RE.match(x)))
    return df.loc[keep].copy(), int((~keep).sum())

def build_pareto(df: pd.DataFrame, group_col: str, label: str) -> pd.DataFrame:
    """
    df: already cohort-filtered (Payment Received in period)
    group_col: column to group by (e.g., Deal Source, Country)
    label: nice name for display
    Returns: table with columns [Label, Count, CumCount, CumPct, Tag]
    """
    if group_col is None or group_col not in df.columns:
        return pd.DataFrame(columns=[label, "Count", "CumCount", "CumPct", "Tag"])

    tmp = (
        df.assign(_grp=df[group_col].fillna("Unknown").astype(str))
          .groupby("_grp").size().sort_values(ascending=False).rename("Count").reset_index()
          .rename(columns={"_grp": label})
    )
    if tmp.empty:
        tmp["CumCount"] = []
        tmp["CumPct"] = []
        tmp["Tag"] = []
        return tmp

    tmp["CumCount"] = tmp["Count"].cumsum()
    total = tmp["Count"].sum()
    tmp["CumPct"] = (tmp["CumCount"] / total) * 100.0
    tmp["Tag"] = np.where(tmp["CumPct"] <= 80.0, "Top 80%", "Bottom 20%")
    return tmp

def pareto_chart(tbl: pd.DataFrame, label: str, title: str):
    """Bar = Count (left axis), Line = CumPct (right axis)."""
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

    rule80 = alt.Chart(pd.DataFrame({"y":[80.0]})).mark_rule(strokeDash=[4,4]).encode(
        y=alt.Y("y:Q")
    )

    chart = alt.layer(bars, line, rule80).resolve_scale(y='independent').properties(
        title=title, height=360
    )
    return chart

def donut_referral_share(df: pd.DataFrame, source_col: str):
    if source_col is None or source_col not in df.columns or df.empty:
        return alt.Chart(pd.DataFrame({"x":[],"y":[]}))

    s = df[source_col].fillna("Unknown").astype(str)
    is_ref = s.str.contains("referr", case=False, na=False)
    pie = pd.DataFrame({
        "Category": ["Referral", "Non-Referral"],
        "Value": [int(is_ref.sum()), int((~is_ref).sum())]
    })

    chart = alt.Chart(pie).mark_arc(innerRadius=70).encode(
        theta="Value:Q",
        color=alt.Color("Category:N", legend=alt.Legend(orient="bottom")),
        tooltip=[alt.Tooltip("Category:N"), alt.Tooltip("Value:Q")]
    ).properties(title="Referral vs Non-Referral (cohort)")
    return chart

# ----------------------------
# Sidebar – Data & Scope
# ----------------------------
st.title("📈 80-20 Pareto – Enrolments by Source & Country")

with st.sidebar:
    st.header("Data")
    data_path = st.text_input("CSV path", value="/mnt/data/Master_sheet-DB.csv",
                              help="Change if your file lives elsewhere.")
    df_raw = load_csv(data_path)

    # Detect key columns
    dealstage_col = find_col(df_raw, ["Deal Stage","Deal stage","Stage","Deal Status","Stage Name","Deal Stage Name"])
    create_col    = find_col(df_raw, ["Create Date","Create date","Create_Date","Created At"])
    pay_col       = find_col(df_raw, ["Payment Received Date","Payment Received date","Payment_Received_Date","Payment Date","Paid At"])
    source_col    = find_col(df_raw, ["JetLearn Deal Source","Deal Source","Source"])
    country_col   = find_col(df_raw, ["Country"])

    if not pay_col:
        st.error("Could not find a Payment Received Date column. Please check your CSV.")
        st.stop()

    df_clean, removed_invalid = exclude_invalid(df_raw, dealstage_col)
    if removed_invalid > 0:
        st.caption(f"Auto-excluded “1.2 Invalid deal(s)”: **{removed_invalid:,}** rows.")

    # Parse dates
    df_clean["_pay_dt"] = to_datetime(df_clean[pay_col])
    df_clean["_create_dt"] = to_datetime(df_clean[create_col]) if create_col else pd.NaT

    # Month / Custom selector based on Payment dates (cohort)
    st.header("Scope (Cohort)")
    unique_months = (
        df_clean["_pay_dt"].dropna().dt.to_period("M").drop_duplicates().sort_values()
    )
    month_labels = [str(p) for p in unique_months]
    use_custom = st.toggle("Use custom date range", value=False, help="Off = pick a month. On = choose dates.")

    if not use_custom:
        if len(month_labels) == 0:
            st.warning("No valid Payment Received dates found. Switch to custom range or check data.")
            st.stop()
        month_pick = st.selectbox("Cohort month (Payment Received in this month)", month_labels, index=len(month_labels)-1)
        # compute bounds for chosen month
        y, m = map(int, month_pick.split("-"))
        from datetime import date as _d
        start_d = _d(y, m, 1)
        end_d = _d(y, m, monthrange(y, m)[1])
    else:
        from datetime import timedelta
        default_start = df_clean["_pay_dt"].min().date() if df_clean["_pay_dt"].notna().any() else date.today().replace(day=1)
        default_end   = df_clean["_pay_dt"].max().date() if df_clean["_pay_dt"].notna().any() else date.today()
        c1, c2 = st.columns(2)
        with c1: start_d = st.date_input("Start date (inclusive)", value=default_start)
        with c2: end_d   = st.date_input("End date (inclusive)", value=default_end)
        if end_d < start_d:
            st.error("End date cannot be before start date.")
            st.stop()

    # Source filter controls
    st.header("Deal Source Filter")
    if source_col and source_col in df_clean.columns:
        all_sources = sorted(df_clean[source_col].dropna().astype(str).unique().tolist())
        excl_ref = st.checkbox("Quick toggle: Exclude Referral", value=False)
        if excl_ref:
            # filter out anything containing "referr"
            all_sources_for_picker = [s for s in all_sources if "referr" not in s.lower()]
            default_pick = all_sources_for_picker
        else:
            all_sources_for_picker = all_sources
            default_pick = all_sources_for_picker

        picked_sources = st.multiselect("Include these sources", options=all_sources_for_picker, default=default_pick)
    else:
        picked_sources = None
        st.info("Deal Source column not found; source filtering is disabled.")

# ----------------------------
# Apply scope + filters
# ----------------------------
scope_mask = df_clean["_pay_dt"].dt.date.between(start_d, end_d)
df_cohort = df_clean.loc[scope_mask].copy()

if picked_sources is not None:
    df_cohort = df_cohort[df_cohort[source_col].astype(str).isin(picked_sources)]

# ----------------------------
# KPIs
# ----------------------------
st.markdown("<div class='section-title'>Cohort KPIs</div>", unsafe_allow_html=True)
total_enr = int(len(df_cohort))
if source_col and source_col in df_cohort.columns:
    s = df_cohort[source_col].fillna("Unknown").astype(str)
    ref_cnt = int(s.str.contains("referr", case=False, na=False).sum())
else:
    ref_cnt = 0

ref_pct = (ref_cnt/total_enr*100.0) if total_enr > 0 else 0.0

# Pareto tables
src_tbl = build_pareto(df_cohort, source_col, "Deal Source") if total_enr > 0 else pd.DataFrame(columns=["Deal Source","Count","CumCount","CumPct","Tag"])
cty_tbl = build_pareto(df_cohort, country_col, "Country")    if total_enr > 0 else pd.DataFrame(columns=["Country","Count","CumCount","CumPct","Tag"])

n_sources_80 = int((src_tbl["CumPct"] <= 80).sum()) if not src_tbl.empty else 0
n_countries_80 = int((cty_tbl["CumPct"] <= 80).sum()) if not cty_tbl.empty else 0

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Enrollments (cohort)</div><div class='kpi-value'>{total_enr:,}</div><div class='kpi-sub'>{start_d} → {end_d}</div></div>", unsafe_allow_html=True)
with k2:
    st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Referral %</div><div class='kpi-value'>{ref_pct:.1f}%</div><div class='kpi-sub'>{ref_cnt:,} of {total_enr:,}</div></div>", unsafe_allow_html=True)
with k3:
    st.markdown(f"<div class='kpi-card'><div class='kpi-title'># of Sources for 80%</div><div class='kpi-value'>{n_sources_80}</div><div class='kpi-sub'>Top contributors</div></div>", unsafe_allow_html=True)
with k4:
    st.markdown(f"<div class='kpi-card'><div class='kpi-title'># of Countries for 80%</div><div class='kpi-value'>{n_countries_80}</div><div class='kpi-sub'>Top contributors</div></div>", unsafe_allow_html=True)

# ----------------------------
# Charts
# ----------------------------
c1, c2 = st.columns([2,1])
with c1:
    st.altair_chart(
        pareto_chart(src_tbl, "Deal Source", "Pareto – Enrolments by Deal Source"),
        use_container_width=True
    )
with c2:
    st.altair_chart(donut_referral_share(df_cohort, source_col), use_container_width=True)

st.altair_chart(
    pareto_chart(cty_tbl, "Country", "Pareto – Enrolments by Country"),
    use_container_width=True
)

# ----------------------------
# Tables + Downloads
# ----------------------------
st.markdown("<div class='section-title'>Tables</div>", unsafe_allow_html=True)
tabs = st.tabs(["Deal Source (80-20)", "Country (80-20)", "Cohort Rows (raw subset)"])

with tabs[0]:
    if src_tbl.empty:
        st.info("No enrollments in scope for the selected period/filters.")
    else:
        st.dataframe(src_tbl, use_container_width=True)
        st.download_button(
            "Download CSV – Deal Source Pareto",
            data=src_tbl.to_csv(index=False).encode("utf-8"),
            file_name="pareto_deal_source.csv",
            mime="text/csv"
        )

with tabs[1]:
    if cty_tbl.empty:
        st.info("No enrollments in scope for the selected period/filters.")
    else:
        st.dataframe(cty_tbl, use_container_width=True)
        st.download_button(
            "Download CSV – Country Pareto",
            data=cty_tbl.to_csv(index=False).encode("utf-8"),
            file_name="pareto_country.csv",
            mime="text/csv"
        )

with tabs[2]:
    show_cols = []
    if create_col: show_cols.append(create_col)
    show_cols.append(pay_col if pay_col else "_pay_dt")
    if source_col: show_cols.append(source_col)
    if country_col: show_cols.append(country_col)
    extra = [c for c in ["Student/Academic Counsellor","Academic Counsellor","Pipeline"] if c in df_cohort.columns]
    show_cols += [c for c in extra if c not in show_cols]
    preview = df_cohort[show_cols].copy() if show_cols else df_cohort.copy()
    st.dataframe(preview.head(1000), use_container_width=True)
    st.download_button(
        "Download CSV – Cohort subset",
        data=preview.to_csv(index=False).encode("utf-8"),
        file_name="cohort_subset.csv",
        mime="text/csv"
    )
