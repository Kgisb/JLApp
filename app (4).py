# app_8020.py 
# JetLearn: 80-20 Pareto + Trajectory + Conversion% + Mix Analyzer
# NEW: Per-source country filters with All/None/Specific in the Interactive Mix Analyzer.
#      Each selected source gets its own country selector; downstream visuals use the OR-union.
# NEW (earlier): In Trajectory, pick a Deal Source basis (All / selected) to choose Top Countries.
# NEW (this update): Trajectory supports ALL JetLearn deal sources (raw) or the key 3 buckets via a toggle.

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import date
from calendar import monthrange
import re

st.set_page_config(page_title="JetLearn 80-20 + Trajectory + Conversion% + Mix", page_icon="📈", layout="wide")

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

# ---- Map raw deal source to 3 key sources ----
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

# Build a unified _src_pick column on any dataframe based on toggle
def assign_src_pick(df: pd.DataFrame, source_col: str | None, use_key: bool) -> pd.DataFrame:
    d = df.copy()
    if source_col and source_col in d.columns:
        if use_key:
            d["_src_pick"] = d[source_col].apply(normalize_key_source)
        else:
            d["_src_pick"] = d[source_col].fillna("Unknown").astype(str)
    else:
        d["_src_pick"] = "Other"
    return d

# ----------------------------
# Load data
# ----------------------------
st.title("📈 80-20 Pareto + Trajectory + Conversion% + Mix Analyzer")

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
# Sidebar – Cohort date scope + Conversion%
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

# Source filter (for the Pareto section only)
if source_col:
    all_sources = sorted(df_clean[source_col].dropna().astype(str).unique())
    excl_ref = st.sidebar.checkbox("Exclude Referral (for Pareto view)", value=False)
    sources_for_pick = [s for s in all_sources if not (excl_ref and "referr" in s.lower())]
    picked_sources = st.sidebar.multiselect("Include Deal Sources (Pareto)", options=sources_for_pick, default=sources_for_pick)
else:
    picked_sources = None

# Conversion mode toggle (kept for other visuals; KPI below ignores this toggle)
st.sidebar.header("Conversion%")
conv_mode = st.sidebar.radio(
    "Mode", ["MTD", "Cohort"], index=0, horizontal=True,
    help="This toggle affects other views where relevant. The KPI below always uses Enrolments (payments in range) / Deals Created (in range)."
)

# ----------------------------
# Apply cohort filter (payments within start_d..end_d) for charts that need it
# ----------------------------
scope_mask = df_clean["_pay_dt"].dt.date.between(start_d, end_d)
df_cohort = df_clean.loc[scope_mask].copy()
if picked_sources is not None and source_col:
    df_cohort = df_cohort[df_cohort[source_col].astype(str).isin(picked_sources)]

# ----------------------------
# Range KPI: Deals Created, Enrolments, Conversion% (range-based)
# ----------------------------
in_create_window = df_clean["_create_dt"].dt.date.between(start_d, end_d)
deals_created = int(in_create_window.sum())

in_pay_window = df_clean["_pay_dt"].dt.date.between(start_d, end_d)
enrolments = int(in_pay_window.sum())

conv_pct_simple = (enrolments / deals_created * 100.0) if deals_created > 0 else 0.0

st.markdown("<div class='section-title'>Range KPI — Deals Created vs Enrolments</div>", unsafe_allow_html=True)
cA, cB, cC = st.columns(3)
with cA:
    st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Deals Created</div><div class='kpi-value'>{deals_created:,}</div><div class='kpi-sub'>{start_d} → {end_d}</div></div>", unsafe_allow_html=True)
with cB:
    st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Enrolments (Payments)</div><div class='kpi-value'>{enrolments:,}</div><div class='kpi-sub'>{start_d} → {end_d}</div></div>", unsafe_allow_html=True)
with cC:
    st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Conversion% (Payments / Created)</div><div class='kpi-value'>{conv_pct_simple:.1f}%</div><div class='kpi-sub'>Num: {enrolments:,} • Den: {deals_created:,}</div></div>", unsafe_allow_html=True)

# ----------------------------
# Cohort KPIs
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
with k1: st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Cohort Enrolments</div><div class='kpi-value'>{total_enr:,}</div><div class='kpi-sub'>{start_d} → {end_d}</div></div>", unsafe_allow_html=True)
with k2: st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Referral % (cohort)</div><div class='kpi-value'>{ref_pct:.1f}%</div><div class='kpi-sub'>{ref_cnt:,} of {total_enr:,}</div></div>", unsafe_allow_html=True)
with k3: st.markdown(f"<div class='kpi-card'><div class='kpi-title'># Sources for 80%</div><div class='kpi-value'>{n_sources_80}</div></div>", unsafe_allow_html=True)
with k4: st.markdown(f"<div class='kpi-card'><div class='kpi-title'># Countries for 80%</div><div class='kpi-value'>{n_countries_80}</div></div>", unsafe_allow_html=True)

# ----------------------------
# 80-20 Charts
# ----------------------------
c1, c2 = st.columns([2,1])
with c1: st.altair_chart(pareto_chart(src_tbl, "Deal Source", "Pareto – Enrolments by Deal Source"), use_container_width=True)
with c2: st.altair_chart(donut_referral_share(df_cohort, source_col), use_container_width=True)
st.altair_chart(pareto_chart(cty_tbl, "Country", "Pareto – Enrolments by Country"), use_container_width=True)

# ----------------------------
# Conversion% by Key Source (bar)
# ----------------------------
def conversion_stats(df_all: pd.DataFrame, start_d: date, end_d: date):
    if create_col is None or pay_col is None:
        return pd.DataFrame(columns=["KeySource","Den","Num","Pct"])
    d = df_all.copy()
    d["_cdate"] = d["_create_dt"].dt.date
    d["_pdate"] = d["_pay_dt"].dt.date
    d["_key_source"] = d[source_col].apply(normalize_key_source) if source_col else "Other"

    denom_mask = d["_cdate"].between(start_d, end_d)  # deals created within range
    num_mask = d["_pdate"].between(start_d, end_d)    # payments within range

    rows = []
    for src in ["Referral", "PM - Search", "PM - Social"]:
        src_mask = (d["_key_source"] == src)
        den = int((denom_mask & src_mask).sum())
        num = int((num_mask & src_mask).sum())
        pct = (num/den*100.0) if den > 0 else 0.0
        rows.append({"KeySource": src, "Den": den, "Num": num, "Pct": pct})
    return pd.DataFrame(rows)

st.markdown("### Conversion% by Key Source (range-based)")
bysrc_conv = conversion_stats(df_clean, start_d, end_d)
if not bysrc_conv.empty:
    conv_chart = alt.Chart(bysrc_conv).mark_bar(opacity=0.9).encode(
        x=alt.X("KeySource:N", sort=["Referral","PM - Search","PM - Social"], title="Source"),
        y=alt.Y("Pct:Q", title="Conversion%"),
        tooltip=[alt.Tooltip("KeySource:N"), alt.Tooltip("Den:Q", title="Deals (Created)"),
                 alt.Tooltip("Num:Q", title="Enrolments (Payments)"), alt.Tooltip("Pct:Q", title="Conversion%", format=".1f")]
    ).properties(height=300, title=f"Conversion% (Payments / Created) • {start_d} → {end_d}")
    st.altair_chart(conv_chart, use_container_width=True)
else:
    st.info("No data to compute Conversion% by key source for this window.")

# ----------------------------
# Trajectory – Top Countries × (Key or Raw Deal Sources)
# ----------------------------
st.markdown("### Trajectory – Top Countries × Referral / PM - Search / PM - Social (or All Raw Sources)")

col_t1, col_t2, col_tg, col_t3 = st.columns([1, 1, 1.4, 1.6])
with col_t1:
    trailing_k = st.selectbox("Trailing window (months)", [3, 6, 12], index=0,
                              help="Monthly trajectory ends at your selected end date.")
with col_t2:
    top_k = st.selectbox("Top countries (by cohort enrolments)", [5, 7], index=0)
with col_tg:
    # NEW: Grouping for this Trajectory section only
    traj_grouping = st.radio(
        "Source grouping",
        ["Key (Referral/PM-Search/PM-Social/Other)", "Raw (all)"],
        index=0, horizontal=False,
        help="Choose how to display sources in Trajectory: 3 key buckets or all raw sources."
    )

# Build trailing cohort
months_list = months_back_list(end_d, trailing_k)
months_str = [str(p) for p in months_list]

df_trail = df_clean[df_clean["_pay_m"].isin(months_list)].copy()

# Assign trajectory source label based on grouping
if traj_grouping.startswith("Key"):
    df_trail["_traj_source"] = df_trail[source_col].apply(normalize_key_source) if source_col else "Other"
    # Options for the "Basis Source" dropdown (used to pick top countries)
    traj_source_options = ["All sources", "Referral", "PM - Search", "PM - Social", "Other"]
else:
    # Raw sources
    df_trail["_traj_source"] = df_trail[source_col].fillna("Unknown").astype(str) if source_col else "Other"
    # All unique raw sources present in trailing window
    unique_raw = sorted(df_trail["_traj_source"].dropna().unique().tolist())
    traj_source_options = ["All sources"] + unique_raw

with col_t3:
    traj_src_pick = st.selectbox(
        "Deal Source for Top Countries",
        options=traj_source_options,
        index=0,
        help="Top countries and charts are computed from payments of this source (or All) in the trailing window."
    )

# Filter trailing cohort by chosen basis for top-country selection
if traj_src_pick != "All sources":
    df_trail_src = df_trail[df_trail["_traj_source"] == traj_src_pick].copy()
else:
    df_trail_src = df_trail.copy()

# Top K countries across trailing window (based on chosen source cohort)
if country_col and not df_trail_src.empty:
    cty_counts = df_trail_src.groupby(country_col).size().sort_values(ascending=False)
    top_countries = cty_counts.head(top_k).index.astype(str).tolist()
else:
    top_countries = []

# Monthly total enrolments across ALL sources (denominator for "% of overall business")
monthly_total = df_trail.groupby("_pay_m").size().rename("TotalAll").reset_index()

# Build Month × Country × TrajSource counts for the chosen source cohort, restricted to top countries
if top_countries and source_col and country_col:
    mcs = (
        df_trail_src[df_trail_src[country_col].astype(str).isin(top_countries)]
        .groupby(["_pay_m", country_col, "_traj_source"]).size().rename("Cnt").reset_index()
    )
else:
    mcs = pd.DataFrame(columns=["_pay_m", country_col if country_col else "Country", "_traj_source", "Cnt"])

if not mcs.empty:
    mcs = mcs.merge(monthly_total, on="_pay_m", how="left")
    mcs["PctOfOverall"] = np.where(mcs["TotalAll"]>0, mcs["Cnt"]/mcs["TotalAll"]*100.0, 0.0)
    mcs["_pay_m_str"] = pd.Categorical(mcs["_pay_m"].astype(str), categories=months_str, ordered=True)

if not mcs.empty:
    # Sort color legend by frequency to keep legend stable & useful
    src_order = mcs["_traj_source"].value_counts().index.tolist()
    title_suffix = f"{traj_src_pick}" if traj_src_pick != "All sources" else "All sources"
    grouping_suffix = "Key" if traj_grouping.startswith("Key") else "Raw"

    facet_chart = alt.Chart(mcs).mark_bar(opacity=0.9).encode(
        x=alt.X("_pay_m_str:N", title="Month", sort=months_str),
        y=alt.Y("PctOfOverall:Q", title="% of overall business", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("_traj_source:N", title="Source", sort=src_order),
        tooltip=[
            alt.Tooltip("_pay_m_str:N", title="Month"),
            alt.Tooltip(f"{country_col}:N", title="Country") if country_col else alt.Tooltip("_pay_m_str:N"),
            alt.Tooltip("_traj_source:N", title="Source"),
            alt.Tooltip("Cnt:Q", title="Count"),
            alt.Tooltip("PctOfOverall:Q", title="% of overall", format=".1f")
        ]
    ).properties(height=220, title=f"Top Countries • Basis: {title_suffix} • Grouping: {grouping_suffix}").facet(
        column=alt.Column(f"{country_col}:N", title="Top Countries", sort=top_countries)
    )
    st.altair_chart(facet_chart, use_container_width=True)

    # Overall contribution lines (restricted to the chosen top countries set)
    overall = (
        mcs.groupby(["_pay_m_str","_traj_source"], as_index=False)
           .agg({"Cnt":"sum", "TotalAll":"first"})
    )
    overall["PctOfOverall"] = np.where(overall["TotalAll"]>0, overall["Cnt"]/overall["TotalAll"]*100.0, 0.0)
    lines = alt.Chart(overall).mark_line(point=True).encode(
        x=alt.X("_pay_m_str:N", title="Month", sort=months_str),
        y=alt.Y("PctOfOverall:Q", title="% of overall business (Top countries)", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("_traj_source:N", title="Source", sort=src_order),
        tooltip=[
            alt.Tooltip("_pay_m_str:N", title="Month"),
            alt.Tooltip("_traj_source:N", title="Source"),
            alt.Tooltip("PctOfOverall:Q", title="% of overall", format=".1f")
        ]
    ).properties(title=f"Overall contribution by source (Top countries • Basis: {title_suffix} • Grouping: {grouping_suffix})", height=320)
    st.altair_chart(lines, use_container_width=True)
else:
    st.info("No data for the selected trailing window to build the trajectory.")

# ============================================================
# Interactive Mix Analyzer — per-source country filters (All/None/Specific)
# ============================================================
st.markdown("### Interactive Mix Analyzer — % of overall business from your selection")

col_im1, col_im2 = st.columns([1.6, 1])
with col_im1:
    use_key_sources = st.checkbox(
        "Use key-source mapping (Referral / PM - Search / PM - Social)",
        value=True,
        help="On = group sources into 3 key buckets. Off = raw deal source names."
    )

# Cohort within window (payments inside window)
cohort_now = df_clean[df_clean["_pay_dt"].dt.date.between(start_d, end_d)].copy()
cohort_now = assign_src_pick(cohort_now, source_col, use_key_sources)

# Source option list
if source_col and source_col in cohort_now.columns:
    if use_key_sources:
        src_options = ["Referral", "PM - Search", "PM - Social", "Other"]
        default_srcs = ["Referral"]
    else:
        src_options = sorted(cohort_now["_src_pick"].unique().tolist())
        default_srcs = src_options[:1] if src_options else []
    picked_srcs = st.multiselect(
        "Select Deal Sources",
        options=src_options,
        default=[s for s in default_srcs if s in src_options],
        help="Pick one or more sources. Each source gets its own Country control below.",
        key="mix_sources_pick"
    )
else:
    picked_srcs = []
    st.info("Deal Source column not found, source filtering disabled for Mix Analyzer.")

# Session keys storers
def _mode_key(src): return f"src_mode::{src}"
def _countries_key(src): return f"src_countries::{src}"

# Build per-source country pickers
per_source_config = {}  # src -> dict(mode, countries, available)

for src in picked_srcs:
    # countries where this source actually has enrolments in window
    available = (
        cohort_now.loc[cohort_now["_src_pick"] == src, country_col]
        .astype(str).fillna("Unknown").value_counts().index.tolist()
        if country_col and country_col in cohort_now.columns else []
    )
    # init defaults in session_state
    if _mode_key(src) not in st.session_state:
        st.session_state[_mode_key(src)] = "All"
    if _countries_key(src) not in st.session_state:
        st.session_state[_countries_key(src)] = available.copy()

    # reconcile if options changed (e.g., date/source change)
    if st.session_state[_mode_key(src)] == "Specific":
        st.session_state[_countries_key(src)] = [c for c in st.session_state[_countries_key(src)] if c in available]
        if not st.session_state[_countries_key(src)] and available:
            st.session_state[_countries_key(src)] = available[:5]  # a small sensible default

    with st.container(border=True):
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown(f"**Source:** {src}")
            mode = st.radio(
                "Country scope",
                options=["All", "None", "Specific"],
                index=["All", "None", "Specific"].index(st.session_state[_mode_key(src)]),
                key=_mode_key(src),
                horizontal=True
            )
        with c2:
            if mode == "Specific":
                st.multiselect(
                    f"Countries for {src}",
                    options=available,
                    default=st.session_state[_countries_key(src)],
                    key=_countries_key(src)
                )
            elif mode == "All":
                st.caption(f"All countries for **{src}** ({len(available)}).")
            else:
                st.caption(f"Excluded **{src}** (no countries).")

    per_source_config[src] = {
        "mode": st.session_state[_mode_key(src)],
        "countries": st.session_state[_countries_key(src)],
        "available": available
    }

# Helper: build masks from per-source config
def make_union_mask(df: pd.DataFrame, per_cfg: dict, use_key: bool) -> pd.Series:
    d = assign_src_pick(df, source_col, use_key)
    base = pd.Series(False, index=d.index)
    if not per_cfg:
        return base  # nothing selected -> nothing passes
    if country_col and country_col in d.columns:
        c_series = d[country_col].astype(str).fillna("Unknown")
    else:
        c_series = pd.Series("Unknown", index=d.index)

    for src, info in per_cfg.items():
        mode = info["mode"]
        if mode == "None":
            continue
        src_mask = (d["_src_pick"] == src)
        if mode == "All":
            base = base | src_mask
        else:  # Specific
            chosen = set(info["countries"])
            if chosen:
                base = base | (src_mask & c_series.isin(chosen))
    return base

def active_sources(per_cfg: dict) -> list[str]:
    return [s for s, v in per_cfg.items() if v["mode"] != "None"]

# ---- View toggle
mix_view = st.radio(
    "Mix view",
    ["Aggregate (range total)", "Month-wise"],
    index=0,
    horizontal=True,
    help="Aggregate = single % for whole range. Month-wise = monthly % time series with one line per picked source."
)

# Compute % of overall (payments)
total_payments = int(len(cohort_now))
if total_payments == 0:
    st.warning("No payments (enrolments) in the selected window.")
else:
    sel_mask = make_union_mask(cohort_now, per_source_config, use_key_sources)
    if not sel_mask.any():
        st.info("No selection applied (pick at least one source in All/Specific).")
    else:
        selected_payments = int(sel_mask.sum())
        pct_of_overall = (selected_payments / total_payments * 100.0) if total_payments > 0 else 0.0

        st.markdown(
            f"<div class='kpi-card'>"
            f"<div class='kpi-title'>Contribution of your selection ({start_d} → {end_d})</div>"
            f"<div class='kpi-value'>{pct_of_overall:.1f}%</div>"
            f"<div class='kpi-sub'>Enrolments in selection: {selected_payments:,} • Total: {total_payments:,}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Quick breakdown by source (as % of overall)
        dsel = cohort_now.loc[sel_mask].copy()
        if not dsel.empty:
            bysrc = dsel.groupby("_src_pick").size().rename("SelCnt").reset_index()
            bysrc["PctOfOverall"] = bysrc["SelCnt"] / total_payments * 100.0
            chart = alt.Chart(bysrc).mark_bar(opacity=0.9).encode(
                x=alt.X("_src_pick:N", title="Source"),
                y=alt.Y("PctOfOverall:Q", title="% of overall business"),
                tooltip=[alt.Tooltip("_src_pick:N", title="Source"),
                         alt.Tooltip("SelCnt:Q", title="Enrolments (selected)"),
                         alt.Tooltip("PctOfOverall:Q", title="% of overall", format=".1f")],
                color=alt.Color("_src_pick:N", legend=alt.Legend(orient="bottom"))
            ).properties(height=320, title="Selection breakdown by source — % of overall")
            st.altair_chart(chart, use_container_width=True)

        # ---------------- Month-wise lines (optional view)
        if mix_view == "Month-wise":
            cohort_now["_pay_m"] = cohort_now["_pay_dt"].dt.to_period("M")
            months_in_range = (
                cohort_now["_pay_m"].dropna().sort_values().unique().astype(str).tolist()
            )

            # Overall monthly totals
            overall_m = cohort_now.groupby("_pay_m").size().rename("TotalAll").reset_index()
            overall_m["Month"] = overall_m["_pay_m"].astype(str)

            # All Selected monthly counts using union mask
            all_sel_m = cohort_now.loc[sel_mask].groupby("_pay_m").size().rename("SelCnt").reset_index()
            all_sel_m["Month"] = all_sel_m["_pay_m"].astype(str)

            all_line = overall_m.merge(all_sel_m[["_pay_m","SelCnt","Month"]], on=["_pay_m","Month"], how="left").fillna({"SelCnt":0})
            all_line["PctOfOverall"] = np.where(all_line["TotalAll"]>0, all_line["SelCnt"]/all_line["TotalAll"]*100.0, 0.0)
            all_line["Series"] = "All Selected"
            all_line = all_line[["Month","Series","SelCnt","TotalAll","PctOfOverall"]]
            all_line["Month"] = pd.Categorical(all_line["Month"], categories=months_in_range, ordered=True)

            # Per-source monthly lines honoring each source's country selection
            per_src_frames = []
            for src in active_sources(per_source_config):
                # build mask for only this source using its chosen mode/countries
                one_cfg = {src: per_source_config[src]}
                smask = make_union_mask(cohort_now, one_cfg, use_key_sources)
                s_cnt = cohort_now.loc[smask].groupby("_pay_m").size().rename("SelCnt").reset_index()
                if s_cnt.empty:
                    continue
                s_cnt["Month"] = s_cnt["_pay_m"].astype(str)
                s_join = overall_m.merge(s_cnt[["_pay_m","SelCnt","Month"]], on=["_pay_m","Month"], how="left").fillna({"SelCnt":0})
                s_join["PctOfOverall"] = np.where(s_join["TotalAll"]>0, s_join["SelCnt"]/s_join["TotalAll"]*100.0, 0.0)
                s_join["Series"] = src
                s_join = s_join[["Month","Series","SelCnt","TotalAll","PctOfOverall"]]
                s_join["Month"] = pd.Categorical(s_join["Month"], categories=months_in_range, ordered=True)
                per_src_frames.append(s_join)

            if per_src_frames:
                lines_df = pd.concat([all_line] + per_src_frames, ignore_index=True)
            else:
                lines_df = all_line.copy()

            avg_monthly_pct = lines_df.loc[lines_df["Series"]=="All Selected", "PctOfOverall"].mean() if not lines_df.empty else 0.0
            st.markdown(
                f"<div class='kpi-card'>"
                f"<div class='kpi-title'>Month-wise: average % contribution (All Selected)</div>"
                f"<div class='kpi-value'>{avg_monthly_pct:.1f}%</div>"
                f"<div class='kpi-sub'>Months: {lines_df['Month'].nunique() if not lines_df.empty else 0}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            stroke_width = alt.condition("datum.Series == 'All Selected'", alt.value(4), alt.value(2))
            chart = alt.Chart(lines_df).mark_line(point=True).encode(
                x=alt.X("Month:N", sort=months_in_range, title="Month"),
                y=alt.Y("PctOfOverall:Q", title="% of overall business", scale=alt.Scale(domain=[0, 100])),
                color=alt.Color("Series:N", title="Series"),
                strokeWidth=stroke_width,
                tooltip=[
                    alt.Tooltip("Month:N"),
                    alt.Tooltip("Series:N"),
                    alt.Tooltip("SelCnt:Q", title="Enrolments (selected)"),
                    alt.Tooltip("TotalAll:Q", title="Total enrolments"),
                    alt.Tooltip("PctOfOverall:Q", title="% of overall", format=".1f"),
                ]
            ).properties(height=360, title="Month-wise % of overall — All Selected vs each picked source")
            st.altair_chart(chart, use_container_width=True)

# =========================
# Deals vs Enrolments — for your current selection (grouped bars + optional Conversion line)
# =========================
st.markdown("### Deals vs Enrolments — for your current selection")

def _build_created_paid_monthly(df_all: pd.DataFrame,
                                start_d: date, end_d: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    df_all is already pre-filtered by (source × countries union).
    Returns:
      monthly_df: Month, CreatedCnt, PaidCnt, ConvPct
      agg_df: aggregate row with CreatedCnt, PaidCnt, ConvPct
    """
    d = df_all.copy()
    d["_cdate"] = d["_create_dt"].dt.date
    d["_pdate"] = d["_pay_dt"].dt.date
    d["_cmonth"] = d["_create_dt"].dt.to_period("M")
    d["_pmonth"] = d["_pay_dt"].dt.to_period("M")

    # window masks
    cwin = d["_cdate"].between(start_d, end_d)
    pwin = d["_pdate"].between(start_d, end_d)

    # calendar months in range
    month_index = pd.period_range(start=start_d.replace(day=1), end=end_d.replace(day=1), freq="M")

    created_m = (
        d.loc[cwin].groupby("_cmonth").size()
          .reindex(month_index, fill_value=0)
          .rename_axis(index="_month").reset_index(name="CreatedCnt")
    )
    paid_m = (
        d.loc[pwin].groupby("_pmonth").size()
          .reindex(month_index, fill_value=0)
          .rename_axis(index="_month").reset_index(name="PaidCnt")
    )

    monthly = created_m.merge(paid_m, on="_month", how="outer").fillna(0)
    monthly["Month"] = monthly["_month"].astype(str)
    monthly = monthly[["Month", "CreatedCnt", "PaidCnt"]]
    monthly["ConvPct"] = np.where(monthly["CreatedCnt"] > 0,
                                  monthly["PaidCnt"] / monthly["CreatedCnt"] * 100.0, 0.0)

    total_created = int(monthly["CreatedCnt"].sum())
    total_paid    = int(monthly["PaidCnt"].sum())
    agg = pd.DataFrame({
        "CreatedCnt": [total_created],
        "PaidCnt":    [total_paid],
        "ConvPct":    [float((total_paid / total_created * 100.0) if total_created > 0 else 0.0)]
    })
    return monthly, agg

# Apply the same union selection to df_clean for created & paid
if 'picked_srcs' in locals() and picked_srcs:
    union_mask_all = make_union_mask(df_clean, per_source_config, use_key_sources)
else:
    union_mask_all = pd.Series(False, index=df_clean.index)  # none selected -> empty

df_sel_all = df_clean.loc[union_mask_all].copy()

monthly_sel, agg_sel = _build_created_paid_monthly(df_sel_all, start_d, end_d)

# KPIs
kpa, kpb, kpc = st.columns(3)
with kpa:
    st.markdown(
        f"<div class='kpi-card'><div class='kpi-title'>Deals (Created)</div>"
        f"<div class='kpi-value'>{int(agg_sel['CreatedCnt'].iloc[0]):,}</div>"
        f"<div class='kpi-sub'>{start_d} → {end_d}</div></div>", unsafe_allow_html=True)
with kpb:
    st.markdown(
        f"<div class='kpi-card'><div class='kpi-title'>Enrolments (Payments)</div>"
        f"<div class='kpi-value'>{int(agg_sel['PaidCnt'].iloc[0]):,}</div>"
        f"<div class='kpi-sub'>{start_d} → {end_d}</div></div>", unsafe_allow_html=True)
with kpc:
    st.markdown(
        f"<div class='kpi-card'><div class='kpi-title'>Conversion% (Payments / Created)</div>"
        f"<div class='kpi-value'>{float(agg_sel['ConvPct'].iloc[0]):.1f}%</div>"
        f"<div class='kpi-sub'>Num: {int(agg_sel['PaidCnt'].iloc[0]):,} • Den: {int(agg_sel['CreatedCnt'].iloc[0]):,}</div></div>",
        unsafe_allow_html=True)

# ---- Month-wise chart: Grouped bars (Deals & Enrolments) + optional Conversion% line
show_conv_line = st.checkbox("Overlay Conversion% line on bars", value=True, key="mix_conv_line")

if not monthly_sel.empty:
    bar_df = monthly_sel.melt(
        id_vars=["Month"],
        value_vars=["CreatedCnt", "PaidCnt"],
        var_name="Metric",
        value_name="Count"
    )
    bar_df["Metric"] = bar_df["Metric"].map({"CreatedCnt": "Deals Created", "PaidCnt": "Enrolments"})

    bars = alt.Chart(bar_df).mark_bar(opacity=0.9).encode(
        x=alt.X("Month:N", sort=monthly_sel["Month"].tolist(), title="Month"),
        y=alt.Y("Count:Q", title="Count"),
        color=alt.Color("Metric:N", title=""),
        xOffset=alt.XOffset("Metric:N"),
        tooltip=[alt.Tooltip("Month:N"), alt.Tooltip("Metric:N"), alt.Tooltip("Count:Q")]
    ).properties(height=360, title="Month-wise — Deals & Enrolments (bars)")

    if show_conv_line:
        line = alt.Chart(monthly_sel).mark_line(point=True).encode(
            x=alt.X("Month:N", sort=monthly_sel["Month"].tolist(), title="Month"),
            y=alt.Y("ConvPct:Q", title="Conversion%", axis=alt.Axis(orient="right")),
            tooltip=[alt.Tooltip("Month:N"), alt.Tooltip("ConvPct:Q", title="Conversion%", format=".1f")],
            color=alt.value("#16a34a")
        )
        st.altair_chart(alt.layer(bars, line).resolve_scale(y='independent'), use_container_width=True)
    else:
        st.altair_chart(bars, use_container_width=True)

    with st.expander("Download: Month-wise Deals / Enrolments / Conversion% (selection)"):
        out_tbl = monthly_sel.rename(columns={
            "CreatedCnt": "Deals Created",
            "PaidCnt": "Enrolments",
            "ConvPct": "Conversion %"
        })
        st.dataframe(out_tbl, use_container_width=True)
        st.download_button(
            "Download CSV – Month-wise Deals/Enrolments/Conversion",
            data=out_tbl.to_csv(index=False).encode("utf-8"),
            file_name="selection_deals_enrolments_conversion_monthwise.csv",
            mime="text/csv"
        )
else:
    st.info("No month-wise data to plot for the current selection. Pick at least one source in All/Specific.")

# ----------------------------
# Tables + Downloads
# ----------------------------
st.markdown("<div class='section-title'>Tables</div>", unsafe_allow_html=True)
tabs = st.tabs(["Deal Source 80-20", "Country 80-20", "Cohort Rows", "Trajectory table", "Conversion by Source"])

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
    if 'mcs' in locals() and not mcs.empty:
        show = mcs.rename(columns={country_col: "Country"})[["Country","_pay_m_str","_traj_source","Cnt","TotalAll","PctOfOverall"]]
        show = show.sort_values(["Country","_pay_m_str","_traj_source"])
        st.dataframe(show, use_container_width=True)
        st.download_button("Download CSV – Trajectory", show.to_csv(index=False).encode("utf-8"),
                           "trajectory_top_countries_sources.csv", "text/csv")
    else:
        st.info("No trajectory table for the current selection.")

with tabs[4]:
    if not bysrc_conv.empty:
        st.dataframe(bysrc_conv, use_container_width=True)
        st.download_button("Download CSV – Conversion by Key Source",
                           bysrc_conv.to_csv(index=False).encode("utf-8"),
                           "conversion_by_key_source.csv", "text/csv")
    else:
        st.info("No conversion table for the current selection.")
