import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import date, timedelta
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
      .chip {
        display:inline-block; padding:4px 8px; border-radius:999px;
        background:#f3f4f6; color:#374151; font-size:.8rem; margin-top:.25rem;
      }
      .muted { color:#6b7280; font-size:.85rem; }
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
    "A_actual": "#2563eb",   # blue
    "Rem_prev": "#6b7280",   # gray
    "Rem_same": "#16a34a",   # green
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

def normalize_pipeline(value: str) -> str:
    if not isinstance(value, str):
        return "Other"
    v = value.strip().lower()
    if "math" in v:
        return "Math"
    if "ai" in v or "coding" in v or "ai-coding" in v or "ai coding" in v:
        return "AI Coding"
    return "Other"

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
# Load data
# ----------------------------
DEFAULT_DATA_PATH = "Master_sheet-DB.csv"
if "data_src" not in st.session_state:
    st.session_state["data_src"] = DEFAULT_DATA_PATH

data_src = st.session_state["data_src"]
df = load_data(data_src)

dealstage_col = find_col(df, ["Deal Stage","Deal stage","Stage","Deal Status","Stage Name","Deal Stage Name"])
df, _removed = exclude_invalid_deals(df, dealstage_col)

create_col = find_col(df, ["Create Date","Create date","Create_Date","Created At"])
pay_col    = find_col(df, ["Payment Received Date","Payment Received date","Payment_Received_Date","Payment Date","Paid At"])
pipeline_col = find_col(df, ["Pipeline"])
counsellor_col = find_col(df, ["Student/Academic Counsellor","Academic Counsellor","Counsellor"])
country_col    = find_col(df, ["Country"])
source_col     = find_col(df, ["JetLearn Deal Source","Deal Source","Source"])

first_cal_sched_col = find_col(df, ["First Calibration Scheduled Date"])
cal_resched_col     = find_col(df, ["Calibration Rescheduled Date"])
cal_done_col        = find_col(df, ["Calibration Done Date"])

today = date.today()

# ----------------------------
# Trend & Analysis (with new metric)
# ----------------------------
def ta_count_table(
    df_scope: pd.DataFrame,
    group_cols: list[str],
    mode: str,
    month_pick: date,
    cutoff_date: date,
    create_col: str,
    metric_cols: dict,
    metrics_selected: list[str],
) -> pd.DataFrame:
    if not group_cols:
        df_work = df_scope.copy()
        df_work["_GroupDummy"] = "All"
        group_cols = ["_GroupDummy"]
    else:
        df_work = df_scope.copy()

    m_start, m_end = month_bounds(month_pick)
    df_work["_create_dt"] = coerce_datetime(df_work[create_col]).dt.date

    if mode == "MTD":
        pop_mask = df_work["_create_dt"].between(m_start, m_end)
        pop = df_work.loc[pop_mask].copy()
    else:
        pop = df_work.copy()

    outs = []
    for disp in metrics_selected:
        col = metric_cols.get(disp)

        if disp == "Create Date (deals) — Count":
            if mode == "MTD":
                gdf = pop[group_cols].copy()
            else:
                mask = df_work["_create_dt"].between(m_start, m_end)
                gdf = df_work.loc[mask, group_cols].copy()
            agg = gdf.assign(_one=1).groupby(group_cols)["_one"].sum().reset_index().rename(columns={"_one": disp}) if not gdf.empty else pd.DataFrame(columns=group_cols+[disp])
            outs.append(agg)
            continue

        if disp == "Future Calibration Scheduled — Count":
            # Effective calibration date = rescheduled if exists, else first
            eff = coerce_datetime(df_work[cal_resched_col]).fillna(coerce_datetime(df_work[first_cal_sched_col]))
            mask_future = eff.dt.date > cutoff_date
            if mode == "MTD":
                gdf = pop.loc[mask_future.loc[pop.index], group_cols].copy()
            else:
                gdf = df_work.loc[mask_future, group_cols].copy()
            agg = gdf.assign(_one=1).groupby(group_cols)["_one"].sum().reset_index().rename(columns={"_one": disp}) if not gdf.empty else pd.DataFrame(columns=group_cols+[disp])
            outs.append(agg)
            continue

        if not col or col not in df_work.columns:
            agg = pop[group_cols].assign(**{disp:0}).groupby(group_cols)[disp].sum().reset_index() if not pop.empty else pd.DataFrame(columns=group_cols+[disp])
            outs.append(agg)
            continue

        ev = coerce_datetime(df_work[col]).dt.date
        if mode == "MTD":
            mask_nonnull = df_work[col].notna()
            gdf = pop.loc[mask_nonnull.loc[pop.index], group_cols].copy()
        else:
            mask_evt_month = ev.between(m_start, m_end)
            gdf = df_work.loc[mask_evt_month, group_cols].copy()
        agg = gdf.assign(_one=1).groupby(group_cols)["_one"].sum().reset_index().rename(columns={"_one": disp}) if not gdf.empty else pd.DataFrame(columns=group_cols+[disp])
        outs.append(agg)

    if outs:
        result = outs[0]
        for f in outs[1:]:
            result = result.merge(f, on=group_cols, how="outer")
    else:
        result = pd.DataFrame(columns=group_cols)

    for m in metrics_selected:
        if m not in result.columns:
            result[m] = 0
    result[metrics_selected] = result[metrics_selected].fillna(0).astype(int)
    return result.reset_index(drop=True)

# ----------------------------
# Trend & Analysis UI
# ----------------------------
st.subheader("Trend & Analysis – Grouped Drilldowns")

available_groups, group_map = [], {}
if counsellor_col: available_groups.append("Academic Counsellor"); group_map["Academic Counsellor"]=counsellor_col
if country_col:    available_groups.append("Country"); group_map["Country"]=country_col
if source_col:     available_groups.append("JetLearn Deal Source"); group_map["JetLearn Deal Source"]=source_col

sel_group_labels = st.multiselect("Group by", options=available_groups, default=available_groups[:1] if available_groups else [])
group_cols = [group_map[l] for l in sel_group_labels if l in group_map]

level = st.radio("Mode", ["MTD","Cohort"], index=0, horizontal=True)

date_mode = st.radio("Date scope", ["This month","Last month","Custom date range"], index=0, horizontal=True)
if date_mode=="This month":
    month_pick = today; cutoff_date = month_bounds(today)[1]
elif date_mode=="Last month":
    lm_start, lm_end = last_month_bounds(today); month_pick=lm_start; cutoff_date=lm_end
else:
    col_d1,col_d2 = st.columns(2)
    with col_d1: custom_start = st.date_input("Start date", value=today.replace(day=1))
    with col_d2: custom_end   = st.date_input("End date", value=month_bounds(today)[1])
    if custom_end<custom_start: st.error("End date cannot be before start date."); st.stop()
    if (custom_start.year,custom_start.month)!=(custom_end.year,custom_end.month):
        st.error("Pick a custom range within one month."); st.stop()
    month_pick=custom_start; cutoff_date=custom_end

all_metrics=[
    "Payment Received Date — Count",
    "First Calibration Scheduled Date — Count",
    "Calibration Rescheduled Date — Count",
    "Calibration Done Date — Count",
    "Create Date (deals) — Count",
    "Future Calibration Scheduled — Count"
]
metrics_selected = st.multiselect("Metrics", options=all_metrics, default=all_metrics)

metric_cols={
    "Payment Received Date — Count": pay_col,
    "First Calibration Scheduled Date — Count": first_cal_sched_col,
    "Calibration Rescheduled Date — Count": cal_resched_col,
    "Calibration Done Date — Count": cal_done_col,
    "Create Date (deals) — Count": create_col,
    "Future Calibration Scheduled — Count": None
}

tbl = ta_count_table(df, group_cols, level, month_pick, cutoff_date, create_col, metric_cols, metrics_selected)
if tbl.empty:
    st.info("No rows match filters.")
else:
    rename_map={group_map.get(lbl):lbl for lbl in sel_group_labels}
    st.dataframe(tbl.rename(columns=rename_map), use_container_width=True)
