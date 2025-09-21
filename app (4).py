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
    "Total": "#6b7280",      # gray-500
    "AI Coding": "#2563eb",  # blue-600
    "Math": "#16a34a",       # green-600
    "ThresholdLow": "#f3f4f6",
    "ThresholdMid": "#e5e7eb",
    "ThresholdHigh": "#d1d5db",
    "A_actual_same": "#2563eb",   # reuse blue
    "B_fore_prev":   "#6b7280",   # gray
    "C_fore_same":   "#16a34a",   # green
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
    # Robust parsing; turns blanks/invalids into NaT
    s = pd.to_datetime(series, errors="coerce", infer_datetime_format=True, dayfirst=True)
    if s.notna().sum() == 0:
        # try epoch seconds / ms fallback if entire col failed
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

# ---------- MIS visuals ----------
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

def bullet_group(title: str, pcts: dict, nums: dict, denoms: dict):
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Total</div>"
                    f"<div class='kpi-value'>{pcts['Total']:.1f}%</div>"
                    f"<div class='kpi-sub'>Den: {denoms.get('Total',0):,} • Num: {nums.get('Total',0):,}</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>AI-Coding</div>"
                    f"<div class='kpi-value' style='color:{PALETTE['AI Coding']}'>{pcts['AI Coding']:.1f}%</div>"
                    f"<div class='kpi-sub'>Den: {denoms.get('AI Coding',0):,} • Num: {nums.get('AI Coding',0):,}</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Math</div>"
                    f"<div class='kpi-value' style='color:{PALETTE['Math']}'>{pcts['Math']:.1f}%</div>"
                    f"<div class='kpi-sub'>Den: {denoms.get('Math',0):,} • Num: {nums.get('Math',0):,}</div></div>", unsafe_allow_html=True)

    st.altair_chart(bullet_gauge(pcts["Total"], "Total", PALETTE["Total"], nums.get("Total",0), denoms.get("Total",0)), use_container_width=True)
    st.altair_chart(bullet_gauge(pcts["AI Coding"], "AI-Coding", PALETTE["AI Coding"], nums.get("AI Coding",0), denoms.get("AI Coding",0)), use_container_width=True)
    st.altair_chart(bullet_gauge(pcts["Math"], "Math", PALETTE["Math"], nums.get("Math",0), denoms.get("Math",0)), use_container_width=True)

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
# PREDICTIBILITY core
# ----------------------------
def add_month_cols(df: pd.DataFrame, create_col: str, pay_col: str) -> pd.DataFrame:
    d = df.copy()
    d["_create_dt"] = coerce_datetime(d[create_col])
    d["_pay_dt"]    = coerce_datetime(d[pay_col])
    d["_create_m"]  = d["_create_dt"].dt.to_period("M")
    d["_pay_m"]     = d["_pay_dt"].dt.to_period("M")
    d["_same_month"] = (d["_create_m"] == d["_pay_m"])
    return d

def source_wavg_shares(d_hist: pd.DataFrame, source_col: str, lookback: int, weighted: bool):
    """
    Per-source weighted-average shares for the pay-month composed of:
      - same_share: proportion where create-month == pay-month
      - prev_share: 1 - same_share
    Computed over the last `lookback` months BEFORE the current month.
    """
    if d_hist.empty:
        # Fallback: 50/50 overall if no history
        return {}, {}, 0.5, 0.5

    # Determine which months to keep (last `lookback` months found in d_hist)
    months = sorted(d_hist["_pay_m"].unique())
    months = months[-lookback:] if len(months) > lookback else months

    # Build recency weights (older=1 ... newest=k) or simple average
    weights = {m: (i+1 if weighted else 1.0) for i, m in enumerate(months)}

    # Compute per-month, per-source same-share
    shares_per_src = {}
    for m in months:
        sub = d_hist[d_hist["_pay_m"] == m]
        if sub.empty:
            continue
        gsize = sub.groupby(source_col).size().rename("total")
        gsame = sub[sub["_same_month"]].groupby(source_col).size().rename("same_cnt")
        by_src = pd.concat([gsize, gsame], axis=1).fillna(0.0).reset_index()
        by_src["same_share_m"] = by_src.apply(
            lambda r: (r["same_cnt"] / r["total"]) if r["total"] > 0 else 0.0, axis=1
        )
        for _, row in by_src.iterrows():
            src = str(row[source_col])
            shares_per_src.setdefault(src, []).append((m, row["same_share_m"]))

    # Weighted avg per source
    same_share_src = {}
    for src, lst in shares_per_src.items():
        num = sum(weights[m] * s for m, s in lst)
        den = sum(weights[m] for m, _ in lst)
        same_share_src[src] = (num/den) if den > 0 else 0.0

    # Overall fallback (across sources) if a source missing
    overall_by_m = {}
    for m in months:
        sub = d_hist[d_hist["_pay_m"] == m]
        tot = len(sub)
        same = int(sub["_same_month"].sum())
        overall_by_m[m] = (same / tot) if tot > 0 else 0.0
    num = sum(overall_by_m[m] * weights[m] for m in months)
    den = sum(weights[m] for m in months)
    overall_same = (num/den) if den > 0 else 0.5

    overall_prev = 1.0 - overall_same
    prev_share_src = {k: 1.0 - v for k, v in same_share_src.items()}
    return same_share_src, prev_share_src, overall_same, overall_prev

def predict_running_month(df_f: pd.DataFrame, create_col: str, pay_col: str, source_col: str,
                          lookback: int, weighted: bool, today: date):
    """
    Returns (table_df, totals_dict) with columns:
      Source, A_Actual_Same, B_Remaining_From_Prev, C_Remaining_From_Same,
      Projected_MonthEnd_Total, WAvg_Same_Share, WAvg_Prev_Share
    """
    # Ensure we have a source column
    if source_col is None or source_col not in df_f.columns:
        df_work = df_f.copy()
        source_col = "_Source"
        df_work[source_col] = "All"
    else:
        df_work = df_f.copy()

    d = add_month_cols(df_work, create_col, pay_col)

    # Current month as a Period for exact month equality (robust vs. timestamps)
    cur_start, cur_end = month_bounds(today)
    cur_period = pd.Period(today, freq="M")

    # Payments realized in current month (month equality, NOT date-range)
    in_cur_pay = d["_pay_m"] == cur_period
    d_cur = d.loc[in_cur_pay].copy()

    # Realized splits to date
    if d_cur.empty:
        realized_by_src = pd.DataFrame(columns=[source_col, "realized_total", "realized_same", "realized_prev"])
    else:
        realized_by_src = d_cur.groupby(source_col).apply(
            lambda x: pd.Series({
                "realized_total": len(x),
                "realized_same": int(x["_same_month"].sum()),
                "realized_prev": int((~x["_same_month"]).sum()),
            })
        ).reset_index()

    # Historical dataset (exclude current month)
    d_hist = d[d["_pay_m"] < cur_period].copy()
    if not d_hist.empty:
        # keep only the last `lookback` months actually present
        months_present = sorted(d_hist["_pay_m"].unique())
        months_keep = months_present[-lookback:] if len(months_present) > lookback else months_present
        d_hist = d_hist[d_hist["_pay_m"].isin(months_keep)]

    same_shares, prev_shares, overall_same, overall_prev = source_wavg_shares(
        d_hist, source_col, lookback, weighted
    )

    # Pace to month-end
    elapsed_days = (today - cur_start).days + 1
    total_days = (cur_end - cur_start).days + 1
    by_src = realized_by_src.set_index(source_col) if not realized_by_src.empty else pd.DataFrame().set_index(source_col)

    # All sources to consider (realized this month OR seen historically)
    sources_realized = set(d_cur[source_col].dropna().astype(str)) if not d_cur.empty else set()
    sources_hist = set(same_shares.keys()) | set(prev_shares.keys())
    all_sources = sorted(sources_realized | sources_hist | ({"All"} if source_col == "_Source" else set()))

    rows, tot_A, tot_B, tot_C = [], 0.0, 0.0, 0.0
    for src in all_sources:
        if not by_src.empty and src in by_src.index:
            r = by_src.loc[src]
            realized_total = int(r.get("realized_total", 0))
            realized_same  = int(r.get("realized_same", 0))
            realized_prev  = int(r.get("realized_prev", 0))
        else:
            realized_total = realized_same = realized_prev = 0

        # Month-end projection via simple MTD pace
        per_day = (realized_total / elapsed_days) if elapsed_days > 0 else 0.0
        projected_total = per_day * total_days

        # Shares: per-source, else overall fallback
        same_share = same_shares.get(src, overall_same)
        prev_share = prev_shares.get(src, overall_prev)

        exp_same_total = projected_total * same_share
        exp_prev_total = projected_total * prev_share

        rem_from_same = max(0.0, exp_same_total - realized_same)
        rem_from_prev = max(0.0, exp_prev_total - realized_prev)

        rows.append({
            "Source": src,
            "A_Actual_Same": float(realized_same),
            "B_Remaining_From_Prev": float(rem_from_prev),
            "C_Remaining_From_Same": float(rem_from_same),
            "Projected_MonthEnd_Total": float(projected_total),
            "WAvg_Same_Share": float(same_share),
            "WAvg_Prev_Share": float(prev_share),
        })

        tot_A += realized_same
        tot_B += rem_from_prev
        tot_C += rem_from_same

    tbl = pd.DataFrame(rows).sort_values("Source").reset_index(drop=True)
    totals = {
        "A_Actual_Same": tot_A,
        "B_Remaining_From_Prev": tot_B,
        "C_Remaining_From_Same": tot_C,
        "Projected_MonthEnd_Total": float(tbl["Projected_MonthEnd_Total"].sum()) if not tbl.empty else 0.0
    }
    return tbl, totals

def predict_chart(tbl: pd.DataFrame):
    if tbl.empty:
        return alt.Chart(pd.DataFrame({"x":[],"y":[]}))
    melt = tbl.melt(
        id_vars=["Source"],
        value_vars=["A_Actual_Same","B_Remaining_From_Prev","C_Remaining_From_Same"],
        var_name="Component",
        value_name="Value"
    )
    color_map = {
        "A_Actual_Same": PALETTE["A_actual_same"],
        "B_Remaining_From_Prev": PALETTE["B_fore_prev"],
        "C_Remaining_From_Same": PALETTE["C_fore_same"],
    }
    chart = alt.Chart(melt).mark_bar().encode(
        x=alt.X("Source:N", sort=alt.SortField("Source")),
        y=alt.Y("Value:Q"),
        color=alt.Color("Component:N",
                        scale=alt.Scale(domain=list(color_map.keys()), range=list(color_map.values())),
                        legend=alt.Legend(title="Component",
                                          orient="top",
                                          labelLimit=240)),
        tooltip=[alt.Tooltip("Source:N"),
                 alt.Tooltip("Component:N"),
                 alt.Tooltip("Value:Q", format=",.1f")]
    ).properties(height=340, title="Predictibility: A (Actual Same) + B (Remaining from Prev) + C (Remaining from Same)")
    return chart

# ----------------------------
# UI
# ----------------------------
with st.sidebar:
    st.header("JetLearn • Navigation")
    view = st.radio("Go to", ["MIS", "Predictibility"], index=0)
    st.caption("Use the MIS quick periods, filters, or the Predictibility tab for month-end forecast.")

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
st.write("Visualizes **Enrolments (Payments)**, **Conversion%** (per-pipeline denominators), **Trend**, and **Predictibility** for the running month.")

# --- Data source (no checkbox; exclusion is ALWAYS ON) ---
col_ds1, col_ds2 = st.columns([3, 2])
with col_ds1:
    default_path = "Master_sheet_DB.csv"
    data_src = st.text_input("Data file path", value=default_path, help="CSV path (pre-uploaded in the repo).")
with col_ds2:
    st.caption("‘1.2 Invalid Deal(s)’ are automatically excluded. Rows with missing **Create Date** are dropped.")

# --- Load data
df = load_data(data_src)

# Identify Deal Stage column and exclude invalids (always)
dealstage_col = find_col(df, ["Deal Stage", "Deal stage", "Stage", "Deal Status", "Stage Name", "Deal Stage Name"])
df, _removed = exclude_invalid_deals(df, dealstage_col)
if dealstage_col:
    st.caption(f"Excluded “1.2 Invalid Deal(s)”: **{_removed:,}** rows (column: **{dealstage_col}**).")
else:
    st.info("Deal Stage column not found — cannot exclude “1.2 Invalid Deal(s)”. Make sure the column exists.")

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

# --- Drop rows with missing/invalid Create Date (after coercion) ---
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

# ----------------------------
# MIS
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

    # Conversion% (per-pipeline denominators)
    mtd_pct, coh_pct, denoms, nums = prepare_conversion_for_range(
        df_f, range_start, range_end, create_col, pay_col, pipeline_col,
        denom_mode="anchor", running_month_anchor=running_month_anchor
    )
    st.caption(
        f"Denominators — Total: {denoms['Total']:,} • AI-Coding: {denoms['AI Coding']:,} • Math: {denoms['Math']:,}"
    )
    bullet_group("MTD Conversion %", mtd_pct, nums["mtd"], denoms)
    bullet_group("Cohort Conversion %", coh_pct, nums["cohort"], denoms)

    # Trend (combined)
    ts = trend_timeseries(
        df_f, range_start, range_end,
        denom_mode="anchor", running_month_anchor=running_month_anchor,
        create_col=create_col, pay_col=pay_col
    )
    st.altair_chart(trend_chart(ts, "Trend: Leads (bars) vs Enrolments (lines)"), use_container_width=True)

if view == "MIS":
    show_all = st.checkbox("Show all preset periods (Yesterday • Today • Last Month • This Month)", value=False)
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

                    # Conversion → bullet gauges
                    mtd_pct, coh_pct, denoms, nums = prepare_conversion_for_range(
                        df_f, custom_start, custom_end, create_col, pay_col, pipeline_col,
                        denom_mode="anchor", running_month_anchor=anchor
                    )
                    st.caption(
                        f"Denominators — Total: {denoms['Total']:,} • AI-Coding: {denoms['AI Coding']:,} • Math: {denoms['Math']:,}"
                    )
                    bullet_group("MTD Conversion %", mtd_pct, nums["mtd"], denoms)
                    bullet_group("Cohort Conversion %", coh_pct, nums["cohort"], denoms)

                    # Trend
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
                        # Counts anchored to custom_start month for MTD counts display
                        anchor_for_counts = custom_start
                        mtd_counts, coh_counts = prepare_counts_for_range(df_f, custom_start, custom_end, anchor_for_counts, create_col, pay_col, pipeline_col)
                        c1, c2 = st.columns(2)
                        with c1:
                            st.altair_chart(bubble_chart_counts("MTD Enrolments (counts)", mtd_counts["Total"], mtd_counts["AI Coding"], mtd_counts["Math"]), use_container_width=True)
                        with c2:
                            st.altair_chart(bubble_chart_counts("Cohort Enrolments (counts)", coh_counts["Total"], coh_counts["AI Coding"], coh_counts["Math"]), use_container_width=True)

                        # Conversion → bullet gauges (custom range denoms)
                        mtd_pct, coh_pct, denoms, nums = prepare_conversion_for_range(
                            df_f, custom_start, custom_end, create_col, pay_col, pipeline_col,
                            denom_mode="range", denom_start=denom_start, denom_end=denom_end
                        )
                        st.caption(
                            f"Denominators — Total: {denoms['Total']:,} • AI-Coding: {denoms['AI Coding']:,} • Math: {denoms['Math']:,}"
                        )
                        bullet_group("MTD Conversion %", mtd_pct, nums["mtd"], denoms)
                        bullet_group("Cohort Conversion %", coh_pct, nums["cohort"], denoms)

                        # Trend
                        ts = trend_timeseries(
                            df_f, custom_start, custom_end,
                            denom_mode="range", denom_start=denom_start, denom_end=denom_end,
                            create_col=create_col, pay_col=pay_col
                        )
                        st.altair_chart(trend_chart(ts, "Trend: Leads (bars) vs Enrolments (lines)"), use_container_width=True)

# ----------------------------
# Predictibility
# ----------------------------
if view == "Predictibility":
    st.subheader("Predictibility – Running Month Enrolment Forecast")
    st.caption(
        "A = actual MTD enrolments from **same-month deals**. "
        "B = forecast for remaining days from **previous-months’ deals** using source-wise lookback shares. "
        "C = forecast for remaining days from **same-month deals**. "
        "Month-end total per source is paced from MTD."
    )

    colp1, colp2, colp3 = st.columns([1,1,2])
    with colp1:
        lookback = st.selectbox("Lookback window (months)", [3, 6, 12], index=0)
    with colp2:
        weighting = st.radio("Share averaging", ["Recency-weighted", "Simple average"], index=0, horizontal=False)
        weighted = (weighting == "Recency-weighted")
    with colp3:
        st.info("Recency-weighted uses linear weights (1..k) with higher weight on more recent months within the selected lookback.")

    # Sanity count for current month (after filters)
    cur_start, cur_end = month_bounds(today)
    d_preview = add_month_cols(df_f, create_col, pay_col)
    cur_period = pd.Period(today, freq="M")
    in_cur_pay = d_preview["_pay_m"] == cur_period
    st.caption(f"Payments found this month (after filters): **{int(in_cur_pay.sum()):,}**")

    tbl, totals = predict_running_month(df_f, create_col, pay_col, source_col, lookback, weighted, today)

    c1, c2 = st.columns([3,2])
    with c1:
        st.altair_chart(predict_chart(tbl), use_container_width=True)
    with c2:
        st.markdown("### Totals")
        st.markdown(
            f"""
            <div class='kpi-card'>
              <div class='kpi-title'>A · Actual from same-month deals (to date)</div>
              <div class='kpi-value' style='color:{PALETTE["A_actual_same"]}'>{totals['A_Actual_Same']:.1f}</div>
              <div class='kpi-sub'>So far this month</div>
            </div>
            <br/>
            <div class='kpi-card'>
              <div class='kpi-title'>B · Remaining from previous-months deals (forecast)</div>
              <div class='kpi-value' style='color:{PALETTE["B_fore_prev"]}'>{totals['B_Remaining_From_Prev']:.1f}</div>
            </div>
            <br/>
            <div class='kpi-card'>
              <div class='kpi-title'>C · Remaining from same-month deals (forecast)</div>
              <div class='kpi-value' style='color:{PALETTE["C_fore_same"]}'>{totals['C_Remaining_From_Same']:.1f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with st.expander("Detailed table"):
        show_cols = ["Source","A_Actual_Same","B_Remaining_From_Prev","C_Remaining_From_Same",
                     "Projected_MonthEnd_Total","WAvg_Same_Share","WAvg_Prev_Share"]
        if not tbl.empty:
            view_tbl = tbl[show_cols].copy()
            view_tbl["Projected_MonthEnd_Total"] = view_tbl["Projected_MonthEnd_Total"].round(1)
            view_tbl["WAvg_Same_Share"] = (view_tbl["WAvg_Same_Share"]*100).round(1)
            view_tbl["WAvg_Prev_Share"] = (view_tbl["WAvg_Prev_Share"]*100).round(1)
            view_tbl = view_tbl.rename(columns={
                "Projected_MonthEnd_Total":"Projected Total (ME)",
                "WAvg_Same_Share":"Same% (w-avg)",
                "WAvg_Prev_Share":"Prev% (w-avg)"
            })
            st.dataframe(view_tbl, use_container_width=True)
        else:
            st.info("No data in scope for the running month after filters.")

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
