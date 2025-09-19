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
      .pill-other { background: #fde68a; } /* amber-200 */
      .section-title {
        font-weight: 700;
        font-size: 1.05rem;
        margin-top: .25rem;
        margin-bottom: .25rem;
      }
      .hint {
        color: #6b7280;
        font-size: 0.9rem;
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
    "Other": "#d97706",      # amber-700
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

# ---------- COUNT LOGIC ----------
def split_counts(series_pipeline: pd.Series):
    sp = series_pipeline.map(normalize_pipeline).fillna("Other")
    ai = int((sp == "AI Coding").sum())
    math = int((sp == "Math").sum())
    other = int((sp == "Other").sum())
    return ai, math, other

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
        ai_c, math_c, other_c = split_counts(cohort_df[pipeline_col])
        ai_m, math_m, other_m = split_counts(mtd_df[pipeline_col])
    else:
        ai_c = math_c = other_c = ai_m = math_m = other_m = 0

    cohort_counts = {
        "Total": int(len(cohort_df)),
        "AI Coding": ai_c, "Math": math_c, "Other": other_c,
    }
    mtd_counts = {
        "Total": int(len(mtd_df)),
        "AI Coding": ai_m, "Math": math_m, "Other": other_m,
    }
    return mtd_counts, cohort_counts

# ---------- CONVERSION% LOGIC ----------
def deals_created_in_anchor_month(df: pd.DataFrame, running_month_any_date: date, create_col: str) -> int:
    df = df.copy()
    df["_create_dt"] = coerce_datetime(df[create_col])
    m_start, m_end = month_bounds(running_month_any_date)
    return int(df["_create_dt"].dt.date.between(m_start, m_end).sum())

def deals_created_in_range(df: pd.DataFrame, denom_start: date, denom_end: date, create_col: str) -> int:
    df = df.copy()
    df["_create_dt"] = coerce_datetime(df[create_col])
    return int(df["_create_dt"].dt.date.between(denom_start, denom_end).sum())

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
    """
    Returns (mtd_pct, coh_pct, denom, numerators_dict)
      - numerators_dict: {"mtd": {"Total": n, "AI Coding": n, "Math": n, "Other": n},
                          "cohort": {...}}
    """
    df = df.copy()
    df["_create_dt"] = coerce_datetime(df[create_col])
    df["_pay_dt"] = coerce_datetime(df[pay_col])

    # Denominator (shared for Total/AI/Math/Other)
    if denom_mode == "range":
        if denom_start is None or denom_end is None:
            return {"Total":0.0,"AI Coding":0.0,"Math":0.0}, {"Total":0.0,"AI Coding":0.0,"Math":0.0}, 0, {"mtd":{}, "cohort":{}}
        denom = deals_created_in_range(df, denom_start, denom_end, create_col)
        in_mtd_create = df["_create_dt"].dt.date.between(denom_start, denom_end)
    else:
        if running_month_anchor is None:
            return {"Total":0.0,"AI Coding":0.0,"Math":0.0}, {"Total":0.0,"AI Coding":0.0,"Math":0.0}, 0, {"mtd":{}, "cohort":{}}
        denom = deals_created_in_anchor_month(df, running_month_anchor, create_col)
        m_start, m_end = month_bounds(running_month_anchor)
        in_mtd_create = df["_create_dt"].dt.date.between(m_start, m_end)

    # Numerators
    in_range_pay = df["_pay_dt"].dt.date.between(start_d, end_d)
    mtd_df = df.loc[in_range_pay & in_mtd_create]
    cohort_df = df.loc[in_range_pay]

    if pipeline_col and pipeline_col in df.columns:
        ai_m, math_m, other_m = split_counts(mtd_df[pipeline_col])
        ai_c, math_c, other_c = split_counts(cohort_df[pipeline_col])
    else:
        ai_m = math_m = other_m = ai_c = math_c = other_c = 0

    mtd_total = int(len(mtd_df))
    coh_total = int(len(cohort_df))

    # One-decimal percentages; shared denominator
    def pct(v): 
        return 0.0 if denom == 0 else round(100.0 * v / denom, 1)

    mtd_pct = {"Total": pct(mtd_total), "AI Coding": pct(ai_m), "Math": pct(math_m)}
    coh_pct = {"Total": pct(coh_total), "AI Coding": pct(ai_c), "Math": pct(math_c)}

    numerators = {
        "mtd": {"Total": mtd_total, "AI Coding": ai_m, "Math": math_m, "Other": other_m},
        "cohort": {"Total": coh_total, "AI Coding": ai_c, "Math": math_c, "Other": other_c},
    }
    return mtd_pct, coh_pct, denom, numerators

# ---------- CHARTS ----------
def bubble_chart_counts(title: str, total: int, ai_cnt: int, math_cnt: int, other_cnt: int):
    data = pd.DataFrame({
        "Label": ["Total", "AI Coding", "Math", "Other"],
        "Value": [total, ai_cnt, math_cnt, other_cnt],
        "Row": [0, 1, 1, 1],
        "Col": [0.5, 0.26, 0.50, 0.74],
    })
    color_domain = ["Total", "AI Coding", "Math", "Other"]
    color_range  = [PALETTE["Total"], PALETTE["AI Coding"], PALETTE["Math"], PALETTE["Other"]]
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

def bubble_chart_pct(title: str, total_pct: float, ai_pct: float, math_pct: float, denom: int, num_dict: dict, cohort_or_mtd: str):
    # num_dict: {"Total": n, "AI Coding": n, "Math": n, "Other": n}
    data = pd.DataFrame({
        "Label": ["Total", "AI Coding", "Math"],
        "Pct": [total_pct, ai_pct, math_pct],
        "PctLabel": [f"{total_pct:.1f}%", f"{ai_pct:.1f}%", f"{math_pct:.1f}%"],
        "Num": [num_dict["Total"], num_dict["AI Coding"], num_dict["Math"]],
        "Den": [denom, denom, denom],
        "Row": [0, 1, 1],
        "Col": [0.5, 0.33, 0.66],
    })
    color_domain = ["Total", "AI Coding", "Math"]
    color_range  = [PALETTE["Total"], PALETTE["AI Coding"], PALETTE["Math"]]
    base = alt.Chart(data).encode(
        x=alt.X("Col:Q", axis=None, scale=alt.Scale(domain=(0, 1))),
        y=alt.Y("Row:Q", axis=None, scale=alt.Scale(domain=(-0.2, 1.2))),
        tooltip=[
            alt.Tooltip("Label:N"),
            alt.Tooltip("Num:Q", title="Numerator"),
            alt.Tooltip("Den:Q", title="Denominator"),
            alt.Tooltip("Pct:Q", title="Conversion %", format=".1f"),
        ],
    )
    circles = base.mark_circle(opacity=0.9).encode(
        size=alt.Size("Pct:Q", scale=alt.Scale(domain=[0, 100], range=[1200, 14000]), legend=None),
        color=alt.Color("Label:N", scale=alt.Scale(domain=color_domain, range=color_range), legend=None),
    )
    text = base.mark_text(fontWeight="bold", dy=0, color="#111827").encode(text="PctLabel:N")
    chart = (circles + text).properties(height=360, title=title)
    caption = st.caption("Note: Percentages are **not additive**; each bubble uses the same denominator.")
    return chart

def mix_share_chart(title: str, ai: int, math_: int, other: int):
    total = max(ai + math_ + other, 1)
    df = pd.DataFrame({
        "Segment": ["AI Coding", "Math", "Other"],
        "Share": [100*ai/total, 100*math_/total, 100*other/total],
    })
    color_domain = ["AI Coding", "Math", "Other"]
    color_range  = [PALETTE["AI Coding"], PALETTE["Math"], PALETTE["Other"]]
    return alt.Chart(df).mark_bar().encode(
        x=alt.X("Segment:N", axis=alt.Axis(title=None)),
        y=alt.Y("Share:Q", axis=alt.Axis(title="Pipeline mix (% of enrolments)", format=".0f")),
        color=alt.Color("Segment:N", scale=alt.Scale(domain=color_domain, range=color_range), legend=None),
        tooltip=[alt.Tooltip("Segment:N"), alt.Tooltip("Share:Q", format=".1f")]
    ).properties(height=220, title=title)

# ----------------------------
# UI
# ----------------------------
with st.sidebar:
    st.header("JetLearn • Navigation")
    view = st.radio("Go to", ["MIS"], index=0)
    st.caption("Use the quick periods, filters, or the Custom tab.")

st.title("📊 JetLearn MIS")
st.markdown(
    """
    <div>
      <span class="legend-pill pill-ai">AI-Coding</span>
      <span class="legend-pill pill-math">Math</span>
      <span class="legend-pill pill-total">Total (Both)</span>
      <span class="legend-pill pill-other">Other</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write(
    "Shows **Enrolments (Payments)** and **Conversion%**. Conversion% uses a **single shared denominator** "
    "for Total/AI/Math (not per-pipeline), so the percentages are **not additive**."
)

# --- Load data
default_path = "Master_sheet_DB.csv"
data_src = st.text_input("Data file path", value=default_path, help="CSV path (pre-uploaded in the repo).")
df = load_data(data_src)

# --- Resolve columns
create_col = find_col(df, ["Create Date", "Create date", "Create_Date", "Created At"])
pay_col = find_col(df, ["Payment Received Date", "Payment Received date", "Payment_Received_Date", "Payment Date", "Paid At"])
pipeline_col = find_col(df, ["Pipeline"])

# Filters
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
def apply_filters(df, counsellor_col, country_col, source_col, sel_counsellors, sel_countries, sel_sources):
    f = df.copy()
    if counsellor_col and sel_counsellors and "All" not in sel_counsellors:
        f = f[f[counsellor_col].astype(str).isin(sel_counsellors)]
    if country_col and sel_countries and "All" not in sel_countries:
        f = f[f[country_col].astype(str).isin(sel_countries)]
    if source_col and sel_sources and "All" not in sel_sources:
        f = f[f[source_col].astype(str).isin(sel_sources)]
    return f

df_f = apply_filters(df, counsellor_col, country_col, source_col, sel_counsellors, sel_countries, sel_sources)
st.caption(f"Rows in scope after filters: **{len(df_f):,}**")

# Show-all toggle + mix toggle
col_toggle1, col_toggle2 = st.columns([2,2])
with col_toggle1:
    show_all = st.checkbox("Show all preset periods (Yesterday • Today • Last Month • This Month)", value=False)
with col_toggle2:
    show_mix = st.checkbox("Show pipeline mix (% of enrolments) under each period", value=False)

# ----------------------------
# Period sections (Counts + Conversion%)
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
            bubble_chart_counts("MTD Enrolments (counts)", mtd_counts["Total"], mtd_counts["AI Coding"], mtd_counts["Math"], mtd_counts["Other"]),
            use_container_width=True
        )
    with c2:
        st.altair_chart(
            bubble_chart_counts("Cohort Enrolments (counts)", coh_counts["Total"], coh_counts["AI Coding"], coh_counts["Math"], coh_counts["Other"]),
            use_container_width=True
        )

    # Conversion % (shared denominator)
    mtd_pct, coh_pct, denom, nums = prepare_conversion_for_range(
        df_f, range_start, range_end, create_col, pay_col, pipeline_col,
        denom_mode="anchor", running_month_anchor=running_month_anchor
    )
    st.caption(f"Conversion% denominator (deals created in running month): **{denom:,}**")
    c3, c4 = st.columns(2)
    with c3:
        st.altair_chart(
            bubble_chart_pct("MTD Conversion %", mtd_pct["Total"], mtd_pct["AI Coding"], mtd_pct["Math"], denom, nums["mtd"], "mtd"),
            use_container_width=True
        )
    with c4:
        st.altair_chart(
            bubble_chart_pct("Cohort Conversion %", coh_pct["Total"], coh_pct["AI Coding"], coh_pct["Math"], denom, nums["cohort"], "cohort"),
            use_container_width=True
        )

    if show_mix:
        m1, m2 = st.columns(2)
        with m1:
            st.altair_chart(
                mix_share_chart("MTD pipeline mix (% of enrolments)", mtd_counts["AI Coding"], mtd_counts["Math"], mtd_counts["Other"]),
                use_container_width=True
            )
        with m2:
            st.altair_chart(
                mix_share_chart("Cohort pipeline mix (% of enrolments)", coh_counts["AI Coding"], coh_counts["Math"], coh_counts["Other"]),
                use_container_width=True
            )

# Preset periods and Custom remain as in the prior version (including Custom denominator option).
today = date.today()
yday = today - timedelta(days=1)
last_m_start, last_m_end = last_month_bounds(today)
this_m_start, this_m_end = month_bounds(today)

if view == "MIS":
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

        # -------- Custom tab (with anchor or custom denominator range) --------
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
                    render_period_block("Custom (anchor month)", custom_start, custom_end, anchor)
                else:
                    cold1, cold2 = st.columns(2)
                    with cold1:
                        denom_start = st.date_input("Denominator start (deals created from)", value=custom_start, key="denom_start")
                    with cold2:
                        denom_end = st.date_input("Denominator end (deals created to)", value=custom_end, key="denom_end")
                    if denom_end < denom_start:
                        st.error("Denominator end cannot be before start.")
                    else:
                        # For counts/MTD month anchoring, use custom_start month (visual consistency)
                        anchor_for_counts = custom_start
                        # Render counts
                        mtd_counts, coh_counts = prepare_counts_for_range(
                            df_f, custom_start, custom_end, anchor_for_counts,
                            create_col, pay_col, pipeline_col
                        )
                        c1, c2 = st.columns(2)
                        with c1:
                            st.altair_chart(bubble_chart_counts("MTD Enrolments (counts)", mtd_counts["Total"], mtd_counts["AI Coding"], mtd_counts["Math"], mtd_counts["Other"]), use_container_width=True)
                        with c2:
                            st.altair_chart(bubble_chart_counts("Cohort Enrolments (counts)", coh_counts["Total"], coh_counts["AI Coding"], coh_counts["Math"], coh_counts["Other"]), use_container_width=True)
                        # Conversion% with custom range denom
                        mtd_pct, coh_pct, denom, nums = prepare_conversion_for_range(
                            df_f, custom_start, custom_end, create_col, pay_col, pipeline_col,
                            denom_mode="range", denom_start=denom_start, denom_end=denom_end
                        )
                        st.caption(f"Conversion% denominator (deals created in custom range): **{denom:,}**")
                        d1, d2 = st.columns(2)
                        with d1:
                            st.altair_chart(bubble_chart_pct("MTD Conversion %", mtd_pct["Total"], mtd_pct["AI Coding"], mtd_pct["Math"], denom, nums["mtd"], "mtd"), use_container_width=True)
                        with d2:
                            st.altair_chart(bubble_chart_pct("Cohort Conversion %", coh_pct["Total"], coh_pct["AI Coding"], coh_pct["Math"], denom, nums["cohort"], "cohort"), use_container_width=True)

                        if show_mix:
                            m1, m2 = st.columns(2)
                            with m1:
                                st.altair_chart(mix_share_chart("MTD pipeline mix (% of enrolments)", mtd_counts["AI Coding"], mtd_counts["Math"], mtd_counts["Other"]), use_container_width=True)
                            with m2:
                                st.altair_chart(mix_share_chart("Cohort pipeline mix (% of enrolments)", coh_counts["AI Coding"], coh_counts["Math"], coh_counts["Other"]), use_container_width=True)

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
