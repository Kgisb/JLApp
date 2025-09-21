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
        # total per source
        gsize = sub.groupby(source_col).size().rename("total")
        # same-month per source
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
