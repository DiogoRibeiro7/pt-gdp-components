"""Long-run structure of the components: a VECM on log CLV levels.

We take the natural log of the chain-linked-volume levels of the expenditure
components and ask whether they share long-run cointegrating relations, and —
the economically interesting part — which components do the adjusting when the
system drifts away from that long-run configuration (the alpha loading matrix).

Excluded series (stated prominently, logged at run time, never silent)
---------------------------------------------------------------------
Changes in inventories (P52), acquisitions of valuables (P53) and the combined
P52_P53 are dropped: inventory-type series take non-positive values, so their
log is undefined. Any other series that happens to contain a non-positive
value is dropped on the same grounds and logged. The VECM therefore runs on
the strictly positive final-demand components only (typically seven).

Size-distortion caveat
----------------------
With ~7 log-level series and only ~120 quarters, the Johansen trace test is
badly size-distorted and tends to over-select the cointegrating rank. The rank
selection here is descriptive, not a hypothesis test to be believed at face
value; results are reported for the selected rank r and for r±1 as sensitivity
(all three alpha tables are written) so the reader can see how much the loadings
depend on that choice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import (
    VECM,
    coint_johansen,
    select_order,
)

from . import config

_EXCLUDE = {"P52", "P53", config.INVENTORY_FALLBACK}


def log_levels(clv: pd.DataFrame) -> pd.DataFrame:
    """Log CLV levels of the strictly positive final-demand components."""
    candidates = [c for c in config.COMPONENTS if c in clv.columns and c not in _EXCLUDE]
    kept, dropped = [], []
    for c in candidates:
        if (clv[c] > 0).all():
            kept.append(c)
        else:
            dropped.append(c)
    for c in _EXCLUDE:
        if c in clv.columns:
            dropped.append(c)
    if dropped:
        print(f"[vecm] excluded non-positive / inventory series (log undefined): "
              f"{sorted(set(dropped))}")
    print(f"[vecm] VECM on log levels of: {kept}")
    return np.log(clv[kept]).dropna()


def johansen_table(data: pd.DataFrame, k_ar_diff: int,
                   det_order: int = 1) -> tuple[pd.DataFrame, int]:
    """Johansen trace test table and the 5%-selected cointegrating rank."""
    joh = coint_johansen(data, det_order, k_ar_diff)
    trace = joh.lr1
    crit = joh.cvt  # columns: 90%, 95%, 99%
    rows, rank, still = [], 0, True
    for i in range(len(trace)):
        reject = bool(trace[i] > crit[i, 1])
        if still and reject:
            rank = i + 1
        elif still:
            still = False
        rows.append({
            "null_rank_r": i,
            "trace_stat": float(trace[i]),
            "crit_90pct": float(crit[i, 0]),
            "crit_95pct": float(crit[i, 1]),
            "crit_99pct": float(crit[i, 2]),
            "reject_at_5pct": reject,
        })
    return pd.DataFrame(rows), rank


def _alpha_frame(res, names: list[str], rank: int) -> pd.DataFrame:
    alpha = np.asarray(res.alpha)
    try:
        se = np.asarray(res.stderr_alpha)
    except Exception:
        se = np.full_like(alpha, np.nan)
    try:
        pval = np.asarray(res.pvalues_alpha)
    except Exception:
        pval = np.full_like(alpha, np.nan)
    rows = []
    for i, comp in enumerate(names):
        for j in range(alpha.shape[1]):
            rows.append({
                "rank": rank,
                "component": comp,
                "relation": f"ce{j + 1}",
                "alpha": float(alpha[i, j]),
                "stderr": float(se[i, j]),
                "pvalue": float(pval[i, j]),
            })
    return pd.DataFrame(rows)


def fit_alpha_tables(data: pd.DataFrame, k_ar_diff: int, selected_rank: int,
                     det_order: int = 1):
    """Fit the VECM at ranks {r−1, r, r+1} and return stacked alpha tables.

    Returns ``(alpha_all, alpha_selected)`` where ``alpha_all`` stacks the
    three sensitivity ranks (clipped to a valid [1, k−1]) and
    ``alpha_selected`` is the table for the selected rank alone (for the
    heatmap). ``deterministic='ci'`` — an intercept restricted to the
    cointegrating relation — matches ``det_order=1`` in the Johansen step.
    """
    k = data.shape[1]
    names = list(data.columns)
    ranks = sorted({r for r in (selected_rank - 1, selected_rank, selected_rank + 1)
                    if 1 <= r <= k - 1})
    tables, selected = [], None
    for r in ranks:
        res = VECM(data, k_ar_diff=k_ar_diff, coint_rank=r,
                   deterministic="ci").fit()
        frame = _alpha_frame(res, names, r)
        tables.append(frame)
        if r == selected_rank:
            selected = frame
    alpha_all = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    if selected is None and tables:
        selected = tables[len(tables) // 2]  # middle rank if selected was clipped away
    return alpha_all, selected


def run(clv: pd.DataFrame, det_order: int = 1):
    """End-to-end VECM: Johansen table, rank selection, alpha sensitivity.

    Returns ``(johansen_df, alpha_all, alpha_selected, k_ar_diff, rank)``.
    """
    data = log_levels(clv)
    lag = select_order(data, maxlags=4, deterministic="ci")
    k_ar_diff = int(getattr(lag, "bic", 1) or 1)
    k_ar_diff = min(max(k_ar_diff, 1), 4)  # BIC over 1..4
    joh_df, rank = johansen_table(data, k_ar_diff, det_order=det_order)
    fit_rank = max(rank, 1)  # VECM needs at least one cointegrating relation
    alpha_all, alpha_selected = fit_alpha_tables(data, k_ar_diff, fit_rank,
                                                 det_order=det_order)
    return joh_df, alpha_all, alpha_selected, k_ar_diff, rank
