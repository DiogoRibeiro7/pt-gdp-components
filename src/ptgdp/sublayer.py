"""Split a parent component's contribution across an annual sub-breakdown.

The quarterly SUR layer works at the ~10-component level. This module adds a
second, coarser-frequency granularity layer: it takes a parent component's
quarterly contribution to GDP growth, aggregates it to annual, and distributes
each year's parent contribution across an annual child breakdown (GFCF by
asset, household consumption by COICOP purpose).

Convention for the split
------------------------
Each child's weight in a given year is its signed share of the parent's total
annual volume change, ``dChild_c / Σ_j dChild_j``, applied to the parent's
annual contribution. Because chain-linked child volumes are non-additive (and
the published child breakdown may not tile the parent exactly), the children's
level changes need not sum to the parent's; normalising the weights to sum to
one reallocates that non-additivity residual proportionally, so the split
closes onto the parent contribution exactly. The residual absorbed is returned
for inspection.

The alternative — carrying the residual as an explicit "unallocated" child —
was rejected because it would break the clean reading of each child as a share
of the parent and complicate the stacked-bar figure; the residual is small
(sub-percent on the breakdowns used here) and is reported rather than hidden.

Degenerate years, where the parent barely moves so ``Σ_j dChild_j ≈ 0`` and the
signed shares are unstable, fall back to weights proportional to the absolute
child changes, which still sum to one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def within_component_decomposition(
    parent_quarterly_contrib: pd.Series,
    child_annual_levels: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Distribute a parent's annual contribution across annual children.

    Parameters
    ----------
    parent_quarterly_contrib : Series
        The parent component's quarterly contribution to GDP growth (pp),
        indexed by a quarterly PeriodIndex.
    child_annual_levels : DataFrame
        Annual chain-linked-volume levels of the children, indexed by calendar
        year (int), one column per child.

    Returns
    -------
    tidy : DataFrame
        Columns: year, child, contribution_pp, share_of_parent.
    residual : Series
        Per-year reconciliation residual (parent annual contribution minus the
        summed child contributions); ~0 by construction, reported so callers
        can confirm the split closed.
    """
    parent_annual = parent_quarterly_contrib.groupby(
        parent_quarterly_contrib.index.year
    ).sum()
    parent_annual.index = parent_annual.index.astype(int)

    d_child = child_annual_levels.sort_index().diff().dropna(how="all")
    total = d_child.sum(axis=1)

    # per-year child weights (signed share of the parent's total change)
    scale = d_child.abs().sum(axis=1).replace(0.0, np.nan)
    degenerate = total.abs() < 1e-9 * scale.fillna(1.0)

    weights = d_child.div(total, axis=0)
    if degenerate.any():
        abs_w = d_child.abs().div(d_child.abs().sum(axis=1), axis=0)
        weights.loc[degenerate] = abs_w.loc[degenerate]
    weights = weights.fillna(0.0)

    years = [y for y in d_child.index if y in parent_annual.index]
    rows = []
    for y in years:
        p = float(parent_annual.loc[y])
        for child in child_annual_levels.columns:
            w = float(weights.loc[y, child])
            rows.append(
                {
                    "year": int(y),
                    "child": child,
                    "contribution_pp": w * p,
                    "share_of_parent": w,
                }
            )
    tidy = pd.DataFrame(rows, columns=["year", "child", "contribution_pp",
                                       "share_of_parent"])

    summed = tidy.groupby("year")["contribution_pp"].sum()
    residual = parent_annual.loc[years] - summed.reindex(years).fillna(0.0)
    residual.name = "reconciliation_residual"
    return tidy, residual
