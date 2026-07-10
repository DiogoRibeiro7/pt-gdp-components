"""Turn levels into contributions to quarter-on-quarter GDP growth.

Contribution of component i in quarter t, chain-linked-volume approximation:

    contrib_{i,t} = sign_i * (X_{i,t} - X_{i,t-1}) / GDP_{t-1} * 100

Chain-linked volumes are non-additive away from the reference year, so the
contributions do not sum exactly to GDP growth. The residual is computed
explicitly and, by default, reallocated across components in proportion to
the absolute size of their contribution, so the adding-up identity that the
SUR design relies on holds exactly by construction. Set
`allocate_residual=False` to keep the raw approximation and carry the
residual as its own column instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def _resolve_components(clv: pd.DataFrame) -> dict[str, dict]:
    """Use P52/P53 if both present, otherwise fall back to P52_P53."""
    comps = dict(config.COMPONENTS)
    if not {"P52", "P53"}.issubset(clv.columns):
        comps.pop("P52", None)
        comps.pop("P53", None)
        if config.INVENTORY_FALLBACK in clv.columns:
            comps[config.INVENTORY_FALLBACK] = {
                "label": "Changes in inventories (incl. valuables)",
                "sign": +1,
            }
    return {k: v for k, v in comps.items() if k in clv.columns}


def _gdp_growth(clv: pd.DataFrame) -> pd.Series:
    gdp = clv[config.GDP_ITEM]
    return 100 * gdp.diff() / gdp.shift(1)


def _raw_approx(clv: pd.DataFrame, comps: dict[str, dict]) -> pd.DataFrame:
    """Naive Δ/GDP approximation: sign_i · ΔX_{i,t} / GDP_{t-1} · 100."""
    gdp = clv[config.GDP_ITEM]
    contrib = pd.DataFrame(index=clv.index)
    for item, meta in comps.items():
        contrib[item] = meta["sign"] * 100 * clv[item].diff() / gdp.shift(1)
    return contrib


def _annual_cp_shares(cp: pd.DataFrame, comps: dict[str, dict]) -> pd.DataFrame:
    """Annual current-price nominal share of each component in nominal GDP.

    Shares are computed from annual sums of the CP frame; the denominator is
    current-price GDP (which, unlike chain-linked volumes, is additive by
    construction). Indexed by calendar year.
    """
    annual = cp.groupby(cp.index.year).sum(min_count=1)
    gdp_cp = annual[config.GDP_ITEM]
    shares = pd.DataFrame(index=annual.index)
    for item in comps:
        shares[item] = annual[item] / gdp_cp
    return shares


def _raw_exact(clv: pd.DataFrame, cp: pd.DataFrame,
               comps: dict[str, dict]) -> pd.DataFrame:
    """Annual-overlap exact contributions to chain-linked-volume growth.

        contrib_{i,t} = sign_i · s_{i,y(t)-1} · (X_{i,t}/X_{i,t-1} − 1) · 100

    with s the previous calendar year's current-price share of the component
    in nominal GDP. This is the standard additive decomposition of a
    chain-linked Laspeyres volume aggregate: within a linking year the
    aggregate is a fixed-previous-year-price index, so previous-year nominal
    shares are the correct weights on quarter-on-quarter volume ratios.
    """
    shares = _annual_cp_shares(cp, comps)
    prev_year = clv.index.year - 1
    contrib = pd.DataFrame(index=clv.index)
    for item, meta in comps.items():
        s_prev = shares[item].reindex(prev_year).to_numpy()
        g = (clv[item] / clv[item].shift(1) - 1.0).to_numpy()
        contrib[item] = meta["sign"] * s_prev * g * 100.0
    return contrib


def _finalize(contrib: pd.DataFrame, gdp_growth: pd.Series,
              allocate_residual: bool) -> tuple[pd.DataFrame, pd.Series]:
    contrib = contrib.dropna()
    gdp_growth = gdp_growth.loc[contrib.index]
    residual = gdp_growth - contrib.sum(axis=1)
    if allocate_residual:
        weights = contrib.abs().div(contrib.abs().sum(axis=1), axis=0)
        contrib = contrib.add(weights.mul(residual, axis=0), fill_value=0.0)
        gap = (gdp_growth - contrib.sum(axis=1)).abs().max()
        assert gap < 1e-9, f"adding-up identity violated, max gap {gap}"
    else:
        contrib = contrib.copy()
        contrib["chain_residual"] = residual
    return contrib, gdp_growth


def contributions_approx(
    clv: pd.DataFrame, allocate_residual: bool = True
) -> tuple[pd.DataFrame, pd.Series, dict[str, dict]]:
    """Naive Δ/GDP approximation, retained as a robustness check.

    Chain-linked volumes are non-additive away from the reference year, so
    these contributions do not sum exactly to GDP growth; with
    ``allocate_residual=True`` the residual is reallocated proportionally to
    the absolute size of each contribution so the adding-up identity holds
    for the SUR layer, otherwise it is carried as a ``chain_residual`` column.
    """
    comps = _resolve_components(clv)
    contrib = _raw_approx(clv, comps)
    contrib, gdp_growth = _finalize(contrib, _gdp_growth(clv), allocate_residual)
    return contrib, gdp_growth, comps


def contributions_exact(
    clv: pd.DataFrame, cp: pd.DataFrame, allocate_residual: bool = True
) -> tuple[pd.DataFrame, pd.Series, dict[str, dict]]:
    """Annual-overlap exact contributions (default method).

    Uses previous-year current-price weights on quarter-on-quarter volume
    ratios (see :func:`_raw_exact`). A small residual against GDP growth
    remains — chain-linking still binds only annually while GDP growth here
    is a quarter-on-quarter volume ratio — and is reallocated proportionally
    when ``allocate_residual=True`` so the SUR adding-up identity closes.
    """
    comps = _resolve_components(clv)
    contrib = _raw_exact(clv, cp, comps)
    contrib, gdp_growth = _finalize(contrib, _gdp_growth(clv), allocate_residual)
    return contrib, gdp_growth, comps


def contributions(
    clv: pd.DataFrame, cp: pd.DataFrame | None = None,
    method: str = "exact", allocate_residual: bool = True
) -> tuple[pd.DataFrame, pd.Series, dict[str, dict]]:
    """Dispatch to the exact (default) or approximate contribution method."""
    if method == "exact":
        if cp is None:
            raise ValueError("exact contributions require the current-price (CP) frame")
        return contributions_exact(clv, cp, allocate_residual=allocate_residual)
    if method == "approx":
        return contributions_approx(clv, allocate_residual=allocate_residual)
    raise ValueError(f"unknown method {method!r}; expected 'exact' or 'approx'")


def convention_comparison(clv: pd.DataFrame, cp: pd.DataFrame) -> pd.DataFrame:
    """Per-quarter comparison of contribution-convention residuals.

    Columns: gdp_growth, sum_exact, sum_approx_raw, sum_approx_reallocated,
    and the three residuals against GDP growth. The reallocated approximation
    closes the identity by construction (residual ≈ 0); the raw sums show how
    much each convention misses before any reallocation.
    """
    comps = _resolve_components(clv)
    gdp_growth = _gdp_growth(clv)
    ex_raw = _raw_exact(clv, cp, comps)
    ap_raw = _raw_approx(clv, comps)

    idx = ex_raw.dropna().index.intersection(ap_raw.dropna().index)
    g = gdp_growth.loc[idx]
    sum_exact = ex_raw.loc[idx].sum(axis=1)
    sum_approx_raw = ap_raw.loc[idx].sum(axis=1)

    ap_re, _ = _finalize(ap_raw.copy(), gdp_growth.copy(), allocate_residual=True)
    sum_approx_reallocated = ap_re.loc[idx].sum(axis=1)

    return pd.DataFrame(
        {
            "gdp_growth": g,
            "sum_exact": sum_exact,
            "sum_approx_raw": sum_approx_raw,
            "sum_approx_reallocated": sum_approx_reallocated,
            "residual_exact": g - sum_exact,
            "residual_approx_raw": g - sum_approx_raw,
            "residual_approx_reallocated": g - sum_approx_reallocated,
        }
    )


def design_matrix(index: pd.PeriodIndex, interactions: bool = False) -> pd.DataFrame:
    """Common regressors for every equation: intercept, trend, regime dummies.

    Trend is in decades so coefficients read as pp of quarterly growth per
    decade; otherwise they are invisibly small.

    With ``interactions=True`` a ``trend×<regime>`` regressor is added for
    each regime, equal to the regime dummy times the trend re-centred at the
    start of that regime's window (``trend − trend_at_regime_entry``, zero
    outside the window). Re-centring makes the two coefficients read cleanly:
    the dummy is the level shift at regime entry (pp of quarterly growth) and
    the interaction is the within-regime slope change (pp per decade). The
    alternative — a raw ``trend×dummy`` product without re-centring — was
    rejected because it confounds the level shift with the slope, so the
    dummy would then measure the extrapolated regime effect at t=0 rather
    than the interpretable jump at regime entry.
    """
    X = pd.DataFrame(index=index)
    X["const"] = 1.0
    trend = np.arange(len(index)) / 40.0  # quarters to decades
    X["trend"] = trend
    for name, (start, end) in config.REGIMES.items():
        dummy = (
            (index >= pd.Period(start, freq="Q")) & (index <= pd.Period(end, freq="Q"))
        ).astype(float)
        X[name] = dummy
    if interactions:
        for name, (start, end) in config.REGIMES.items():
            pos = int(index.searchsorted(pd.Period(start, freq="Q")))
            centre = pos / 40.0  # trend value at regime entry
            X[f"trend_{name}"] = X[name].to_numpy() * (trend - centre)
    return X
