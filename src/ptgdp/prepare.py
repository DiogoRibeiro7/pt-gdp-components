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


def contributions(
    clv: pd.DataFrame, allocate_residual: bool = True
) -> tuple[pd.DataFrame, pd.Series, dict[str, dict]]:
    """Returns (contributions in pp of QoQ growth, GDP QoQ growth in %, components used)."""
    comps = _resolve_components(clv)
    gdp = clv[config.GDP_ITEM]
    gdp_growth = 100 * gdp.diff() / gdp.shift(1)

    contrib = pd.DataFrame(index=clv.index)
    for item, meta in comps.items():
        contrib[item] = meta["sign"] * 100 * clv[item].diff() / gdp.shift(1)

    contrib = contrib.dropna()
    gdp_growth = gdp_growth.loc[contrib.index]

    residual = gdp_growth - contrib.sum(axis=1)
    if allocate_residual:
        weights = contrib.abs().div(contrib.abs().sum(axis=1), axis=0)
        contrib = contrib.add(weights.mul(residual, axis=0), fill_value=0.0)
    else:
        contrib["chain_residual"] = residual

    # sanity: identity must close after allocation
    if allocate_residual:
        gap = (gdp_growth - contrib.sum(axis=1)).abs().max()
        assert gap < 1e-9, f"adding-up identity violated, max gap {gap}"

    return contrib, gdp_growth, comps


def design_matrix(index: pd.PeriodIndex) -> pd.DataFrame:
    """Common regressors for every equation: intercept, trend, regime dummies.

    Trend is in decades so coefficients read as pp of quarterly growth per
    decade; otherwise they are invisibly small.
    """
    X = pd.DataFrame(index=index)
    X["const"] = 1.0
    X["trend"] = np.arange(len(index)) / 40.0  # quarters to decades
    for name, (start, end) in config.REGIMES.items():
        X[name] = (
            (index >= pd.Period(start, freq="Q")) & (index <= pd.Period(end, freq="Q"))
        ).astype(float)
    return X
