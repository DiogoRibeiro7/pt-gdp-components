"""Distributional effects via quantile regression.

The SUR mean model describes the conditional mean of each contribution. That is
the right object for an adding-up decomposition, but it is silent on the tails:
whether a regime hits the bad quarters harder than the good ones, or whether the
trend is steeper where growth is already weak. Quantile regression answers those
questions by fitting the same design (intercept, trend in decades, regime
dummies) at a grid of conditional quantiles of GDP growth and of each component
contribution.

The reading is deliberately complementary, not a replacement: where a regime
coefficient is roughly flat across quantiles the mean effect is a good summary;
where it fans out, the effect is asymmetric and the mean understates the damage
in the lower tail. The alternative of reporting only the OLS mean was rejected
because it cannot show that asymmetry, which for a small open economy is exactly
where the policy-relevant risk sits.

Quantile fits that fail to converge for a particular series and quantile are
logged and skipped rather than crashing the run.
"""

from __future__ import annotations

import pandas as pd
from statsmodels.regression.quantile_regression import QuantReg

DEFAULT_TAUS = (0.1, 0.25, 0.5, 0.75, 0.9)


def quantile_paths(y: pd.Series, X: pd.DataFrame,
                   taus=DEFAULT_TAUS) -> pd.DataFrame:
    """Coefficient paths across quantiles for one dependent series.

    Returns a tidy frame: tau, regressor, coef, ci_low, ci_high.
    """
    y = y.loc[X.index]
    rows = []
    for tau in taus:
        try:
            res = QuantReg(y, X).fit(q=tau)
        except Exception as exc:  # noqa: BLE001 - log and skip this quantile
            print(f"[quantile] skipped q={tau}: {exc}")
            continue
        ci = res.conf_int()
        for reg in X.columns:
            rows.append({
                "tau": tau,
                "regressor": reg,
                "coef": float(res.params[reg]),
                "ci_low": float(ci.loc[reg, 0]),
                "ci_high": float(ci.loc[reg, 1]),
            })
    return pd.DataFrame(rows)


def quantile_table(contrib: pd.DataFrame, gdp_growth: pd.Series,
                   X: pd.DataFrame, taus=DEFAULT_TAUS) -> pd.DataFrame:
    """Quantile coefficient paths for GDP growth and every component.

    Returns a tidy frame: equation, tau, regressor, coef, ci_low, ci_high.
    """
    frames = []
    gp = quantile_paths(gdp_growth, X, taus)
    gp.insert(0, "equation", "GDP (system sum)")
    frames.append(gp)
    for comp in contrib.columns:
        f = quantile_paths(contrib[comp], X, taus)
        f.insert(0, "equation", comp)
        frames.append(f)
    return pd.concat(frames, ignore_index=True)
