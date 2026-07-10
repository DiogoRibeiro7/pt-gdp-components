"""System estimation of component contributions with the adding-up constraint.

Every equation shares the same regressors (intercept, trend, regime
dummies). Under identical regressors, GLS/SUR collapses to equation-by-
equation OLS (Kruskal's theorem), and because the dependent variables sum
to GDP growth by construction, the coefficient vectors sum across
equations to the coefficients of the GDP-growth equation exactly. So the
"constraint" is not imposed; it is inherited, and verified numerically.

Inference is per-equation OLS with Newey-West (HAC) standard errors,
maxlags=4, which is the honest choice for quarterly contributions with
serial correlation. The residual covariance across equations is singular
(the residuals sum to the GDP-equation residual), which is expected and
harmless for this design; it is reported for inspection.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass
class SURResult:
    params: pd.DataFrame          # components x regressors
    bse: pd.DataFrame             # HAC standard errors
    pvalues: pd.DataFrame
    gdp_params: pd.Series         # GDP-growth equation coefficients
    gdp_bse: pd.Series
    gdp_pvalues: pd.Series
    adding_up_gap: float          # max |sum of component betas - gdp beta|
    resid: pd.DataFrame = field(repr=False)
    nobs: int = 0

    def summary_table(self) -> pd.DataFrame:
        """Long-format table: component, regressor, coef, se, p."""
        rows = []
        for comp in self.params.index:
            for reg in self.params.columns:
                rows.append(
                    {
                        "component": comp,
                        "regressor": reg,
                        "coef": self.params.loc[comp, reg],
                        "hac_se": self.bse.loc[comp, reg],
                        "pvalue": self.pvalues.loc[comp, reg],
                    }
                )
        for reg in self.gdp_params.index:
            rows.append(
                {
                    "component": "GDP (system sum)",
                    "regressor": reg,
                    "coef": self.gdp_params[reg],
                    "hac_se": self.gdp_bse[reg],
                    "pvalue": self.gdp_pvalues[reg],
                }
            )
        return pd.DataFrame(rows)


def fit(contrib: pd.DataFrame, gdp_growth: pd.Series, X: pd.DataFrame,
        hac_maxlags: int = 4) -> SURResult:
    X = X.loc[contrib.index]
    params, bse, pvals, resids = {}, {}, {}, {}

    for comp in contrib.columns:
        res = sm.OLS(contrib[comp], X).fit(
            cov_type="HAC", cov_kwds={"maxlags": hac_maxlags}
        )
        params[comp], bse[comp], pvals[comp] = res.params, res.bse, res.pvalues
        resids[comp] = res.resid

    gdp_res = sm.OLS(gdp_growth.loc[X.index], X).fit(
        cov_type="HAC", cov_kwds={"maxlags": hac_maxlags}
    )

    params = pd.DataFrame(params).T
    gap = float((params.sum(axis=0) - gdp_res.params).abs().max())

    return SURResult(
        params=params,
        bse=pd.DataFrame(bse).T,
        pvalues=pd.DataFrame(pvals).T,
        gdp_params=gdp_res.params,
        gdp_bse=gdp_res.bse,
        gdp_pvalues=gdp_res.pvalues,
        adding_up_gap=gap,
        resid=pd.DataFrame(resids),
        nobs=int(gdp_res.nobs),
    )


def regime_means(contrib: pd.DataFrame, gdp_growth: pd.Series,
                 regimes: dict[str, tuple[str, str]]) -> pd.DataFrame:
    """Average contribution per component per regime, plus 'normal' periods."""
    idx = contrib.index
    masks = {}
    covered = np.zeros(len(idx), dtype=bool)
    for name, (start, end) in regimes.items():
        m = (idx >= pd.Period(start, "Q")) & (idx <= pd.Period(end, "Q"))
        masks[name] = m
        covered |= np.asarray(m)
    masks["normal"] = ~covered

    out = {}
    for name, m in masks.items():
        block = contrib.loc[m].mean()
        block["GDP growth"] = gdp_growth.loc[m].mean()
        out[name] = block
    return pd.DataFrame(out)
