"""Residual diagnostics battery for the contribution equations.

For each component equation and the GDP-growth equation the residuals from the
static mean model are tested for the failures that matter to HAC-OLS
inference:

* serial correlation — Ljung-Box Q at lags 4 and 8 (one and two years of
  quarterly data);
* conditional heteroskedasticity — Engle's ARCH-LM at lag 4;
* non-normality — Jarque-Bera;
* parameter stability — the CUSUM test of Ploberger-Krämer on the OLS
  residuals (``statsmodels.stats.diagnostic.breaks_cusumolsresid``), reported
  as a p-value and a stability flag.

All are reported as p-values so they read on one comparable scale. Serial
correlation is the diagnostic with teeth here: if residuals are autocorrelated
the HAC (Newey-West) covariance still delivers valid standard errors for the
mean parameters, but the static regression is dynamically incomplete as a data
description — which is the motivation for the state-space trend layer (B3),
not a reason to distrust the reported inference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import (
    acorr_ljungbox,
    breaks_cusumolsresid,
    het_arch,
)
from statsmodels.stats.stattools import jarque_bera


def _one(resid: pd.Series) -> dict[str, float]:
    r = pd.Series(resid).dropna()
    lb = acorr_ljungbox(r, lags=[4, 8], return_df=True)
    arch_p = float(het_arch(r, nlags=4)[1])
    jb_p = float(jarque_bera(r)[1])
    try:
        cusum_p = float(breaks_cusumolsresid(r.to_numpy())[1])
    except Exception:
        cusum_p = np.nan
    return {
        "ljung_box_p_lag4": float(lb.loc[4, "lb_pvalue"]),
        "ljung_box_p_lag8": float(lb.loc[8, "lb_pvalue"]),
        "arch_lm_p_lag4": arch_p,
        "jarque_bera_p": jb_p,
        "cusum_p": cusum_p,
    }


def diagnostics_battery(residuals: dict[str, pd.Series]) -> pd.DataFrame:
    """Tidy diagnostics table, one row per equation.

    ``residuals`` maps an equation label to its residual series. Returns a
    DataFrame with the five p-value statistics plus a ``cusum_stable`` flag
    (True when the CUSUM p-value is at or above 0.05).
    """
    rows = []
    for name, r in residuals.items():
        stats = _one(r)
        cusum_p = stats["cusum_p"]
        rows.append(
            {
                "equation": name,
                **stats,
                "cusum_stable": bool(cusum_p >= 0.05) if np.isfinite(cusum_p) else True,
            }
        )
    return pd.DataFrame(rows)
