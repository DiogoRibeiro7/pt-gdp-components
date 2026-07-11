"""A common cycle across the components: a dynamic factor model.

The SUR layer treats each component contribution as its own equation and reports
a singular cross-equation residual covariance as a by-product. That covariance
says the components co-move, but it does not name the co-movement. A dynamic
factor model does: it extracts a single latent common factor (with AR(1)
dynamics) plus idiosyncratic AR(1) noise per series, and asks how much of each
component's variation the one common cycle accounts for.

The factor is standardised and sign-normalised so that it loads positively on
the average component, which makes it read as a broad "demand cycle": components
with large positive loadings move with it, those near zero march to their own
drummer. The average share of component variance tracked by the factor is a
one-number summary of how synchronised the expenditure side is.

The alternative of static principal components was rejected because it ignores
the serial dependence in both the factor and the idiosyncratic terms, which for
quarterly macro series is first-order; the state-space dynamic factor handles it
directly. Non-convergence is retried under Powell and then logged and skipped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor


def fit_factor(contrib: pd.DataFrame):
    """Fit a one-factor dynamic factor model to the standardised contributions.

    Returns ``(factor_df, loadings_df, mean_r2)`` where ``factor_df`` has the
    smoothed common factor with a 90% band, ``loadings_df`` has the per-component
    loading and R-squared on that factor, and ``mean_r2`` is the average share of
    component variance the factor accounts for. Returns ``(None, None, None)`` on
    a failed fit.
    """
    data = contrib.dropna()
    sd = data.std().replace(0.0, 1.0)
    z = (data - data.mean()) / sd

    res = None
    for method in ("lbfgs", "powell"):
        try:
            res = DynamicFactor(z, k_factors=1, factor_order=1,
                                error_order=1).fit(method=method, disp=False,
                                                   maxiter=200)
            break
        except Exception as exc:  # noqa: BLE001 - retry then skip
            last = exc
            res = None
    if res is None:
        print(f"[factor] skipped: {last}")
        return None, None, None

    factor = np.asarray(res.factors.smoothed).reshape(len(z), -1)[:, 0]
    # sign-normalise so the factor loads positively on the average component
    ref = z.mean(axis=1).to_numpy()
    if np.corrcoef(factor, ref)[0, 1] < 0:
        factor = -factor

    try:
        cov = np.asarray(res.factors.smoothed_cov)
        se = np.sqrt(np.clip(cov.reshape(-1, len(z))[0], 0.0, None)) \
            if cov.ndim == 3 else np.full(len(z), np.nan)
    except Exception:
        se = np.full(len(z), np.nan)

    factor_df = pd.DataFrame(
        {"factor": factor, "lo90": factor - 1.645 * se, "hi90": factor + 1.645 * se},
        index=z.index,
    )

    rows = []
    for c in z.columns:
        slope = float(np.polyfit(factor, z[c].to_numpy(), 1)[0])
        r = float(np.corrcoef(factor, z[c].to_numpy())[0, 1])
        rows.append({"component": c, "loading": slope, "r2": r ** 2})
    loadings_df = pd.DataFrame(rows)
    mean_r2 = float(loadings_df["r2"].mean())
    return factor_df, loadings_df, mean_r2
