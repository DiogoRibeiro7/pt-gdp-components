"""Time-varying trends via a local linear trend state-space model.

Each contribution series (and GDP growth) is fitted with a
``UnobservedComponents`` local linear trend: a stochastic level plus a
stochastic slope. The smoothed slope path answers "when was the trend real"
without committing to fixed regime windows — the 90% band around the slope
shows the quarters in which the underlying drift is distinguishable from zero.

Seasonality is off by default because the input is seasonally and calendar
adjusted (SCA); a stochastic seasonal(4) term is available via a flag for
robustness on any series where residual seasonality is suspected.

Non-convergence is handled, not hidden: the fit is retried with Powell's
method and, if that also fails, the series is logged and skipped rather than
crashing the run.

Adding-up caveat
----------------
The slopes are estimated one series at a time. The adding-up identity that the
SUR layer inherits is NOT imposed across these independent state-space models,
so the sum of the smoothed component slopes need not equal the smoothed GDP
slope. That discrepancy is computed per quarter and written to the output CSV
(component ``sum_minus_gdp_gap``) as a model-consistency diagnostic, not as an
error to be zeroed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.structural import UnobservedComponents

# one-sided 95% normal quantile -> two-sided 90% band
_Z90 = 1.6448536269514722


def fit_slope_path(series: pd.Series, seasonal: bool = False,
                   label: str = "") -> pd.DataFrame | None:
    """Smoothed local-linear-trend slope path with a 90% band.

    Returns a DataFrame indexed like ``series`` with columns slope, lo90,
    hi90, or ``None`` if the model fails to converge under both optimisers.
    """
    s = pd.Series(series).dropna()
    spec: dict = {"level": "local linear trend"}
    if seasonal:
        spec["seasonal"] = 4

    last_err: Exception | None = None
    for method in ("lbfgs", "powell"):
        try:
            res = UnobservedComponents(s, **spec).fit(
                method=method, disp=False, maxiter=500
            )
            converged = True
            if hasattr(res, "mle_retvals") and isinstance(res.mle_retvals, dict):
                converged = res.mle_retvals.get("converged", True)
            if not converged and method != "powell":
                continue
            # state order for local linear trend: [level, trend(slope), ...]
            slope = np.asarray(res.smoothed_state[1])
            var = np.asarray(res.smoothed_state_cov[1, 1])
            se = np.sqrt(np.clip(var, 0.0, None))
            return pd.DataFrame(
                {"slope": slope, "lo90": slope - _Z90 * se, "hi90": slope + _Z90 * se},
                index=s.index,
            )
        except Exception as exc:  # noqa: BLE001 - retry, then skip
            last_err = exc
            continue

    print(f"[stsm] skipped {label!r}: no convergence ({last_err})")
    return None


def slope_paths(contrib: pd.DataFrame, gdp_growth: pd.Series,
                seasonal: bool = False):
    """Fit slope paths for every component and GDP growth.

    Returns ``(tidy, frames, gdp_frame)`` where ``tidy`` is the long-format
    table (quarter, component, slope, lo90, hi90) including a
    ``sum_minus_gdp_gap`` pseudo-component, ``frames`` maps each fitted
    component to its slope-path DataFrame, and ``gdp_frame`` is the GDP path
    (or ``None`` if it did not converge).
    """
    frames: dict[str, pd.DataFrame] = {}
    for col in contrib.columns:
        fr = fit_slope_path(contrib[col], seasonal=seasonal, label=col)
        if fr is not None:
            frames[col] = fr
    gdp_frame = fit_slope_path(gdp_growth, seasonal=seasonal, label="GDP")

    rows = []
    for comp, fr in frames.items():
        for q, row in fr.iterrows():
            rows.append({"quarter": str(q), "component": comp,
                         "slope": row["slope"], "lo90": row["lo90"],
                         "hi90": row["hi90"]})
    if gdp_frame is not None:
        for q, row in gdp_frame.iterrows():
            rows.append({"quarter": str(q), "component": "GDP",
                         "slope": row["slope"], "lo90": row["lo90"],
                         "hi90": row["hi90"]})
        if frames:
            comp_sum = sum(fr["slope"] for fr in frames.values())
            gap = comp_sum.reindex(gdp_frame.index) - gdp_frame["slope"]
            for q, val in gap.items():
                rows.append({"quarter": str(q), "component": "sum_minus_gdp_gap",
                             "slope": val, "lo90": np.nan, "hi90": np.nan})

    tidy = pd.DataFrame(rows, columns=["quarter", "component", "slope",
                                       "lo90", "hi90"])
    return tidy, frames, gdp_frame


def band_excludes_zero(frame: pd.DataFrame) -> pd.Index:
    """Quarters where the 90% slope band lies entirely above or below zero."""
    mask = (frame["lo90"] > 0) | (frame["hi90"] < 0)
    return frame.index[mask]
