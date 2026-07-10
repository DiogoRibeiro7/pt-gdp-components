"""Pseudo-out-of-sample evaluation of the decomposition against a univariate benchmark.

Expanding-window, one-quarter-ahead forecasts of GDP growth from 2010Q1 onward
under three models:

* (a) the SUR mean model — the static regressors (intercept, trend, regime
  dummies) are known one step ahead, so the forecast is simply the fitted mean
  evaluated at the target quarter's known design row;
* (b) per-component AR(1) on the contributions, summed to a GDP forecast;
* (c) a direct AR(1) on GDP growth, the benchmark.

Purpose
-------
This is a specification check, not a forecasting exercise. The question is
whether resolving GDP growth into a modelled decomposition buys anything a
one-line univariate AR(1) does not. If the decomposition-based models (a) and
(b) cannot beat the AR(1) benchmark out of sample, the honest report says so
plainly — the decomposition earns its keep as an accounting and inference
device, not as a predictor.

No look-ahead: each forecast is trained strictly on quarters before the target,
asserted in the loop. The pandemic quarters (2020Q1–2021Q4) are set aside in a
second set of error metrics because a mean/AR model cannot be expected to track
that shock and it would otherwise dominate the RMSE.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tsa.ar_model import AutoReg

_PANDEMIC = (pd.Period("2020Q1", "Q"), pd.Period("2021Q4", "Q"))


def _ar1_onestep(values: np.ndarray) -> float:
    """One-step-ahead AR(1) forecast from a training array."""
    res = AutoReg(np.asarray(values, dtype=float), lags=1, old_names=False).fit()
    n = len(values)
    return float(res.predict(start=n, end=n)[0])


def _metrics(err: np.ndarray) -> tuple[float, float]:
    err = np.asarray(err, dtype=float)
    return float(np.sqrt(np.mean(err ** 2))), float(np.mean(np.abs(err)))


def dm_test(err_model: np.ndarray, err_bench: np.ndarray, h: int = 1
            ) -> tuple[float, float]:
    """Diebold-Mariano test (squared-error loss, HAC, horizon h).

    Positive statistic ⇒ the model has larger loss than the benchmark. Returns
    (statistic, two-sided p-value). With h=1 the long-run variance uses zero
    lags (the loss differential is serially uncorrelated at horizon one).
    """
    d = np.asarray(err_model, dtype=float) ** 2 - np.asarray(err_bench, dtype=float) ** 2
    n = len(d)
    dbar = d.mean()
    lag = max(h - 1, 0)
    var = np.mean((d - dbar) ** 2)
    for k in range(1, lag + 1):
        cov = np.mean((d[k:] - dbar) * (d[:-k] - dbar))
        var += 2.0 * (1.0 - k / (lag + 1)) * cov
    if var <= 0 or n == 0:
        return float("nan"), float("nan")
    stat = dbar / np.sqrt(var / n)
    p = 2.0 * (1.0 - norm.cdf(abs(stat)))
    return float(stat), float(p)


def backtest(contrib: pd.DataFrame, gdp_growth: pd.Series, X: pd.DataFrame,
             start: str = "2010Q1") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the expanding-window backtest; return (metrics, per-quarter forecasts)."""
    idx = gdp_growth.index
    targets = [q for q in idx if q >= pd.Period(start, "Q")]
    recs = []
    for t in targets:
        mask = idx < t
        assert idx[mask].max() < t, "look-ahead: training window not strictly before target"
        Xtr, ytr = X.loc[mask], gdp_growth.loc[mask]
        beta, *_ = np.linalg.lstsq(Xtr.to_numpy(), ytr.to_numpy(), rcond=None)
        sur_fc = float(X.loc[t].to_numpy() @ beta)
        comp_fc = float(sum(_ar1_onestep(contrib[c].loc[mask].to_numpy())
                            for c in contrib.columns))
        ar1_fc = _ar1_onestep(gdp_growth.loc[mask].to_numpy())
        recs.append({"quarter": t, "actual": float(gdp_growth.loc[t]),
                     "sur": sur_fc, "ar1_components": comp_fc, "ar1_gdp": ar1_fc})

    fc = pd.DataFrame(recs).set_index("quarter")
    models = {"sur": "SUR mean model", "ar1_components": "AR(1) components (summed)",
              "ar1_gdp": "AR(1) on GDP growth (benchmark)"}
    ex = np.asarray(~((fc.index >= _PANDEMIC[0]) & (fc.index <= _PANDEMIC[1])))
    err_bench = (fc["actual"] - fc["ar1_gdp"]).to_numpy()

    rows = []
    for key, name in models.items():
        err = (fc["actual"] - fc[key]).to_numpy()
        rmse_f, mae_f = _metrics(err)
        rmse_x, mae_x = _metrics(err[ex])
        if key == "ar1_gdp":
            dm_stat, dm_p = float("nan"), float("nan")
        else:
            dm_stat, dm_p = dm_test(err, err_bench, h=1)
        rows.append({
            "model": name, "rmse_full": rmse_f, "mae_full": mae_f,
            "rmse_ex_pandemic": rmse_x, "mae_ex_pandemic": mae_x,
            "dm_stat_vs_ar1": dm_stat, "dm_pvalue_vs_ar1": dm_p,
        })
    return pd.DataFrame(rows), fc
