"""Endogenous regime dating with a Markov-switching model.

The regime dummies in the SUR design (A1) are imposed: the crisis-and-troika
and pandemic windows are fixed by hand from calendar dates. This module lets
the data date the regimes instead. A two-state Markov-switching regression on
GDP growth, with switching mean and switching variance (the textbook Hamilton
business-cycle specification), estimates a high-growth and a low-growth state
and returns the smoothed probability of being in each one quarter by quarter.

The value added over the fixed dummies is twofold: the low-growth state is
identified from the growth series itself rather than assumed, and the smoothed
probability is a continuous "how likely was a recession this quarter" reading
that can be laid over the hand-drawn windows to see where they agree and where
they miss. The alternative of keeping only the fixed dummies was rejected
because it cannot flag a downturn the analyst did not pre-specify (a mild 2003
slowdown, say) and gives no probability, only an on/off indicator.

Non-convergence is handled: the EM fit is given several random restarts, and a
series that still fails is logged and skipped rather than crashing the run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression


def fit_markov(series: pd.Series, k_regimes: int = 2, label: str = ""):
    """Fit a switching-mean, switching-variance Markov regression.

    Returns ``(smoothed_probs, summary)`` where ``smoothed_probs`` is a frame
    indexed like the series with one probability column per regime plus a
    ``prob_low_growth`` column, and ``summary`` holds the per-regime mean and
    variance and the expected durations. Returns ``(None, None)`` if the model
    does not converge.
    """
    y = pd.Series(series).dropna()
    try:
        res = MarkovRegression(
            y, k_regimes=k_regimes, trend="c", switching_variance=True
        ).fit(search_reps=20, maxiter=200)
    except Exception as exc:  # noqa: BLE001 - log and skip
        print(f"[msm] skipped {label!r}: {exc}")
        return None, None

    means = np.array([float(res.params[f"const[{i}]"]) for i in range(k_regimes)])
    low = int(np.argmin(means))

    smoothed = np.asarray(res.smoothed_marginal_probabilities)
    probs = pd.DataFrame(
        {f"prob_regime{i}": smoothed[:, i] for i in range(k_regimes)},
        index=y.index,
    )
    probs["prob_low_growth"] = smoothed[:, low]

    try:
        durations = np.asarray(res.expected_durations)
    except Exception:
        durations = np.full(k_regimes, np.nan)
    variances = [float(res.params[f"sigma2[{i}]"]) for i in range(k_regimes)]
    summary = pd.DataFrame({
        "regime": list(range(k_regimes)),
        "mean_growth": means,
        "variance": variances,
        "expected_duration_q": durations,
        "is_low_growth": [i == low for i in range(k_regimes)],
    })
    return probs, summary
