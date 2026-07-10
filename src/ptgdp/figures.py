"""Publication figures for the contribution decomposition."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
    }
)

PALETTE = [
    "#1f6f8b", "#e07a5f", "#3d405b", "#81b29a", "#f2cc8f",
    "#9a8c98", "#c9ada7", "#4a5759", "#bc4749", "#6a994e",
]


def _credit(fig):
    fig.text(0.01, 0.005, config.FIGURE_CREDIT, fontsize=6.5, color="#666666")


def stacked_contributions(contrib: pd.DataFrame, gdp_growth: pd.Series,
                          labels: dict[str, str], path, annual: bool = True):
    """Stacked bars of contributions with the GDP growth line on top."""
    if annual:
        c = contrib.groupby(contrib.index.year).sum()
        g = gdp_growth.groupby(gdp_growth.index.year).sum()
        x = c.index.to_numpy()
        xlabel_note = "annualised (sum of quarterly contributions)"
    else:
        c, g = contrib, gdp_growth
        x = np.arange(len(c))
        xlabel_note = "quarterly"

    fig, ax = plt.subplots(figsize=(11, 5.5))
    pos_bottom = np.zeros(len(c))
    neg_bottom = np.zeros(len(c))
    for i, col in enumerate(c.columns):
        vals = c[col].to_numpy()
        bottom = np.where(vals >= 0, pos_bottom, neg_bottom)
        ax.bar(x, vals, bottom=bottom, width=0.8,
               color=PALETTE[i % len(PALETTE)], label=labels.get(col, col),
               linewidth=0)
        pos_bottom += np.clip(vals, 0, None)
        neg_bottom += np.clip(vals, None, 0)

    ax.plot(x, g.to_numpy(), color="black", lw=1.6, marker="o", ms=2.5,
            label="GDP growth")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel(f"pp of GDP growth, {xlabel_note}")
    ax.set_title("Portugal - contributions to GDP growth by expenditure component")
    ax.legend(loc="lower left", fontsize=7, ncol=2, framealpha=0.9)
    _credit(fig)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def small_multiples(contrib: pd.DataFrame, fitted: pd.DataFrame,
                    labels: dict[str, str], path):
    """Each component's contribution with the fitted (trend+regime) path."""
    cols = list(contrib.columns)
    n = len(cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 2.4 * nrows),
                             sharex=True)
    axes = np.atleast_2d(axes)
    t = contrib.index.to_timestamp()

    for k, col in enumerate(cols):
        ax = axes[k // ncols, k % ncols]
        ax.plot(t, contrib[col], lw=0.7, color="#888888", alpha=0.8)
        ax.plot(t, fitted[col], lw=1.8, color=PALETTE[k % len(PALETTE)])
        ax.axhline(0, color="black", lw=0.6)
        ax.set_title(labels.get(col, col), fontsize=8.5)
    for k in range(n, nrows * ncols):
        axes[k // ncols, k % ncols].axis("off")

    fig.suptitle("Contributions to quarterly GDP growth - actual (grey) and "
                 "fitted trend + regime path (colour)", fontsize=10, y=1.0)
    _credit(fig)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def coefficient_decomposition(result, regressor: str, labels: dict[str, str], path):
    """How the GDP-level coefficient on `regressor` splits across components."""
    coefs = result.params[regressor].sort_values()
    ses = result.bse[regressor].loc[coefs.index]
    names = [labels.get(c, c) for c in coefs.index]

    fig, ax = plt.subplots(figsize=(8, 0.45 * len(coefs) + 1.5))
    y = np.arange(len(coefs))
    colors = ["#bc4749" if v < 0 else "#1f6f8b" for v in coefs]
    ax.barh(y, coefs, xerr=1.96 * ses, color=colors, height=0.6,
            error_kw={"lw": 0.8, "capsize": 2})
    ax.set_yticks(y, names, fontsize=8)
    ax.axvline(0, color="black", lw=0.8)
    total = result.gdp_params[regressor]
    ax.axvline(total, color="#3d405b", lw=1.2, ls="--")
    ax.annotate(f"GDP total: {total:+.3f}", xy=(total, len(coefs) - 0.5),
                fontsize=8, color="#3d405b",
                xytext=(5, 0), textcoords="offset points")
    ax.set_xlabel(f"coefficient on '{regressor}' (pp of quarterly growth), +/-1.96 HAC SE")
    ax.set_title(f"Decomposition of the GDP-level '{regressor}' effect across components")
    _credit(fig)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
