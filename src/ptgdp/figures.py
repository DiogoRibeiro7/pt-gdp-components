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


def domestic_vs_external(adjusted: pd.DataFrame, gdp_growth: pd.Series, path,
                         annual: bool = True):
    """Stacked area of net domestic-demand vs net-export contributions.

    ``adjusted`` is the import-content-adjusted (net) contribution frame from
    :func:`import_content.adjusted_contributions`; export columns
    (``config.EXPORT_ITEMS``) are summed into net external demand and the
    remainder into net domestic demand. GDP growth is overlaid as a check that
    the two net blocks add back up.
    """
    export_cols = [c for c in config.EXPORT_ITEMS if c in adjusted.columns]
    domestic_cols = [c for c in adjusted.columns if c not in export_cols]
    external = adjusted[export_cols].sum(axis=1)
    domestic = adjusted[domestic_cols].sum(axis=1)

    if annual:
        external = external.groupby(external.index.year).sum()
        domestic = domestic.groupby(domestic.index.year).sum()
        g = gdp_growth.groupby(gdp_growth.index.year).sum()
        x = external.index.to_numpy()
        note = "annualised (sum of quarterly contributions)"
    else:
        g = gdp_growth
        x = np.arange(len(external))
        note = "quarterly"

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.stackplot(
        x, domestic.to_numpy(), external.to_numpy(),
        labels=["Net domestic demand", "Net external demand (exports)"],
        colors=["#1f6f8b", "#e07a5f"], alpha=0.9,
    )
    ax.plot(x, g.to_numpy(), color="black", lw=1.6, marker="o", ms=2.5,
            label="GDP growth")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel(f"pp of GDP growth, {note}")
    ax.set_title("Portugal - import-content-adjusted contributions: "
                 "domestic vs external demand")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    _credit(fig)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def markov_probabilities(probs: pd.DataFrame, gdp_growth: pd.Series,
                         regimes: dict, path):
    """Smoothed low-growth-regime probability with fixed regime windows shaded.

    Overlays the hand-drawn regime windows so agreement (and disagreement)
    between the data-dated recession probability and the fixed dummies is
    visible at a glance.
    """
    t = probs.index.to_timestamp()
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                  gridspec_kw={"height_ratios": [2, 1]})
    ax.fill_between(t, 0, probs["prob_low_growth"], color="#bc4749", alpha=0.5,
                    linewidth=0)
    ax.plot(t, probs["prob_low_growth"], color="#bc4749", lw=1.2)
    ax.axhline(0.5, color="black", lw=0.7, ls="--")
    for (start, end) in regimes.values():
        ax.axvspan(pd.Period(start, "Q").to_timestamp(),
                   pd.Period(end, "Q").to_timestamp(),
                   color="#999999", alpha=0.2, linewidth=0)
    ax.set_ylabel("P(low-growth regime)")
    ax.set_ylim(0, 1)
    ax.set_title("Portugal - smoothed probability of the low-growth regime "
                 "(fixed regime windows shaded)")

    g = gdp_growth.loc[probs.index]
    ax2.plot(t, g.to_numpy(), color="#1f6f8b", lw=1.0)
    ax2.axhline(0, color="black", lw=0.7)
    ax2.set_ylabel("GDP growth (%)")
    _credit(fig)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def slope_paths(frames: dict, gdp_frame, labels: dict[str, str], regimes: dict,
                path):
    """Small multiples of smoothed state-space slope paths with 90% bands.

    Each panel shades the regime windows, marks the zero line, and draws the
    slope with its band; the GDP-growth path is included as a final panel.
    """
    items = list(frames.items())
    if gdp_frame is not None:
        items.append(("GDP (system sum)", gdp_frame))
    n = len(items)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 2.4 * nrows), sharex=True)
    axes = np.atleast_2d(axes)

    for k, (name, fr) in enumerate(items):
        ax = axes[k // ncols, k % ncols]
        t = fr.index.to_timestamp()
        ax.fill_between(t, fr["lo90"], fr["hi90"], color=PALETTE[k % len(PALETTE)],
                        alpha=0.25, linewidth=0)
        ax.plot(t, fr["slope"], lw=1.4, color=PALETTE[k % len(PALETTE)])
        ax.axhline(0, color="black", lw=0.6)
        for (start, end) in regimes.values():
            ax.axvspan(pd.Period(start, "Q").to_timestamp(),
                       pd.Period(end, "Q").to_timestamp(),
                       color="#999999", alpha=0.15, linewidth=0)
        ax.set_title(labels.get(name, name), fontsize=8.5)
    for k in range(n, nrows * ncols):
        axes[k // ncols, k % ncols].axis("off")

    fig.suptitle("Smoothed local-linear-trend slope paths with 90% bands "
                 "(regime windows shaded)", fontsize=10, y=1.0)
    _credit(fig)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def residual_panel(residuals: pd.DataFrame, labels: dict[str, str], path,
                   window: int = 12):
    """Small multiples of residual series with ±2 rolling-σ bands.

    ``residuals`` has one column per equation; the rolling standard deviation
    (default 12-quarter window) traces time-varying dispersion so
    heteroskedasticity and outliers read off the panel directly.
    """
    cols = list(residuals.columns)
    n = len(cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 2.4 * nrows), sharex=True)
    axes = np.atleast_2d(axes)
    t = residuals.index.to_timestamp() if hasattr(residuals.index, "to_timestamp") \
        else np.arange(len(residuals))

    for k, col in enumerate(cols):
        ax = axes[k // ncols, k % ncols]
        r = residuals[col]
        band = 2 * r.rolling(window, min_periods=window // 2).std()
        ax.plot(t, r, lw=0.8, color=PALETTE[k % len(PALETTE)])
        ax.plot(t, band, lw=0.7, color="#888888", ls="--")
        ax.plot(t, -band, lw=0.7, color="#888888", ls="--")
        ax.axhline(0, color="black", lw=0.6)
        ax.set_title(labels.get(col, col), fontsize=8.5)
    for k in range(n, nrows * ncols):
        axes[k // ncols, k % ncols].axis("off")

    fig.suptitle("Equation residuals with ±2 rolling-σ bands "
                 f"({window}-quarter window)", fontsize=10, y=1.0)
    _credit(fig)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def sublayer_stacked(tidy: pd.DataFrame, title: str, path):
    """Stacked annual bars of a parent contribution split by child.

    ``tidy`` is the frame from
    :func:`sublayer.within_component_decomposition` (year, child,
    contribution_pp, share_of_parent).
    """
    wide = tidy.pivot(index="year", columns="child", values="contribution_pp")
    x = wide.index.to_numpy()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    pos_bottom = np.zeros(len(wide))
    neg_bottom = np.zeros(len(wide))
    for i, col in enumerate(wide.columns):
        vals = wide[col].to_numpy()
        bottom = np.where(vals >= 0, pos_bottom, neg_bottom)
        ax.bar(x, vals, bottom=bottom, width=0.8,
               color=PALETTE[i % len(PALETTE)], label=col, linewidth=0)
        pos_bottom += np.clip(vals, 0, None)
        neg_bottom += np.clip(vals, None, 0)
    ax.plot(x, wide.sum(axis=1).to_numpy(), color="black", lw=1.6, marker="o",
            ms=3, label="parent total")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("pp of GDP growth, annual (sum of quarterly contributions)")
    ax.set_title(title)
    ax.legend(loc="lower left", fontsize=7, ncol=2, framealpha=0.9)
    _credit(fig)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def vecm_alpha_heatmap(alpha_selected: pd.DataFrame, labels: dict[str, str], path):
    """Heatmap of the VECM alpha (loading) matrix with significance stars.

    ``alpha_selected`` is the tidy alpha table for the selected rank
    (component, relation, alpha, pvalue). Stars: * p<0.1, ** p<0.05, *** p<0.01.
    """
    mat = alpha_selected.pivot(index="component", columns="relation", values="alpha")
    pmat = alpha_selected.pivot(index="component", columns="relation", values="pvalue")
    rows = list(mat.index)
    cols = list(mat.columns)

    fig, ax = plt.subplots(figsize=(1.6 * len(cols) + 3, 0.5 * len(rows) + 1.5))
    vmax = np.nanmax(np.abs(mat.to_numpy())) or 1.0
    im = ax.imshow(mat.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    def _stars(p):
        if not np.isfinite(p):
            return ""
        return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""

    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            val = mat.loc[r, c]
            ax.text(j, i, f"{val:+.2f}{_stars(pmat.loc[r, c])}",
                    ha="center", va="center", fontsize=7.5, color="black")
    ax.set_xticks(range(len(cols)), cols, fontsize=8)
    ax.set_yticks(range(len(rows)), [labels.get(r, r) for r in rows], fontsize=8)
    ax.set_title("VECM adjustment loadings (alpha) — who corrects the disequilibrium")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="alpha")
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
