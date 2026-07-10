"""End-to-end pipeline: fetch -> contributions -> SUR system -> tables + figures.

Usage:
    python run_pipeline.py            # uses cached data if present
    python run_pipeline.py --refresh  # re-downloads from DBnomics
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd

from ptgdp import config, fetch, figures, import_content, model, prepare, sublayer


def main(refresh: bool = False, clv: pd.DataFrame | None = None,
         cp: pd.DataFrame | None = None, interactions: bool = False,
         import_content_arg=None, sublayer_flag: bool = False):
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if clv is None:
        clv, cp = fetch.load_all(refresh=refresh)
    elif cp is None:
        raise ValueError("pass the current-price (cp) frame alongside clv")
    print(f"Levels: {clv.shape[0]} quarters, {clv.index.min()} -> {clv.index.max()}")

    contrib, gdp_growth, comps = prepare.contributions(clv, cp, method="exact")
    labels = {k: v["label"] for k, v in comps.items()}
    print(f"Contributions: {contrib.shape[0]} quarters x {contrib.shape[1]} components")

    # ---- chain-linking convention comparison --------------------------
    conv = prepare.convention_comparison(clv, cp)
    conv.to_csv(config.OUTPUT_DIR / "convention_comparison.csv")
    print("\nChain-linking residual by convention (pp of quarterly growth):")
    for col in ["residual_exact", "residual_approx_raw", "residual_approx_reallocated"]:
        print(f"  {col:<28} mean={conv[col].mean():+.4f}  "
              f"max|.|={conv[col].abs().max():.4f}")

    X = prepare.design_matrix(contrib.index, interactions=interactions)
    result = model.fit(contrib, gdp_growth, X)
    print(f"SUR fit on n={result.nobs}; adding-up gap = {result.adding_up_gap:.2e}")

    if interactions:
        X_restricted = prepare.design_matrix(contrib.index, interactions=False)
        wald = model.slope_break_tests(contrib, gdp_growth, X_restricted, X)
        wald.to_csv(config.OUTPUT_DIR / "slope_break_tests.csv", index=False)
        print("\nRegime slope-break Wald tests (joint H0: interactions = 0, HAC):")
        wtbl = wald.copy()
        wtbl["component"] = [labels.get(c, c) for c in wtbl["component"]]
        print(wtbl.round(4).to_string(index=False))

    # ---- tables -------------------------------------------------------
    tbl = result.summary_table()
    tbl.to_csv(config.OUTPUT_DIR / "sur_coefficients.csv", index=False)

    rm = model.regime_means(contrib, gdp_growth, config.REGIMES)
    rm.to_csv(config.OUTPUT_DIR / "regime_means.csv")
    print("\nAverage contribution to quarterly GDP growth by regime (pp):")
    print(rm.round(3).to_string())

    print("\nTrend coefficients (pp of quarterly growth per decade, HAC p-values):")
    trend = pd.DataFrame({
        "coef": result.params["trend"],
        "p": result.pvalues["trend"],
    }).sort_values("coef")
    trend.index = [labels.get(i, i) for i in trend.index]
    print(trend.round(3).to_string())
    print(f"GDP total trend: {result.gdp_params['trend']:+.3f} "
          f"(p={result.gdp_pvalues['trend']:.3f})")

    # ---- figures ------------------------------------------------------
    fitted = (X @ result.params.T)[contrib.columns]
    figures.stacked_contributions(
        contrib, gdp_growth, labels, config.OUTPUT_DIR / "contributions_stacked.png"
    )
    figures.small_multiples(
        contrib, fitted, labels, config.OUTPUT_DIR / "contributions_small_multiples.png"
    )
    decomp_regressors = ["trend", *config.REGIMES]
    if interactions:
        decomp_regressors += [f"trend_{name}" for name in config.REGIMES]
    for reg in decomp_regressors:
        figures.coefficient_decomposition(
            result, reg, labels, config.OUTPUT_DIR / f"decomposition_{reg}.png"
        )

    # ---- import-content adjustment (optional) -------------------------
    if import_content_arg is not None:
        if isinstance(import_content_arg, str):
            shares = import_content.load_import_content(import_content_arg)
            print(f"\nImport-content shares loaded from {import_content_arg}")
        else:
            shares = dict(import_content.DEFAULT_IMPORT_CONTENT)
            print("\nImport-content shares: hardcoded defaults "
                  "(REPLACE_WITH_CURRENT_VINTAGE)")
        adjusted = import_content.adjusted_contributions(contrib, shares)
        adjusted.to_csv(config.OUTPUT_DIR / "adjusted_contributions.csv")
        figures.domestic_vs_external(
            adjusted, gdp_growth, config.OUTPUT_DIR / "domestic_vs_external.png"
        )
        gap = float((adjusted.sum(axis=1) - gdp_growth).abs().max())
        print(f"Adjusted contributions written; adding-up gap = {gap:.2e}")

    # ---- annual sub-component layer (optional) ------------------------
    if sublayer_flag:
        for parent, spec in config.SUBLAYERS.items():
            if parent not in contrib.columns:
                continue
            child_levels = fetch.fetch_annual(
                spec["dataset"], spec["items"], config.UNIT_CLV
            )
            tidy, residual = sublayer.within_component_decomposition(
                contrib[parent], child_levels
            )
            tidy.to_csv(config.OUTPUT_DIR / spec["csv"], index=False)
            fig_path = config.OUTPUT_DIR / spec["csv"].replace(".csv", ".png")
            figures.sublayer_stacked(tidy, spec["title"], fig_path)
            print(f"Sub-layer {parent}: {spec['csv']} written; "
                  f"max|reconciliation residual| = {residual.abs().max():.2e}")

    print(f"\nOutputs written to {config.OUTPUT_DIR}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-download from DBnomics")
    ap.add_argument("--interactions", action="store_true",
                    help="add regime-specific trend slopes and Wald slope-break tests")
    ap.add_argument("--import-content", nargs="?", const=True, default=None,
                    metavar="PATH", dest="import_content",
                    help="import-content-adjusted contributions; optional CSV path "
                         "overrides the default shares")
    ap.add_argument("--sublayer", action="store_true",
                    help="annual GFCF-by-asset and consumption-by-purpose breakdowns")
    args = ap.parse_args()
    main(refresh=args.refresh, interactions=args.interactions,
         import_content_arg=args.import_content, sublayer_flag=args.sublayer)
