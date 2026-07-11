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

from ptgdp import (backtest, config, diagnostics, fetch, figures, import_content,
                   model, msm, prepare, stsm, sublayer, vecm)


def main(refresh: bool = False, clv: pd.DataFrame | None = None,
         cp: pd.DataFrame | None = None, interactions: bool = False,
         import_content_arg=None, sublayer_flag: bool = False,
         stsm_flag: bool = False, stsm_seasonal: bool = False,
         vecm_flag: bool = False, backtest_flag: bool = False,
         msm_flag: bool = False):
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

    # ---- residual diagnostics battery (unconditional) -----------------
    resid_frame = result.resid.copy()
    resid_frame["GDP (system sum)"] = result.gdp_resid
    diag = diagnostics.diagnostics_battery(
        {name: resid_frame[name] for name in resid_frame.columns}
    )
    diag.to_csv(config.OUTPUT_DIR / "diagnostics.csv", index=False)
    figures.residual_panel(
        resid_frame, {**labels, "GDP (system sum)": "GDP (system sum)"},
        config.OUTPUT_DIR / "residual_panel.png"
    )
    flagged = diag[diag["ljung_box_p_lag4"] < 0.05]["equation"].tolist()
    print("\nResidual diagnostics (p-values) written to diagnostics.csv.")
    if flagged:
        pretty = ", ".join(labels.get(e, e) for e in flagged)
        print(f"Ljung-Box p<0.05 at lag 4 for: {pretty}.")
        print("  Implication: HAC (Newey-West) SEs remain valid for inference on "
              "the mean parameters,\n  but the static mean model is dynamically "
              "incomplete - this motivates the state-space\n  trend layer (--stsm, "
              "Lane B3).")
    else:
        print("No equation shows Ljung-Box p<0.05 at lag 4.")

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
            try:
                child_levels = fetch.fetch_annual(
                    spec["dataset"], spec["items"], config.UNIT_CLV
                )
            except Exception as exc:  # noqa: BLE001 - annual fetch is best-effort
                print(f"Sub-layer {parent}: skipped ({spec['dataset']} fetch "
                      f"failed: {exc})")
                continue
            tidy, residual = sublayer.within_component_decomposition(
                contrib[parent], child_levels
            )
            tidy.to_csv(config.OUTPUT_DIR / spec["csv"], index=False)
            fig_path = config.OUTPUT_DIR / spec["csv"].replace(".csv", ".png")
            figures.sublayer_stacked(tidy, spec["title"], fig_path)
            print(f"Sub-layer {parent}: {spec['csv']} written; "
                  f"max|reconciliation residual| = {residual.abs().max():.2e}")

    # ---- state-space time-varying slopes (optional) ------------------
    if stsm_flag:
        tidy, frames, gdp_frame = stsm.slope_paths(
            contrib, gdp_growth, seasonal=stsm_seasonal
        )
        tidy.to_csv(config.OUTPUT_DIR / "stsm_slopes.csv", index=False)
        figures.slope_paths(
            frames, gdp_frame, labels, config.REGIMES,
            config.OUTPUT_DIR / "slope_paths.png"
        )
        print("\nState-space slope paths (quarters where the 90% band excludes zero):")
        for comp, fr in frames.items():
            qs = stsm.band_excludes_zero(fr)
            if len(qs):
                span = f"{qs.min()}..{qs.max()} ({len(qs)} quarters)"
            else:
                span = "never"
            print(f"  {labels.get(comp, comp):<32} {span}")
        if gdp_frame is not None:
            qs = stsm.band_excludes_zero(gdp_frame)
            print(f"  {'GDP (system sum)':<32} "
                  f"{f'{qs.min()}..{qs.max()} ({len(qs)} quarters)' if len(qs) else 'never'}")

    # ---- VECM long-run structure (optional) --------------------------
    if vecm_flag:
        joh_df, alpha_all, alpha_selected, k_ar_diff, rank = vecm.run(clv)
        joh_df.to_csv(config.OUTPUT_DIR / "vecm_johansen.csv", index=False)
        alpha_all.to_csv(config.OUTPUT_DIR / "vecm_alpha.csv", index=False)
        print(f"\nVECM: k_ar_diff={k_ar_diff} (BIC), Johansen 5%-selected rank={rank} "
              f"(descriptive; size-distorted at this sample).")
        print(joh_df.round(3).to_string(index=False))
        if alpha_selected is not None:
            figures.vecm_alpha_heatmap(
                alpha_selected, labels, config.OUTPUT_DIR / "vecm_alpha_heatmap.png"
            )

    # ---- pseudo-out-of-sample backtest (optional) --------------------
    if backtest_flag:
        bt, _fc = backtest.backtest(contrib, gdp_growth, X, start="2010Q1")
        bt.to_csv(config.OUTPUT_DIR / "backtest.csv", index=False)
        print("\nPseudo-out-of-sample backtest (one-quarter-ahead GDP growth):")
        print(bt.round(4).to_string(index=False))
        print("Diebold-Mariano vs AR(1): positive stat => model loses to the "
              "benchmark; a decomposition\nthat cannot beat AR(1) is a "
              "specification check passed, not a forecasting win.")

    # ---- Markov-switching endogenous regimes (optional) --------------
    if msm_flag:
        probs, msm_summary = msm.fit_markov(gdp_growth, k_regimes=2, label="GDP")
        if probs is not None:
            out = probs.copy()
            out.index = out.index.astype(str)
            out.to_csv(config.OUTPUT_DIR / "msm_probabilities.csv")
            msm_summary.to_csv(config.OUTPUT_DIR / "msm_regimes.csv", index=False)
            figures.markov_probabilities(
                probs, gdp_growth, config.REGIMES,
                config.OUTPUT_DIR / "msm_probabilities.png"
            )
            hi = probs["prob_low_growth"] > 0.5
            n_rec = int(hi.sum())
            print(f"\nMarkov-switching: {n_rec} quarters with P(low-growth) > 0.5.")
            low = msm_summary[msm_summary["is_low_growth"]].iloc[0]
            hi_r = msm_summary[~msm_summary["is_low_growth"]].iloc[0]
            print(f"  low-growth regime mean={low['mean_growth']:.3f}%, "
                  f"expected duration {low['expected_duration_q']:.1f}q; "
                  f"high-growth mean={hi_r['mean_growth']:.3f}%, "
                  f"expected duration {hi_r['expected_duration_q']:.1f}q.")

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
    ap.add_argument("--stsm", action="store_true",
                    help="local-linear-trend state-space slope paths")
    ap.add_argument("--stsm-seasonal", action="store_true", dest="stsm_seasonal",
                    help="add a stochastic seasonal(4) term to the state-space models")
    ap.add_argument("--vecm", action="store_true",
                    help="Johansen + VECM on log CLV component levels")
    ap.add_argument("--backtest", action="store_true",
                    help="expanding-window one-step-ahead GDP-growth backtest")
    ap.add_argument("--msm", action="store_true",
                    help="Markov-switching endogenous regime dating on GDP growth")
    args = ap.parse_args()
    main(refresh=args.refresh, interactions=args.interactions,
         import_content_arg=args.import_content, sublayer_flag=args.sublayer,
         stsm_flag=args.stsm, stsm_seasonal=args.stsm_seasonal,
         vecm_flag=args.vecm, backtest_flag=args.backtest, msm_flag=args.msm)
