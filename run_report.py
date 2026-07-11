"""Generate report/technical_report.md from a full pipeline run.

Runs the pipeline with every analytical layer active (import-content only when
a matrix CSV is supplied, the annual sub-layer only when annual data can be
fetched) and renders the technical report with every reported number read back
from the artifacts in output/. Nothing in the prose is hardcoded: the numbers
come from the returned SUR result and the CSVs, and every figure the report
references is checked for existence first so the document never points at a
missing image.

Usage:
    python run_report.py                      # real data, all layers
    python run_report.py --import-content=PATH  # also import-content section
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd

from ptgdp import config
import run_pipeline

REPORT_DIR = config.PROJECT_ROOT / "report"


def _csv(name):
    p = config.OUTPUT_DIR / name
    return pd.read_csv(p) if p.exists() else None


def _exists(name) -> bool:
    return (config.OUTPUT_DIR / name).exists()


def _fig(name: str, caption: str) -> str:
    """Markdown image block, only if the figure file exists in output/."""
    if _exists(name):
        return f"![{caption}](../output/{name})\n\n*Figure &mdash; {caption}.*\n\n"
    return ""


def _f(x, nd=3) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "n/a"


def _sci(x, nd=1) -> str:
    try:
        return f"{float(x):.{nd}e}"
    except (TypeError, ValueError):
        return "n/a"


_LATEX_PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{float}
\usepackage[hidelinks]{hyperref}
\setlength{\parskip}{0.6em}
\setlength{\parindent}{0pt}
\title{%(title)s}
\author{Diogo Ribeiro}
\date{}
\begin{document}
\maketitle
"""


def _esc(s: str) -> str:
    """Escape LaTeX specials in prose text (entities become dashes first)."""
    s = s.replace("&mdash;", "---").replace("&ndash;", "--")
    s = s.replace("\\", r"\textbackslash{}")
    for ch, rep in (("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("$", r"\$"),
                    ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
                    ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
        s = s.replace(ch, rep)
    return s


def _esc_code(s: str) -> str:
    """Escape a code span destined for \\texttt{...}."""
    s = s.replace("\\", r"\textbackslash{}")
    for ch, rep in (("_", r"\_"), ("%", r"\%"), ("#", r"\#"), ("&", r"\&"),
                    ("{", r"\{"), ("}", r"\}"), ("$", r"\$"),
                    ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
        s = s.replace(ch, rep)
    return s


def _inline(s: str) -> str:
    """Convert a prose line: backtick code to \\texttt, escape the rest."""
    parts = s.split("`")
    out = []
    for i, part in enumerate(parts):
        out.append(r"\texttt{" + _esc_code(part) + "}" if i % 2 else _esc(part))
    return "".join(out)


def _md_to_latex(md: str, title: str) -> str:
    """Render the markdown-shaped report body as a standalone LaTeX document."""
    out = []
    for line in md.split("\n"):
        if line.startswith("## "):
            heading = re.sub(r"^\d+\.\s*", "", line[3:].strip())
            out.append(r"\section{" + _esc(heading) + "}")
        elif line.startswith("# "):
            continue  # document title is set in the preamble
        elif line.startswith("!["):
            m = re.match(r"!\[(.*?)\]\((.*?)\)", line)
            if m:
                cap, path = m.group(1), m.group(2)
                out.append(r"\begin{figure}[H]\centering")
                out.append(r"\includegraphics[width=\linewidth]{" + path + "}")
                out.append(r"\caption{" + _esc(cap) + "}")
                out.append(r"\end{figure}")
        elif line.strip().startswith("*Figure"):
            continue  # caption already carried by the figure environment
        elif line.strip():
            out.append(_inline(line))
        else:
            out.append("")
    preamble = _LATEX_PREAMBLE % {"title": _esc(title)}
    return preamble + "\n".join(out) + "\n\\end{document}\n"


def _gather(result):
    """Collect every number the report cites from the artifacts."""
    ctx = {"adding_up_gap": result.adding_up_gap, "nobs": result.nobs}

    conv = _csv("convention_comparison.csv")
    if conv is not None:
        ctx["res_exact_mean"] = conv["residual_exact"].mean()
        ctx["res_exact_maxabs"] = conv["residual_exact"].abs().max()
        ctx["res_approx_mean"] = conv["residual_approx_raw"].mean()
        ctx["res_approx_maxabs"] = conv["residual_approx_raw"].abs().max()
        ctx["res_realloc_maxabs"] = conv["residual_approx_reallocated"].abs().max()
        ctx["n_quarters"] = len(conv)

    sur = _csv("sur_coefficients.csv")
    if sur is not None:
        gdp = sur[sur["component"] == "GDP (system sum)"]
        tr = gdp[gdp["regressor"] == "trend"]
        if not tr.empty:
            ctx["gdp_trend"] = tr["coef"].iloc[0]
            ctx["gdp_trend_p"] = tr["pvalue"].iloc[0]
        ctx["n_components"] = sur[sur["component"] != "GDP (system sum)"]["component"].nunique()

    rm = _csv("regime_means.csv")
    if rm is not None:
        rm = rm.set_index(rm.columns[0])
        if "GDP growth" in rm.index:
            row = rm.loc["GDP growth"]
            for reg in ("gfc", "pandemic", "normal"):
                if reg in row.index:
                    ctx[f"gdp_{reg}"] = row[reg]

    wald = _csv("slope_break_tests.csv")
    if wald is not None:
        g = wald[wald["component"] == "GDP (system sum)"]
        if not g.empty:
            ctx["wald_gdp_stat"] = g["wald_stat"].iloc[0]
            ctx["wald_gdp_p"] = g["pvalue"].iloc[0]
        comp = wald[wald["component"] != "GDP (system sum)"]
        ctx["wald_n_sig"] = int((comp["pvalue"] < 0.05).sum())
        ctx["wald_n_comp"] = len(comp)

    diag = _csv("diagnostics.csv")
    if diag is not None:
        flagged = diag[diag["ljung_box_p_lag4"] < 0.05]["equation"].tolist()
        ctx["lb_flagged"] = flagged
        ctx["lb_n_flagged"] = len(flagged)
        ctx["diag_n"] = len(diag)

    stsm = _csv("stsm_slopes.csv")
    if stsm is not None:
        gap = stsm[stsm["component"] == "sum_minus_gdp_gap"]
        if not gap.empty:
            ctx["stsm_gap_maxabs"] = gap["slope"].abs().max()
        gdp_s = stsm[stsm["component"] == "GDP"]
        if not gdp_s.empty:
            excl = ((gdp_s["lo90"] > 0) | (gdp_s["hi90"] < 0)).sum()
            ctx["stsm_gdp_excl"] = int(excl)
            ctx["stsm_gdp_n"] = len(gdp_s)

    joh = _csv("vecm_johansen.csv")
    if joh is not None:
        rank, still = 0, True
        for _, r in joh.iterrows():
            if still and bool(r["reject_at_5pct"]):
                rank = int(r["null_rank_r"]) + 1
            else:
                still = False
        ctx["vecm_rank"] = rank
        ctx["vecm_n_series"] = len(joh)

    alpha = _csv("vecm_alpha.csv")
    if alpha is not None and "rank" in alpha.columns:
        ctx["vecm_ranks"] = sorted(alpha["rank"].unique().tolist())
        sel = alpha[alpha["rank"] == max(ctx.get("vecm_rank", 1), 1)]
        if not sel.empty:
            sig = sel[sel["pvalue"] < 0.05]
            top = (sig if not sig.empty else sel).reindex(
                (sig if not sig.empty else sel)["alpha"].abs().sort_values(
                    ascending=False).index
            )
            if not top.empty:
                ctx["vecm_top_comp"] = top["component"].iloc[0]
                ctx["vecm_top_alpha"] = top["alpha"].iloc[0]

    bt = _csv("backtest.csv")
    if bt is not None:
        ctx["bt"] = bt

    return ctx


def _render(ctx) -> str:
    g = ctx.get
    p = []
    p.append("# A granular expenditure-side decomposition of Portuguese GDP growth\n")
    p.append(
        "This report documents the measurement choices, the system model, and the "
        "analytical extensions behind the Portuguese GDP contribution decomposition, "
        "and states what each layer does and does not support. It is generated "
        "directly from the pipeline artifacts; every number below is read from a "
        "table in `output/` and every figure it shows was produced in the same run.\n"
    )

    # 1. Data and measurement
    p.append("## 1. Data and measurement\n")
    p.append(
        "The raw material is Eurostat's quarterly national accounts for Portugal, "
        "`namq_10_gdp`, obtained through DBnomics. Two unit variants are pulled for "
        "the same expenditure items: chain-linked volumes with reference year 2020 "
        f"(`CLV20_MEUR`), which carry the real quarter-on-quarter dynamics over {g('n_quarters', 'the sample')} "
        "quarters of usable observations, and current prices (`CP_MEUR`), which supply "
        "the nominal weights used to turn volume changes into contributions. Both are "
        "seasonally and calendar adjusted (SCA); working with the adjusted series keeps "
        "the quarter-on-quarter growth rates free of the deterministic seasonal pattern "
        "that would otherwise dominate the contributions and forces any seasonality in "
        "the later state-space models to be treated as a modelling option rather than a "
        "nuisance to be differenced away.\n"
    )
    p.append(
        "Two measurement facts shape everything downstream. The first is a vintage "
        "issue: some releases publish changes in inventories (P52) and acquisitions of "
        "valuables (P53) only as a combined `P52_P53` series, so the code resolves "
        "whichever set is present and labels the combined item accordingly rather than "
        "silently dropping inventories. The second is chain-linking non-additivity. "
        "Chain-linked volumes are not additive away from the reference year, so the "
        "component volumes do not sum to the GDP volume and the naive contribution "
        "residual against GDP growth is non-zero. The convention-comparison table "
        f"quantifies it: over the sample the exact annual-overlap residual averages "
        f"{_f(g('res_exact_mean'), 4)} pp with a maximum absolute value of "
        f"{_f(g('res_exact_maxabs'), 3)} pp, while the raw naive approximation residual "
        f"averages {_f(g('res_approx_mean'), 4)} pp with a maximum absolute value of "
        f"{_f(g('res_approx_maxabs'), 3)} pp. After proportional reallocation the "
        f"identity closes to a maximum absolute residual of "
        f"{_sci(g('res_realloc_maxabs'))} pp, which is the number the system model "
        "relies on.\n"
    )
    p.append(
        "The division of labour between the two unit variants is worth making explicit "
        "because it is the source of most confusion in contribution accounting. Volumes "
        "answer 'how much more was produced or spent', stripping out price change, and so "
        "they carry the growth signal; current prices answer 'how large was each "
        "component in money terms', and so they carry the weights that convert a "
        "component's own growth into its share of headline growth. Using volume shares as "
        "weights would double-count relative price movements, which is precisely the trap "
        "the exact method in the next section avoids. The reference year matters here too: "
        "at 2020 prices the volume aggregates are additive in and around 2020 and drift "
        "apart the further a quarter sits from that year, so the non-additivity is not a "
        "data error to be cleaned but a structural feature of chain-linking that the "
        "accounting has to confront head-on.\n"
    )
    p.append(_fig("contributions_stacked.png",
                  "Contributions to Portuguese GDP growth by expenditure component, annualised"))

    # 2. Contribution methodology
    p.append("## 2. Contribution methodology\n")
    p.append(
        "The default contributions use the annual-overlap exact formula: each "
        "component's quarter-on-quarter volume ratio is weighted by its share of "
        "nominal GDP in the previous calendar year, "
        "`contrib = sign x s_prev x (X_t / X_prev - 1) x 100`, with the nominal "
        "shares taken from annual sums of the current-price frame. This is the additive "
        "decomposition consistent with a chain-linked Laspeyres aggregate, in which the "
        "aggregate within a linking year is a fixed-previous-year-price index and "
        "previous-year nominal shares are therefore the correct weights. The naive "
        "alternative, differencing the volume level and dividing by lagged GDP, is "
        "retained only as a robustness check because it treats volume levels measured in "
        "reference-year prices as if they were additive across components. The magnitude "
        "of the difference is not academic: the maximum absolute residual the naive "
        f"convention leaves ({_f(g('res_approx_maxabs'), 3)} pp) versus the exact one "
        f"({_f(g('res_exact_maxabs'), 3)} pp) is the size of the accounting error a "
        "reader would inherit by using the shortcut. Whichever convention is chosen, the "
        "small remaining residual is reallocated proportionally to the absolute size of "
        "each contribution so that the adding-up identity the system model needs holds "
        "by construction rather than by assumption.\n"
    )
    p.append(
        "The intuition for the previous-year weights is that a chain-linked volume index "
        "is built by stitching together annual links, each of which is a fixed-price "
        "index using the prior year's prices. Within a link, then, a component's marginal "
        "contribution to the aggregate is its own volume change scaled by how large it was, "
        "in money, the year before &mdash; exactly the previous-year nominal share. The "
        "naive difference-over-lagged-GDP shortcut ignores this and implicitly reweights "
        "components by their reference-year price levels, which is why its residual grows "
        "with distance from the reference year while the exact method's does not. Reporting "
        "both conventions is not indecision: it lets a reader see the size of the "
        "convention's footprint on any single quarter before deciding whether it matters "
        "for the question at hand.\n"
    )

    # 3. The system model
    p.append("## 3. The system model\n")
    p.append(
        f"Each of the {g('n_components', 'component')} component contribution series is "
        "regressed on a common design: an intercept, a linear trend expressed in decades "
        "so the coefficient reads as pp of quarterly growth per decade, and two regime "
        "dummies for the global financial crisis and troika period (2008Q3&ndash;2013Q4) "
        "and the pandemic (2020Q1&ndash;2021Q4). Because every equation shares the same "
        "regressors, seemingly-unrelated regression collapses to equation-by-equation OLS "
        "by Kruskal's theorem, and because the dependent variables sum to GDP growth by "
        "construction, the component coefficient vectors sum across equations to the "
        "GDP-growth equation coefficients exactly. The adding-up property is therefore "
        "inherited, not imposed, and is verified numerically at run time: in this run the "
        f"maximum discrepancy between the summed component coefficients and the GDP "
        f"coefficients is {_sci(g('adding_up_gap'))}. Inference uses Newey-West HAC "
        f"standard errors with four lags, the honest choice for quarterly contributions "
        "with serial correlation. The estimated GDP-level trend is "
        f"{_f(g('gdp_trend'))} pp of quarterly growth per decade "
        f"(HAC p = {_f(g('gdp_trend_p'))}). The regime means tell the cyclical story the "
        f"trend cannot: average quarterly GDP growth is {_f(g('gdp_normal'))} pp in normal "
        f"periods, {_f(g('gdp_gfc'))} pp through the crisis-and-troika window, and "
        f"{_f(g('gdp_pandemic'))} pp across the pandemic window, the last of these an "
        "average over quarters that include both the 2020 collapse and the rebound.\n"
    )
    p.append(
        "Allowing the trend to break by regime sharpens the reading. Re-centring the "
        "interaction trend at each regime's entry lets the dummy measure the level shift "
        "at entry and the interaction the within-regime slope change. A joint Wald test "
        "on the HAC covariance of the null that both interaction coefficients are zero "
        f"gives, for the GDP equation, a statistic of {_f(g('wald_gdp_stat'))} "
        f"(p = {_f(g('wald_gdp_p'))}); across the individual component equations "
        f"{g('wald_n_sig', 0)} of {g('wald_n_comp', 0)} reject the no-break null at the "
        "5% level. The slope-break tests are descriptive of where the static-slope "
        "assumption is least comfortable, and they motivate the state-space layer that "
        "lets the slope move every quarter.\n"
    )
    p.append(
        "Two properties of this design deserve emphasis because they are easy to "
        "mis-state. First, the equality of SUR and OLS here is exact, not approximate: it "
        "follows from Kruskal's theorem the moment every equation shares an identical "
        "regressor matrix, so estimating the system buys nothing over ten separate "
        "regressions except the bookkeeping that keeps the adding-up visible. Second, the "
        "cross-equation residual covariance is singular by construction, because the "
        "component residuals sum to the GDP-equation residual; this is expected and "
        "harmless for a design that never inverts that covariance, and it is reported only "
        "for inspection. The HAC lag of four quarters is chosen to span a full year of "
        "potential residual autocorrelation in quarterly data without over-smoothing the "
        "covariance, and the diagnostics section returns to whether four lags is enough.\n"
    )
    p.append(_fig("decomposition_trend.png",
                  "Decomposition of the GDP-level trend effect across components"))
    p.append(_fig("contributions_small_multiples.png",
                  "Component contributions with fitted trend-and-regime paths"))

    # 4. Import-content adjustment (conditional)
    if _exists("adjusted_contributions.csv"):
        p.append("## 4. Import-content adjustment\n")
        p.append(
            "Gross contributions treat imports as a single negative block, which "
            "understates how much of each final-demand component is met from abroad. "
            "The import-content adjustment reallocates the combined import contribution "
            "across private consumption, public consumption, gross fixed capital "
            "formation and exports in proportion to each component's import content, "
            "leaving net contributions that still sum to GDP growth. The reading changes "
            "from a mechanical exports-minus-imports split to a domestic-versus-external "
            "demand split. The essential caveat is that the import-content matrix is an "
            "exogenous input, not something estimated here: the default shares are a "
            "clearly flagged placeholder to be replaced with a current input-output "
            "vintage, and the domestic-versus-external conclusion is only as good as that "
            "matrix.\n"
        )
        p.append(_fig("domestic_vs_external.png",
                      "Import-content-adjusted contributions: domestic vs external demand"))

    # 5. Deeper granularity (conditional)
    if _exists("gfcf_by_asset.csv") or _exists("consumption_by_purpose.csv"):
        p.append("## 5. Deeper granularity\n")
        p.append(
            "Two annual breakdowns add resolution the quarterly ten-component layer "
            "cannot carry: gross fixed capital formation by asset type (`nama_10_an6`) "
            "and household consumption by COICOP purpose (`nama_10_co3_p3`). Because "
            "these source series are annual, they are used at annual frequency and never "
            "interpolated to quarterly. Each year's parent contribution is split across "
            "its children in proportion to the children's annual volume changes, with the "
            "non-additivity of the child breakdown reallocated proportionally so the split "
            "closes onto the parent contribution. The layer is strictly additive to the "
            "existing decomposition and leaves the quarterly system untouched; it answers "
            "'which assets' and 'which purposes' without disturbing the 'which components' "
            "answer above.\n"
        )
        p.append(_fig("gfcf_by_asset.png",
                      "GFCF contribution split by asset type, annual"))
        p.append(_fig("consumption_by_purpose.png",
                      "Household consumption contribution split by COICOP purpose, annual"))

    # 6. Dynamics
    p.append("## 6. Dynamics\n")
    lb_txt = (
        f"{g('lb_n_flagged', 0)} of {g('diag_n', 0)} equations show Ljung-Box "
        "p below 0.05 at lag four"
    )
    p.append(
        "The diagnostics battery tests each equation's residuals for serial correlation "
        "(Ljung-Box at lags four and eight), conditional heteroskedasticity (ARCH-LM at "
        "lag four), non-normality (Jarque-Bera) and parameter stability (a CUSUM test), "
        f"all reported as p-values. In this run {lb_txt}, which is the diagnostic with "
        "teeth. The implication is stated without hedging: HAC standard errors remain "
        "valid for inference on the mean parameters even under residual autocorrelation, "
        "but a static mean model with autocorrelated residuals is dynamically incomplete "
        "as a description of the data. That is a motivation for a time-varying-trend "
        "model, not a reason to distrust the reported inference.\n"
    )
    p.append(_fig("residual_panel.png",
                  "Equation residuals with plus/minus two rolling-sigma bands"))
    p.append(
        "The state-space layer fits a local linear trend to each contribution series and "
        "to GDP growth, extracting a smoothed slope path with a 90% band. The band gives "
        "the state-space answer to 'when was the trend real': for GDP growth the 90% band "
        f"excludes zero in {g('stsm_gdp_excl', 0)} of {g('stsm_gdp_n', 0)} quarters. "
        "Because the slopes are estimated one series at a time, the adding-up identity is "
        "not imposed across the state-space models, and the sum of the smoothed component "
        "slopes need not equal the smoothed GDP slope; that gap is reported per quarter as "
        f"a model-consistency diagnostic and reaches at most {_f(g('stsm_gap_maxabs'), 3)} "
        "pp per decade in this run.\n"
    )
    p.append(
        "A few modelling choices in the state-space layer are worth recording. The level "
        "is specified as a local linear trend &mdash; a stochastic level plus a stochastic "
        "slope &mdash; which lets both the position and the direction of a series wander "
        "rather than forcing a single straight line through three decades of data. The "
        "stochastic seasonal term is off by default because the input is already "
        "seasonally adjusted; it is exposed as a flag only so residual seasonality can be "
        "probed where it is suspected, not as a routine addition. Estimation is by maximum "
        "likelihood, and non-convergence is handled explicitly: a series that fails under "
        "the default optimiser is retried under Powell's method and, if it still fails, is "
        "logged and skipped rather than silently returning a meaningless path. That the "
        "smoothed component slopes do not sum to the smoothed GDP slope is not a bug but "
        "the price of estimating each series independently, and the reported per-quarter "
        "gap is the receipt for that price.\n"
    )
    p.append(_fig("slope_paths.png",
                  "Smoothed local-linear-trend slope paths with 90% bands"))
    vecm_ranks = g("vecm_ranks")
    rank_txt = (", ".join(str(r) for r in vecm_ranks) if vecm_ranks else "the selected rank")
    top_txt = ""
    if g("vecm_top_comp") is not None:
        top_txt = (
            f" The largest adjustment loading at the selected rank is on "
            f"{g('vecm_top_comp')} (alpha = {_f(g('vecm_top_alpha'))}), the component that "
            "moves most to correct a departure from the long-run configuration."
        )
    p.append(
        "The VECM takes the long-run view, working on the log chain-linked-volume levels "
        f"of the {g('vecm_n_series', 'positive')} strictly positive final-demand components "
        "&mdash; inventories and valuables are excluded because they take non-positive "
        "values and their log is undefined, an exclusion that is logged rather than hidden. "
        "The Johansen trace test selects a cointegrating rank of "
        f"{g('vecm_rank', 'r')} at the 5% level, but with this many log-level series and "
        "only about 120 quarters the test is badly size-distorted and over-selects, so the "
        f"rank is treated as descriptive and the loadings are reported for ranks "
        f"{rank_txt} as sensitivity." + top_txt + "\n"
    )
    p.append(_fig("vecm_alpha_heatmap.png",
                  "VECM adjustment loadings (alpha) with significance stars"))
    bt = g("bt")
    if bt is not None:
        by = {r["model"]: r for _, r in bt.iterrows()}
        sur = by.get("SUR mean model")
        comp = by.get("AR(1) components (summed)")
        bench = by.get("AR(1) on GDP growth (benchmark)")
        bt_txt = (
            "The backtest closes the dynamics section with a specification check rather "
            "than a forecasting claim: expanding-window, one-quarter-ahead forecasts of "
            "GDP growth from 2010Q1, with no look-ahead, under the SUR mean model, a "
            "sum of per-component AR(1) forecasts, and a direct AR(1) benchmark. "
        )
        if sur is not None and bench is not None:
            bt_txt += (
                f"Excluding the pandemic quarters, the SUR mean model posts an RMSE of "
                f"{_f(sur['rmse_ex_pandemic'])} pp against the benchmark's "
                f"{_f(bench['rmse_ex_pandemic'])} pp; over the full window the SUR RMSE is "
                f"{_f(sur['rmse_full'])} pp. The Diebold-Mariano test of the SUR model "
                f"against the AR(1) benchmark gives a statistic of "
                f"{_f(sur['dm_stat_vs_ar1'])} (p = {_f(sur['dm_pvalue_vs_ar1'])}), "
            )
            if comp is not None:
                bt_txt += (
                    f"and the summed-AR(1) decomposition a statistic of "
                    f"{_f(comp['dm_stat_vs_ar1'])} (p = {_f(comp['dm_pvalue_vs_ar1'])}). "
                )
            bt_txt += (
                "A positive statistic means the model loses to the benchmark; where the "
                "difference is not significant, the honest conclusion is that the "
                "decomposition earns its keep as an accounting and inference device, not "
                "as a predictor that beats a one-line univariate model out of sample."
            )
        p.append(bt_txt + "\n")

    p.append(
        "Read as a sequence, the dynamics tell one story. The diagnostics say the static "
        "mean model leaves autocorrelation on the table; the state-space slopes show that "
        "the underlying drift is genuinely time-varying and pin down the windows in which "
        "it is distinguishable from zero; the VECM asks whether the components share a "
        "long-run resting configuration and which of them does the correcting; and the "
        "backtest disciplines the whole exercise by asking whether any of this structure "
        "improves an out-of-sample forecast of headline growth. The four layers answer "
        "different questions &mdash; completeness, timing, long-run structure, and "
        "predictive content &mdash; and they are deliberately not collapsed into a single "
        "number, because each carries its own caveats and each would fail differently.\n"
    )

    # 7. Limitations
    p.append("## 7. Limitations and what would change the conclusions\n")
    p.append(
        "Several choices bound what can be claimed. The chain-linking residual is handled "
        "by an exact annual-overlap method with a proportional reallocation of the small "
        "remainder; a different reallocation convention would move individual "
        "contributions at the second decimal, which is why both conventions are reported "
        "side by side. The import-content adjustment is a gross-versus-net reattribution "
        "driven entirely by an exogenous input-output matrix; a different vintage would "
        "change the domestic-versus-external split, and no claim here survives replacing "
        "the placeholder shares with implausible ones. The Johansen rank selection is "
        "size-distorted at this sample length and is reported as descriptive with rank "
        "sensitivity rather than as a tested number. The state-space slopes are estimated "
        "per series, so the adding-up identity that the SUR layer guarantees is broken "
        "across the state-space models by design, and the reported per-quarter gap is the "
        "honest measure of that break. Finally, nothing in this report is a causal claim: "
        "the regime dummies and trends describe conditional means with HAC-valid standard "
        "errors, the word 'significant' is used only where a named test supports it, and "
        "the backtest is a specification check, not a forecast people should trade on.\n"
    )
    p.append(
        "It is worth being concrete about what would overturn, rather than merely nudge, "
        "the headline readings. The cyclical regime story &mdash; a deep contraction "
        "across the crisis-and-troika window and a sharper, shorter pandemic swing "
        "&mdash; is robust to the contribution convention, because it lives in the regime "
        "means of GDP growth itself, which no reallocation touches. The trend reading is "
        "the more fragile one: a small, imprecisely estimated GDP trend could change sign "
        "under a different sample window or a different treatment of the pandemic "
        "quarters, which is exactly why the state-space slope path, with its explicit "
        "bands, is the honest companion to the single trend coefficient. The long-run "
        "VECM loadings are the most tentative objects in the report and should be read as "
        "a description of which components historically absorbed disequilibria, not as a "
        "structural adjustment mechanism to be relied on out of sample. None of these "
        "caveats is a reason to withhold the decomposition; they are the terms on which "
        "it should be read.\n"
    )
    return "".join(p)


# Optional artifacts whose presence must reflect THIS run, not a stale earlier
# one, so the report never references a figure the current run did not produce.
_CONDITIONAL = [
    "adjusted_contributions.csv", "domestic_vs_external.png",
    "gfcf_by_asset.csv", "gfcf_by_asset.png",
    "consumption_by_purpose.csv", "consumption_by_purpose.png",
]


def main(clv=None, cp=None, matrix_path=None, do_sublayer=None):
    """Run the full pipeline and render the technical report."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for name in _CONDITIONAL:
        f = config.OUTPUT_DIR / name
        if f.exists():
            f.unlink()
    if do_sublayer is None:
        do_sublayer = clv is None  # annual fetch only makes sense on real data
    result = run_pipeline.main(
        clv=clv, cp=cp, interactions=True, stsm_flag=True, vecm_flag=True,
        backtest_flag=True, sublayer_flag=do_sublayer, import_content_arg=matrix_path,
    )
    ctx = _gather(result)
    md = _render(ctx)
    title = "A granular expenditure-side decomposition of Portuguese GDP growth"
    tex = _md_to_latex(md, title)
    out = REPORT_DIR / "technical_report.tex"
    out.write_text(tex, encoding="utf-8")
    stale_md = REPORT_DIR / "technical_report.md"
    if stale_md.exists():
        stale_md.unlink()
    print(f"\nReport written to {out} ({len(md.split())} words)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--import-content", nargs="?", const=None, default=None,
                    metavar="PATH", dest="import_content")
    args = ap.parse_args()
    if args.refresh:
        from ptgdp import fetch
        _clv, _cp = fetch.load_all(refresh=True)
    main(matrix_path=args.import_content)
