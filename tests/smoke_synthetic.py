"""Smoke test: run the full pipeline on synthetic data shaped like namq_10_gdp.

Validates the contribution arithmetic, the adding-up identity, the SUR
coefficient-sum property, and figure generation without touching the
network. Not a substitute for the real data; run `python run_pipeline.py
--refresh` for that.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd

from ptgdp import config, prepare, sublayer
import run_pipeline

rng = np.random.default_rng(7)
idx = pd.period_range("1995Q1", "2025Q4", freq="Q")
n = len(idx)

# base levels roughly in the ballpark of PT quarterly CLV (million EUR)
base = {
    "P31_S14_S15": 30000, "P3_S13": 9000, "P51G": 9000,
    "P52": 300, "P53": 30, "P61": 8000, "P62": 5000,
    "P71": 10000, "P72": 3000,
}

def simulate(level, drift, vol, crisis_hit, pandemic_hit):
    g = rng.normal(drift, vol, n)
    crisis = (idx >= pd.Period("2008Q3")) & (idx <= pd.Period("2013Q4"))
    pand = (idx >= pd.Period("2020Q1")) & (idx <= pd.Period("2021Q4"))
    g[crisis] += crisis_hit
    g[pand] += pandemic_hit
    return level * np.cumprod(1 + g / 100)

clv = pd.DataFrame(index=idx)
clv["P31_S14_S15"] = simulate(base["P31_S14_S15"], 0.45, 0.8, -0.9, -1.5)
clv["P3_S13"]      = simulate(base["P3_S13"],      0.35, 0.5, -0.5,  0.3)
clv["P51G"]        = simulate(base["P51G"],        0.30, 2.0, -2.0, -1.0)
clv["P52"]         = base["P52"] + rng.normal(0, 250, n).cumsum() * 0.05 + rng.normal(0, 300, n)
clv["P53"]         = base["P53"] * np.abs(1 + rng.normal(0, 0.15, n))
clv["P61"]         = simulate(base["P61"], 0.9, 1.8, -1.2, -2.0)
clv["P62"]         = simulate(base["P62"], 1.1, 2.2, -1.0, -6.0)
clv["P71"]         = simulate(base["P71"], 0.8, 1.7, -1.5, -1.8)
clv["P72"]         = simulate(base["P72"], 0.9, 2.0, -1.0, -3.0)

signs = {k: v["sign"] for k, v in config.COMPONENTS.items()}
comp_cols = list(signs)

# Current-price frame consistent with the CLV frame: CP = CLV x a slowly
# drifting per-component deflator (prices diverge across components over
# time, which is exactly what makes chain-linked volumes non-additive).
deflator = pd.DataFrame(index=idx)
for j, c in enumerate(comp_cols):
    drift = 0.0006 + 0.0004 * j / len(comp_cols)  # slightly different per component
    deflator[c] = np.exp(np.arange(n) * drift + rng.normal(0, 0.001, n).cumsum())
cp = pd.DataFrame(index=idx)
for c in comp_cols:
    cp[c] = clv[c] * deflator[c]
# nominal GDP is additive in current prices
cp["B1GQ"] = sum(signs[c] * cp[c] for c in comp_cols)

# Build the GDP chain-linked volume as an annual-overlap aggregate of the
# component volumes, so the exact method reconstructs it near-perfectly while
# the naive Delta/GDP approximation (which sums CLV-unit differences across
# components with diverging deflators) carries a larger residual.
annual_cp = cp.groupby(cp.index.year).sum()
shares = pd.DataFrame(index=annual_cp.index)
for c in comp_cols:
    shares[c] = annual_cp[c] / annual_cp["B1GQ"]
prev_year = idx.year - 1
gdp_g = np.zeros(n)
for c in comp_cols:
    s_prev = shares[c].reindex(prev_year).to_numpy()
    ratio = (clv[c] / clv[c].shift(1) - 1.0).to_numpy()
    gdp_g += signs[c] * np.nan_to_num(s_prev * ratio)
gdp_level = np.empty(n)
gdp_level[0] = sum(signs[c] * clv[c].iloc[0] for c in comp_cols)
for t in range(1, n):
    gdp_level[t] = gdp_level[t - 1] * (1 + gdp_g[t])
clv["B1GQ"] = gdp_level

conv = prepare.convention_comparison(clv, cp)
mae_exact = conv["residual_exact"].abs().mean()
mae_approx_raw = conv["residual_approx_raw"].abs().mean()
assert mae_exact < mae_approx_raw, (
    f"exact residual not smaller than raw approximation: "
    f"exact={mae_exact:.4g} approx={mae_approx_raw:.4g}"
)

# --- sub-component layer reconciliation (B1) -----------------------------
contrib_q, _gq, _cq = prepare.contributions(clv, cp, method="exact")
parent_level = clv["P51G"].groupby(clv.index.year).sum()
child_fracs = {"assetA": 0.5, "assetB": 0.3, "assetC": 0.2}
child_levels = pd.DataFrame(index=parent_level.index)
for name, frac in child_fracs.items():
    # sub-0.5% per-child non-additivity noise so children sum to the parent closely
    child_levels[name] = parent_level.to_numpy() * frac * (
        1 + rng.normal(0, 0.0012, len(parent_level))
    )
coverage = (child_levels.sum(axis=1) / parent_level - 1.0).abs().max()
assert coverage < 0.005, f"synthetic children deviate from parent by {coverage:.4f}"

tidy, resid = sublayer.within_component_decomposition(contrib_q["P51G"], child_levels)
parent_annual = contrib_q["P51G"].groupby(contrib_q.index.year).sum()
recon = tidy.groupby("year")["contribution_pp"].sum()
common = recon.index.intersection(parent_annual.index.astype(int))
close = (recon.loc[common] - parent_annual.loc[common]).abs().max()
assert close < 1e-9, f"sub-layer reconciliation did not close, gap {close:.2e}"
assert resid.abs().max() < 1e-9, "reconciliation residual not closed"

result = run_pipeline.main(clv=clv, cp=cp, stsm_flag=True, vecm_flag=True,
                           backtest_flag=True, msm_flag=True, quantile_flag=True,
                           factor_flag=True)

# dynamic factor common cycle ran and wrote a factor path + loadings
if (config.OUTPUT_DIR / "factor_path.csv").exists():  # skipped only on non-convergence
    fac = pd.read_csv(config.OUTPUT_DIR / "factor_loadings.csv")
    assert {"component", "loading", "r2"}.issubset(fac.columns), "factor loadings malformed"

# quantile regression ran and wrote coefficient paths for GDP + components
qtab = pd.read_csv(config.OUTPUT_DIR / "quantile_coefficients.csv")
assert {"equation", "tau", "regressor", "coef"}.issubset(qtab.columns)
assert (qtab["equation"] == "GDP (system sum)").any(), "quantile GDP paths missing"

# Markov-switching regime dating ran and wrote smoothed probabilities
msm_csv = config.OUTPUT_DIR / "msm_probabilities.csv"
if msm_csv.exists():  # skipped only on a non-convergence (logged, not a crash)
    msm_df = pd.read_csv(msm_csv)
    assert "prob_low_growth" in msm_df.columns, "MSM probabilities malformed"
    assert (config.OUTPUT_DIR / "msm_regimes.csv").exists(), "MSM regimes missing"

# backtest ran with DM p-values present
bt_df = pd.read_csv(config.OUTPUT_DIR / "backtest.csv")
assert {"model", "rmse_full", "mae_full", "dm_stat_vs_ar1",
        "dm_pvalue_vs_ar1"}.issubset(bt_df.columns)
non_bench = bt_df[bt_df["model"].str.contains("benchmark") == False]  # noqa: E712
assert non_bench["dm_pvalue_vs_ar1"].notna().all(), "DM p-values missing"

# state-space slope paths ran without an unhandled convergence failure
stsm_csv = config.OUTPUT_DIR / "stsm_slopes.csv"
assert stsm_csv.exists(), "stsm_slopes.csv not written"
stsm_df = pd.read_csv(stsm_csv)
assert {"quarter", "component", "slope", "lo90", "hi90"}.issubset(stsm_df.columns)
assert (stsm_df["component"] == "sum_minus_gdp_gap").any(), "model-consistency gap missing"

# VECM ran and wrote the Johansen table and stacked alpha sensitivity tables
assert (config.OUTPUT_DIR / "vecm_johansen.csv").exists(), "vecm_johansen.csv missing"
alpha_df = pd.read_csv(config.OUTPUT_DIR / "vecm_alpha.csv")
assert {"rank", "component", "relation", "alpha"}.issubset(alpha_df.columns)
assert alpha_df["rank"].nunique() >= 1, "no VECM alpha tables written"

assert result.adding_up_gap < 1e-8, "coefficient adding-up property failed"

# diagnostics battery: one row per component + GDP, all five statistics present
diag = pd.read_csv(config.OUTPUT_DIR / "diagnostics.csv")
diag_stats = ["ljung_box_p_lag4", "ljung_box_p_lag8", "arch_lm_p_lag4",
              "jarque_bera_p", "cusum_p"]
assert len(diag) == result.params.shape[0] + 1, "diagnostics row count wrong"
assert all(c in diag.columns for c in diag_stats), "diagnostics missing statistics"

outputs = list(config.OUTPUT_DIR.glob("*"))
assert len(outputs) >= 6, f"expected tables+figures, got {outputs}"
print(f"\nResidual MAE: exact={mae_exact:.4g}  approx_raw={mae_approx_raw:.4g}")
print("SMOKE TEST PASSED -", len(outputs), "output files")
