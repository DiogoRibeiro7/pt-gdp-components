# pt-gdp-components

Granular expenditure-side decomposition of Portuguese GDP: quarterly
contributions to growth for ~10 components (household and NPISH
consumption, government consumption, GFCF, inventories, valuables, and
goods/services splits of exports and imports), modelled as a SUR system
with an adding-up constraint.

## Data

Eurostat `namq_10_gdp` via DBnomics — chain-linked volumes (CLV20_MEUR,
seasonally and calendar adjusted) for growth and contributions, current
prices for nominal weights. Vintages that publish inventories only as
`P52_P53` are handled automatically.

## Method

1. **Contributions.** Default is the annual-overlap exact method,
   `contrib_{i,t} = sign_i · s_{i,y(t)-1} · (X_{i,t}/X_{i,t-1} − 1) · 100`,
   with `s` the previous-year current-price share of the component in
   nominal GDP. The naive `sign_i · ΔX_{i,t} / GDP_{t-1} · 100`
   approximation is kept as a robustness check (`contributions_approx`).
   Chain-linked volumes are non-additive, so any residual against GDP
   growth is computed explicitly and reallocated proportionally (or kept as
   a column with `allocate_residual=False`); `convention_comparison`
   reports the residuals side by side.
2. **System estimation.** Each component's contribution is regressed on a
   common design (intercept, trend in decades, GFC/troika dummy 2008Q3–
   2013Q4, pandemic dummy 2020Q1–2021Q4). With identical regressors, SUR
   equals per-equation OLS (Kruskal), and because the dependent variables
   sum to GDP growth, coefficients sum across equations to the GDP-growth
   equation coefficients exactly — verified numerically at run time
   (`adding_up_gap`). Inference uses Newey–West HAC SEs (maxlags=4).
3. **Outputs.** Coefficient tables, regime-mean tables, a stacked
   contributions chart, small multiples with fitted trend+regime paths,
   and coefficient-decomposition charts showing how each GDP-level effect
   splits across components.

## Usage

```bash
pip install -r requirements.txt
python run_pipeline.py --refresh   # downloads from DBnomics, caches to data/
python run_pipeline.py             # re-runs from cache
python tests/smoke_synthetic.py    # offline end-to-end check
```

## Extending granularity

- GFCF by asset type: `nama_10_an6` (annual) — dwellings, other
  construction, transport, ICT, other machinery, cultivated assets, IPP.
- Household consumption by COICOP purpose: `nama_10_co3_p3` (12 divisions).
- Add items to `COMPONENTS` in `src/ptgdp/config.py`; signs and labels are
  the only metadata required.

## Known limitations

- Contributions use the annual-overlap exact method by default: each
  component's quarter-on-quarter volume ratio is weighted by its
  previous-year current-price share of nominal GDP, the additive
  decomposition consistent with chain-linked Laspeyres volumes. The naive
  `ΔX/GDP` approximation is retained only as a robustness check
  (`contributions_approx`); `convention_comparison` and
  `output/convention_comparison.csv` quantify the residual each convention
  leaves against GDP growth before any reallocation. A small residual still
  remains under the exact method (chain-linking binds annually, GDP growth
  is quarter-on-quarter) and is reallocated proportionally so the SUR
  adding-up identity closes.
- Import contributions are gross by default. A demand-side attribution
  (import-content-adjusted contributions, à la Banco de Portugal) is
  available via `--import-content[=path]`: it reallocates the total import
  block across final-demand components using an exogenous import-content
  matrix (hardcoded placeholder shares flagged `REPLACE_WITH_CURRENT_VINTAGE`,
  or a user CSV). The input–output vintage is an input, never fetched.
- Regime dummies shift means only; interact them with `trend` in
  `prepare.design_matrix` for regime-specific slopes.
