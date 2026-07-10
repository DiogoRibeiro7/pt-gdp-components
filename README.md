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

1. **Contributions.** `contrib_{i,t} = sign_i · ΔX_{i,t} / GDP_{t-1} · 100`,
   the standard CLV approximation. Chain-linked volumes are non-additive,
   so the residual against GDP growth is computed explicitly and
   reallocated proportionally (or kept as a column with
   `allocate_residual=False`).
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

- The proportional reallocation of the chain-linking residual is a
  convention, not a statistical model; for published figures, state it.
- Import contributions are gross: a demand-side attribution of imports
  (import-content-adjusted contributions, à la Banco de Portugal) requires
  input–output weights and is out of scope here.
- Regime dummies shift means only; interact them with `trend` in
  `prepare.design_matrix` for regime-specific slopes.
