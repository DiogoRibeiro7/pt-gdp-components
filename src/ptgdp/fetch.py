"""Fetch Portuguese quarterly national accounts from Eurostat via DBnomics.

Downloads chain-linked volumes (for growth/contributions) and current
prices (for nominal weights) for the expenditure-side components, caches
them as parquet, and returns tidy wide DataFrames indexed by quarter.
"""

from __future__ import annotations

import pandas as pd

from . import config


def _series_id(na_item: str, unit: str) -> str:
    return f"Eurostat/namq_10_gdp/{config.FREQ}.{unit}.{config.S_ADJ}.{na_item}.{config.GEO}"


def _wanted_items() -> list[str]:
    return list(config.COMPONENTS) + [config.GDP_ITEM, config.INVENTORY_FALLBACK]


def fetch_unit(unit: str) -> pd.DataFrame:
    """Fetch all component series for one unit; wide frame, PeriodIndex[Q]."""
    from dbnomics import fetch_series

    ids = [_series_id(item, unit) for item in _wanted_items()]
    raw = fetch_series(ids)
    if raw.empty:
        raise RuntimeError(f"DBnomics returned no data for unit={unit}")

    raw = raw[raw["value"].notna()]
    wide = (
        raw.pivot_table(index="original_period", columns="na_item", values="value")
        .rename_axis(index="quarter", columns=None)
    )
    wide.index = pd.PeriodIndex(wide.index, freq="Q")
    wide = wide.sort_index().loc[config.SAMPLE_START:]
    return wide


def load(unit: str, refresh: bool = False) -> pd.DataFrame:
    """Load from cache if present, otherwise fetch and cache."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache = config.DATA_DIR / f"namq_10_gdp_{config.GEO}_{unit}.parquet"
    if cache.exists() and not refresh:
        df = pd.read_parquet(cache)
        df.index = pd.PeriodIndex(df.index, freq="Q")
        return df
    df = fetch_unit(unit)
    out = df.copy()
    out.index = out.index.astype(str)
    out.to_parquet(cache)
    return df


def load_all(refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (chain-linked volumes, current prices)."""
    return load(config.UNIT_CLV, refresh), load(config.UNIT_CP, refresh)


# Annual breakdown datasets and the DBnomics dimension carrying their split.
_ANNUAL_DATASETS = {
    "nama_10_an6": "asset10",     # GFCF by AN_F6 asset type (gross, "*G" codes)
    "nama_10_co3_p3": "coicop",   # household consumption by COICOP purpose
}


def fetch_annual(dataset: str, items: list[str], unit: str) -> pd.DataFrame:
    """Fetch an annual Eurostat breakdown (chain-linked volumes) via DBnomics.

    Supports ``nama_10_an6`` (GFCF by AN_F6 asset, gross ``*G`` codes such as
    N111G, N112G, N1131G, N1132G, N11OG, N115G, N117G, on the ``asset10``
    dimension) and ``nama_10_co3_p3`` (household consumption by COICOP
    CP01..CP12), geo=PT. The data is annual and is
    returned as annual (year-indexed) — it is never interpolated to quarterly,
    since interpolation would invent within-year dynamics the source does not
    contain. The breakdown codes actually present in the response are logged;
    requested codes with no data are dropped rather than filled.
    """
    from dbnomics import fetch_series

    if dataset not in _ANNUAL_DATASETS:
        raise ValueError(f"unsupported annual dataset {dataset!r}; "
                         f"expected one of {list(_ANNUAL_DATASETS)}")
    dim = _ANNUAL_DATASETS[dataset]
    dims = {"freq": ["A"], "unit": [unit], "geo": [config.GEO], dim: list(items)}
    raw = fetch_series("Eurostat", dataset, dimensions=dims)
    if raw.empty:
        raise RuntimeError(f"DBnomics returned no data for {dataset} unit={unit}")

    raw = raw[raw["value"].notna()]
    present = sorted(raw[dim].unique())
    missing = [i for i in items if i not in present]
    print(f"[fetch_annual] {dataset}: found {present}"
          + (f"; requested-but-missing {missing}" if missing else ""))

    wide = (
        raw.pivot_table(index="original_period", columns=dim, values="value")
        .rename_axis(index="year", columns=None)
    )
    wide.index = pd.PeriodIndex(wide.index, freq="A").year
    return wide.sort_index()


if __name__ == "__main__":
    clv, cp = load_all(refresh=True)
    print("CLV:", clv.shape, clv.index.min(), "->", clv.index.max())
    print("CP :", cp.shape, cp.index.min(), "->", cp.index.max())
    print("Columns:", sorted(clv.columns))
