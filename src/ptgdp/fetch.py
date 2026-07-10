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


if __name__ == "__main__":
    clv, cp = load_all(refresh=True)
    print("CLV:", clv.shape, clv.index.min(), "->", clv.index.max())
    print("CP :", cp.shape, cp.index.min(), "->", cp.index.max())
    print("Columns:", sorted(clv.columns))
