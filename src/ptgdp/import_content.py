"""Demand-side attribution of imports: domestic vs external demand.

Gross expenditure-side contributions treat imports (P71 + P72) as a single
negative block. That understates how much of each final-demand component is
actually met from abroad and overstates domestic value added. This module
reallocates the total import contribution across the final-demand components
in proportion to each component's own import content, so the remaining "net"
contributions read as domestic value added by final use.

Default import-content shares
-----------------------------
The default matrix is a hardcoded stand-in and MUST be replaced with a
current vintage before any published use. Source of the placeholder values:

    Banco de Portugal, "The import content of global demand in Portugal",
    Economic Studies, Vol. II, No. 2 (2016) — Leontief import-content
    estimates by final-demand component.

The numbers below are rounded, illustrative, and flagged
REPLACE_WITH_CURRENT_VINTAGE; do not cite them. A demand-side attribution is
only as good as its input-output vintage, which is why the matrix is an
exogenous input here and the loader lets a user override every value.

We deliberately do NOT download or derive input-output tables in code: the
import-content matrix is a modelling choice that belongs to the analyst, not
a silently fetched artifact.
"""

from __future__ import annotations

import pandas as pd

# REPLACE_WITH_CURRENT_VINTAGE — illustrative import-content shares (fraction
# of each final-demand component sourced from imports), Portugal.
DEFAULT_IMPORT_CONTENT: dict[str, float] = {
    "private_consumption": 0.24,   # REPLACE_WITH_CURRENT_VINTAGE
    "public_consumption": 0.09,    # REPLACE_WITH_CURRENT_VINTAGE
    "gfcf": 0.34,                  # REPLACE_WITH_CURRENT_VINTAGE
    "exports": 0.38,              # REPLACE_WITH_CURRENT_VINTAGE
}

# Map each final-demand group to the contribution columns it covers.
DEMAND_MAP: dict[str, list[str]] = {
    "private_consumption": ["P31_S14_S15"],
    "public_consumption": ["P3_S13"],
    "gfcf": ["P51G"],
    "exports": ["P61", "P62"],
}

# Import columns (sign −1 in config.COMPONENTS) whose contribution is redistributed.
IMPORT_COLS: tuple[str, ...] = ("P71", "P72")


def load_import_content(path) -> dict[str, float]:
    """Load a user CSV (columns: component, import_share) overriding defaults.

    Any group present in the CSV replaces the default; groups absent from the
    CSV keep their default value. Group names must match the keys of
    ``DEFAULT_IMPORT_CONTENT``.
    """
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    if "component" not in cols or "import_share" not in cols:
        raise ValueError("import-content CSV needs 'component' and 'import_share' columns")
    shares = dict(DEFAULT_IMPORT_CONTENT)
    for _, row in df.iterrows():
        key = str(row[cols["component"]]).strip()
        if key not in DEFAULT_IMPORT_CONTENT:
            raise ValueError(f"unknown import-content component {key!r}; "
                             f"expected one of {list(DEFAULT_IMPORT_CONTENT)}")
        shares[key] = float(row[cols["import_share"]])
    return shares


def _column_shares(contrib: pd.DataFrame, shares: dict[str, float]) -> pd.Series:
    """Expand group shares to a per-column share vector over demand columns."""
    out = {}
    for group, cols in DEMAND_MAP.items():
        for col in cols:
            if col in contrib.columns:
                out[col] = shares[group]
    return pd.Series(out, dtype=float)


def adjusted_contributions(contrib: pd.DataFrame,
                           shares: dict[str, float]) -> pd.DataFrame:
    """Reallocate total import contributions across demand components.

    The combined import contribution (P71 + P72, a negative block) is spread
    across the final-demand components in proportion to ``share_i · contrib_i``
    each quarter. The returned frame drops the standalone import columns; the
    demand columns become net of their import content and any component
    without an assigned share (e.g. inventories) is carried through unchanged.

    Because the redistributed weights sum to one each quarter, the net
    contributions still sum to GDP growth exactly — asserted against the
    original row sums.
    """
    demand_share = _column_shares(contrib, shares)
    import_cols = [c for c in IMPORT_COLS if c in contrib.columns]
    total_import = contrib[import_cols].sum(axis=1)

    demand_cols = list(demand_share.index)
    signed = contrib[demand_cols].mul(demand_share, axis=1)  # share_i * contrib_i
    denom = signed.sum(axis=1)

    weights = signed.div(denom, axis=0)
    # Fallback where the signed weights cancel (denominator ~ 0): split by
    # share alone so the weights still sum to one and the identity holds.
    bad = denom.abs() < 1e-12
    if bad.any():
        share_fallback = demand_share / demand_share.sum()
        for col in demand_cols:
            weights.loc[bad, col] = share_fallback[col]

    adjusted = contrib.drop(columns=import_cols).copy()
    for col in demand_cols:
        adjusted[col] = contrib[col] + weights[col] * total_import

    gap = (adjusted.sum(axis=1) - contrib.sum(axis=1)).abs().max()
    assert gap < 1e-9, f"adjusted contributions broke adding-up, max gap {gap}"
    return adjusted
