"""Configuration for the Portuguese GDP expenditure-side decomposition pipeline.

Data source: Eurostat namq_10_gdp via DBnomics.
Series key format: Eurostat/namq_10_gdp/{FREQ}.{UNIT}.{S_ADJ}.{NA_ITEM}.{GEO}
"""

from pathlib import Path

GEO = "PT"
FREQ = "Q"
S_ADJ = "SCA"          # seasonally and calendar adjusted
UNIT_CLV = "CLV20_MEUR"  # chain-linked volumes, reference year 2020
UNIT_CP = "CP_MEUR"      # current prices (for nominal weights)

# Expenditure-side components at the ~10-component level.
# sign = +1 for demand items, -1 for imports (they subtract from GDP).
COMPONENTS = {
    "P31_S14_S15": {"label": "Household + NPISH consumption", "sign": +1},
    "P3_S13":      {"label": "Government consumption",        "sign": +1},
    "P51G":        {"label": "Gross fixed capital formation", "sign": +1},
    "P52":         {"label": "Changes in inventories",        "sign": +1},
    "P53":         {"label": "Acquisitions of valuables",     "sign": +1},
    "P61":         {"label": "Exports of goods",              "sign": +1},
    "P62":         {"label": "Exports of services",           "sign": +1},
    "P71":         {"label": "Imports of goods",              "sign": -1},
    "P72":         {"label": "Imports of services",           "sign": -1},
}

GDP_ITEM = "B1GQ"

# Some vintages publish P52 and P53 only as combined P52_P53; the fetch
# layer falls back to the combined series and the prepare layer handles
# whichever set arrives.
INVENTORY_FALLBACK = "P52_P53"

# Regime dummies for the SUR design matrix (inclusive quarter ranges).
REGIMES = {
    "gfc":      ("2008Q3", "2013Q4"),  # global financial crisis + troika programme
    "pandemic": ("2020Q1", "2021Q4"),
}

# Sample window (namq_10_gdp for PT is dense from the mid-1990s onward).
SAMPLE_START = "1995Q1"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

FIGURE_CREDIT = "Source: Eurostat (namq_10_gdp) via DBnomics | Figure: Diogo Ribeiro"
