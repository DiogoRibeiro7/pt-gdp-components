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

from ptgdp import config
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
clv["B1GQ"] = sum(signs[c] * clv[c] for c in signs)
# chain-linking non-additivity: perturb GDP slightly away from the exact sum
clv["B1GQ"] *= 1 + rng.normal(0, 0.0008, n)

result = run_pipeline.main(clv=clv)

assert result.adding_up_gap < 1e-8, "coefficient adding-up property failed"
outputs = list(config.OUTPUT_DIR.glob("*"))
assert len(outputs) >= 6, f"expected tables+figures, got {outputs}"
print("\nSMOKE TEST PASSED -", len(outputs), "output files")
