"""Verify every numeral in the technical report against the output artifacts.

Adversarial check: extract the decimal numerals from the report prose and
require each to be reproducible from a value in one of the CSVs in output/ --
either a raw cell or a simple column/group aggregate (mean, sum, min, max,
absolute max/mean) -- within the rounding the report itself used. Integers
(counts, ranks, lags, years, quarter labels) are treated as structural and are
not checked against the tables; a small set of standard statistical thresholds
and numerically-zero gap values are whitelisted.

Exit code 0 and a clean message means the prose numbers are all backed by an
artifact; a non-zero exit lists the unsupported numerals so they can be traced
to a table or removed.
"""

import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pandas as pd

from ptgdp import config

REPORT = config.PROJECT_ROOT / "report" / "technical_report.tex"

# standard thresholds that legitimately appear in prose without being estimates
THRESHOLDS = {0.05, 0.1, 0.01, 0.9, 0.95, 0.99, 0.5}
# below this magnitude a number is an "identity closes to ~0" gap, not a claim
ZERO_TOL = 1e-6

FLOAT_RE = re.compile(r"-?\d+\.\d+(?:[eE][+-]?\d+)?")


def supported_values() -> set[float]:
    """Raw cells plus simple column and per-group aggregates from every CSV."""
    vals: set[float] = set()
    for csv in sorted(config.OUTPUT_DIR.glob("*.csv")):
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        num = df.select_dtypes("number")
        for col in num.columns:
            s = num[col].dropna()
            vals.update(float(v) for v in s)
            if len(s):
                vals.update([
                    float(s.mean()), float(s.sum()), float(s.min()),
                    float(s.max()), float(s.abs().max()), float(s.abs().mean()),
                ])
        for gcol in df.select_dtypes(exclude="number").columns:
            for ncol in num.columns:
                try:
                    grp = df.groupby(gcol)[ncol]
                    vals.update(float(v) for v in grp.max().dropna())
                    vals.update(float(v) for v in grp.min().dropna())
                    vals.update(float(v) for v in
                                grp.apply(lambda x: x.abs().max()).dropna())
                except Exception:
                    continue
    return vals


def _body() -> str:
    t = REPORT.read_text(encoding="utf-8")
    m = re.search(r"\\begin\{document\}(.*)\\end\{document\}", t, re.S)
    return m.group(1) if m else t


def _matches(tok: str, supported: set[float]) -> bool:
    val = float(tok)
    if abs(val) < ZERO_TOL:
        return True  # numerically-zero gap / residual-closure value
    if "e" in tok.lower():
        target = f"{val:.1e}"
        return (any(f"{s:.1e}" == target for s in supported)
                or any(math.isclose(s, val, rel_tol=0.1, abs_tol=1e-18)
                       for s in supported))
    dec = len(tok.split(".")[1])
    tol = 0.5 * 10 ** (-dec) * 1.0001
    return any(abs(s - val) <= tol for s in supported)


def main() -> int:
    supported = supported_values()
    body = _body()
    unsupported = []
    for tok in FLOAT_RE.findall(body):
        val = float(tok)
        if val in THRESHOLDS or abs(val) in THRESHOLDS:
            continue
        if _matches(tok, supported):
            continue
        unsupported.append(tok)
    if unsupported:
        print("check_report_numbers: UNSUPPORTED numerals (not found in output/ CSVs):")
        for tok in unsupported:
            print(f"  {tok}")
        return 1
    print("check_report_numbers: all prose numerals are backed by output/ artifacts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
