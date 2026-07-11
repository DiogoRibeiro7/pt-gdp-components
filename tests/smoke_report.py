"""Smoke test for the technical report: render it on synthetic data.

Reuses the synthetic frames built by the base smoke test, generates the report
through the same code path as production, and asserts the rendered file exists,
carries no leftover template placeholders, and references only figures that
were actually produced.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parents[1] / "tests"))

import smoke_synthetic as ss  # runs the base smoke and builds synthetic clv/cp
import run_report
from ptgdp import config

out = run_report.main(clv=ss.clv, cp=ss.cp, do_sublayer=False)
text = out.read_text(encoding="utf-8")

assert out.exists() and out.suffix == ".tex", "LaTeX report not written"
assert r"\begin{document}" in text and r"\end{document}" in text, "not a full document"
assert r"\section{" in text, "no sections rendered"

# no unrendered template placeholders left behind (doubled braces / literal markers)
assert "{{" not in text and "}}" not in text, "unrendered template placeholder found"
assert "PLACEHOLDER" not in text, "literal placeholder marker found"

# every included figure must exist in output/
referenced = re.findall(r"\\includegraphics\[[^\]]*\]\{\.\./output/([^}]+)\}", text)
for name in referenced:
    assert (config.OUTPUT_DIR / name).exists(), f"report references missing figure {name}"

assert referenced, "report references no figures"
print(f"\nSMOKE REPORT PASSED - {len(text.split())} words, "
      f"{len(referenced)} figures referenced")
