# Technical report — QA findings

Adversarial review of `report/technical_report.tex` against the artifacts in
`output/`. Numbers were checked with `scripts/check_report_numbers.py`, which
extracts every decimal numeral from the report body and requires each to be
reproducible from a cell or a simple aggregate of an `output/` CSV within the
rounding the report used (integers, years, quarter labels and standard
statistical thresholds are whitelisted; residual-closure values below 1e-6 are
treated as numerically zero). The checker passes clean after the fixes below.

## Applied fixes (mechanical)

1. **Broken citation marker — §2 (Contribution methodology).** The
   `[[cite:eurostat_qna]]` marker rendered as literal text because the `_` in
   the key was LaTeX-escaped (`eurostat\_qna`) before the marker→`\cite`
   conversion ran, so the conversion regex no longer matched. Fixed by renaming
   the bibliography key to `eurostatqna` (no underscore) and hardening the
   marker regex to tolerate an escaped underscore. All eleven `\cite` keys now
   resolve with no `LaTeX Warning: Citation undefined`.

2. **Hardcoded count inconsistent with the data — §3 (The system model).** The
   prose read "buys nothing over ten separate regressions" while the same
   section states the model has 7 component equations (P52/P53 are absent from
   this vintage). Reworded to "the equation-by-equation regressions" so no
   hardcoded count can contradict the component set.

3. **Overreaching convention comparison — §2 (Contribution methodology).** The
   text framed the naive approximation as "the shortcut" whose larger accounting
   error a reader "would inherit", but on this sample the exact method's maximum
   absolute residual (1.901 pp) is in fact *larger* than the naive one
   (1.409 pp) — the annual-overlap ratio form amplifies quarters where a small
   component's volume swings sharply. Reworded to (a) report both maxima
   neutrally, (b) determine the direction of the comparison from the data at
   render time rather than assuming it, and (c) justify the exact method on the
   correctness of its previous-year nominal weights rather than on residual
   size. The companion claim that "the exact method's [residual] does not [grow
   with reference-year distance]" was softened to attribute the naive method's
   growth to a systematic reference-year-price bias and the exact method's
   occasional spikes to ratio noise.

## Open items (judgment calls, left flagged)

4. **Trend-coefficient interpretation — §3.** The cited GDP-level trend
   (−0.034 pp per decade) is the main-effect (baseline-regime) slope from the
   interactions-augmented design the report runs, not the average trend from a
   no-interaction fit. The number is artifact-backed and the regime interactions
   are described in the next paragraph, but a reader expecting the plain trend
   may misread it. Proposed fix (not auto-applied): add a half-sentence noting
   the trend is the baseline-regime slope when interactions are active.

5. **Granularity section describes an unavailable breakdown — §5.** The section
   describes both the GFCF-by-asset (`nama_10_an6`) and COICOP consumption
   (`nama_10_co3_p3`) breakdowns, but only the consumption figure is produced in
   this run: the annual GFCF-by-asset series returned no data from the current
   DBnomics mirror, so the layer was logged-and-skipped. The figure list
   honestly shows only what exists, but the prose reads as if both materialised.
   Proposed fix (not auto-applied): have the generator note when a sub-layer
   breakdown was requested but returned no data.

6. **Error-correction language — §6 (VECM).** "The component that moves most to
   correct a departure from the long-run configuration" uses standard VECM
   adjustment terminology. It is not a causal economic claim, and the
   limitations section explicitly disclaims causality and rereads the loadings
   as "which components historically absorbed disequilibria". No change made;
   recorded for transparency.

## Statistical-overreach scan

No causal language survives outside the limitations section, which disclaims it
explicitly. Every use of "significant" is tied to a named test (the Wald
slope-break test in §3, the Diebold–Mariano test in §6). The convention
(exact default, naive robustness check) is now described consistently in §1, §2
and §7.
