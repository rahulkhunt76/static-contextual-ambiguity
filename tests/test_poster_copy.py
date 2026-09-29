"""The current poster takes its displayed scores from saved experiment results."""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_canonical_poster_has_explanatory_sections() -> None:
    source = (ROOT / "poster/poster.tex").read_text(encoding="utf-8")
    for heading in (
        "Introduction",
        "Inspiration",
        "Background",
        "Hypothesis",
        "Methodology",
        "Results",
        "Conclusion",
    ):
        assert rf"\sectiontitle{{{heading}}}" in source
    assert "The appendix records the full method" not in source
    assert r"\input{generated_results.tex}" in source


def test_chart_widths_are_generated_from_saved_metrics() -> None:
    with (ROOT / "results/final/metrics.csv").open(encoding="utf-8", newline="") as handle:
        metrics = {row["system"]: row for row in csv.DictReader(handle)}
    macros = (ROOT / "poster/generated_results.tex").read_text(encoding="utf-8")
    source = (ROOT / "poster/poster.tex").read_text(encoding="utf-8")
    for system, name in (
        ("glove_target", "TargetBarValue"),
        ("glove_context_2", "ContextBarValue"),
        ("bert_mean_last_four", "BertBarValue"),
    ):
        expected = float(metrics[system]["accuracy"]) * 100
        assert rf"\newcommand{{\{name}}}{{{expected:.1f}}}" in macros
        assert rf"\{name}" in source
