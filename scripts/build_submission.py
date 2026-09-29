"""Generate LaTeX inputs from results and stage the two final PDFs."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
from pypdf import PdfReader, PdfWriter

DISPLAY_NAMES = {
    "glove_target": "GloVe target",
    "glove_context_2": "GloVe context ($\\pm 2$)",
    "bert_mean_last_four": "BERT mean last four",
}

PAIR_LABELS = {
    "glove_target - glove_context_2": "Target GloVe minus nearby GloVe",
    "glove_target - bert_mean_last_four": "Target GloVe minus BERT",
    "glove_context_2 - bert_mean_last_four": "Nearby GloVe minus BERT",
}

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "macro_f1": "Macro F1",
    "roc_auc": "ROC-AUC",
}


def atomic_write_text(path: Path, content: str) -> None:
    """Write generated LaTeX without importing the inference-only ML stack."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f"{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as temporary:
        temporary.write(content)
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def _percent(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}"


def _macro(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


def _tex_escape(value: object) -> str:
    return str(value).replace("_", "\\_")


def generate_inputs(root: Path) -> None:
    results = root / "results/final"
    metrics = pd.read_csv(results / "metrics.csv").set_index("system")
    intervals = pd.read_csv(results / "bootstrap_ci.csv")
    layers = pd.read_csv(results / "layer_metrics.csv")
    paired = pd.read_csv(results / "paired_differences.csv")
    summary = json.loads((results / "analysis_summary.json").read_text(encoding="utf-8"))
    config = json.loads((results / "config.json").read_text(encoding="utf-8"))
    diagnostics = json.loads((results / "run_diagnostics.json").read_text(encoding="utf-8"))
    ledger = json.loads(
        (root / "results/raw/selection_ledger.json").read_text(encoding="utf-8")
    )
    target = metrics.loc["glove_target"]
    context = metrics.loc["glove_context_2"]
    bert = metrics.loc["bert_mean_last_four"]
    difference = paired.loc[
        (paired["contrast"] == "glove_context_2 - bert_mean_last_four")
        & (paired["metric"] == "accuracy")
    ].iloc[0]
    macro_difference = paired.loc[
        (paired["contrast"] == "glove_context_2 - bert_mean_last_four")
        & (paired["metric"] == "macro_f1")
    ].iloc[0]
    audits = {record["split"]: record for record in diagnostics["audits"]}
    best_layer = layers.loc[layers["accuracy"].idxmax()]
    context_different_recall = context["tn"] / (context["tn"] + context["fp"])
    bert_different_recall = bert["tn"] / (bert["tn"] + bert["fp"])
    context_window = ledger["selected_static_context"].removeprefix("glove_context_")
    context_candidates = ", ".join(
        "sentence" if value == "sentence" else f"$\\pm {value}$"
        for value in config["glove"]["context_windows"]
    )
    macros = "".join(
        (
            _macro("TargetAccuracy", f"{_percent(target['accuracy'])}\\%"),
            _macro("ContextAccuracy", f"{_percent(context['accuracy'])}\\%"),
            _macro("BertAccuracy", f"{_percent(bert['accuracy'])}\\%"),
            _macro("TargetAccuracyFraction", f"{target['accuracy']:.6f}"),
            _macro("ContextAccuracyFraction", f"{context['accuracy']:.6f}"),
            _macro("BertAccuracyFraction", f"{bert['accuracy']:.6f}"),
            _macro("TargetBarValue", f"{target['accuracy'] * 100:.1f}"),
            _macro("ContextBarValue", f"{context['accuracy'] * 100:.1f}"),
            _macro("BertBarValue", f"{bert['accuracy'] * 100:.1f}"),
            _macro("TargetCorrect", str(int(target["tn"] + target["tp"]))),
            _macro("BertCorrect", str(int(bert["tn"] + bert["tp"]))),
            _macro("ContextCorrect", str(int(context["tn"] + context["tp"]))),
            _macro("GainLower", _percent(-float(difference["upper"]))),
            _macro("GainUpper", _percent(-float(difference["lower"]))),
            _macro(
                "AccuracyGain",
                f"{_percent(bert['accuracy'] - context['accuracy'])}",
            ),
            _macro("BertMacroF", f"{_percent(bert['macro_f1'])}\\%"),
            _macro("ContextMacroF", f"{_percent(context['macro_f1'])}\\%"),
            _macro("BertAuc", f"{bert['roc_auc']:.3f}"),
            _macro("DifferenceLower", _percent(float(difference["lower"]))),
            _macro("DifferenceUpper", _percent(float(difference["upper"]))),
            _macro("MacroDifferenceLower", _percent(float(macro_difference["lower"]))),
            _macro("MacroDifferenceUpper", _percent(float(macro_difference["upper"]))),
            _macro(
                "BertGainCorrect",
                str(int(bert["tn"] + bert["tp"] - context["tn"] - context["tp"])),
            ),
            _macro("BertSameRecall", f"{_percent(bert['recall'])}\\%"),
            _macro("BertDifferentRecall", f"{_percent(bert_different_recall)}\\%"),
            _macro("ContextSameRecall", f"{_percent(context['recall'])}\\%"),
            _macro("ContextDifferentRecall", f"{_percent(context_different_recall)}\\%"),
            _macro("TrainCount", f"{audits['train']['examples']:,}"),
            _macro("ValidationCount", str(int(bert["count"]))),
            _macro("TestCount", f"{audits['test']['examples']:,}"),
            _macro("TuneCount", f"{ledger['tune_examples']:,}"),
            _macro("HoldoutCount", f"{ledger['holdout_examples']:,}"),
            _macro("ExperimentSeed", str(ledger["seed"])),
            _macro("GloveDimensions", str(config["glove"]["dimensions"])),
            _macro("ContextCandidates", context_candidates),
            _macro("SelectedContextWindow", f"$\\pm {context_window}$"),
            _macro(
                "BootstrapResamples",
                f"{config['evaluation']['bootstrap_resamples']:,}",
            ),
            _macro("BestAccuracyLayer", str(int(best_layer["system"].removeprefix("layer_")))),
            _macro("FinalLayerNumber", str(len(layers))),
            _macro(
                "BertOnlyCount", str(summary["partition_counts"]["bert_only_correct"])
            ),
            _macro(
                "BothCorrectCount", str(summary["partition_counts"]["both_primary_correct"])
            ),
            _macro(
                "StaticOnlyCount",
                str(summary["partition_counts"]["static_only_correct"]),
            ),
            _macro(
                "BothWrongCount", str(
                    summary["partition_counts"]["both_primary_wrong"]
                    + summary["partition_counts"]["all_wrong"]
                )
            ),
            _macro("BertErrorCount", str(int(bert["fp"] + bert["fn"]))),
        )
    )
    atomic_write_text(root / "poster/generated_results.tex", macros)

    primary_rows = []
    for system, row in metrics.iterrows():
        ci = intervals.loc[
            (intervals["system"] == system) & (intervals["metric"] == "accuracy")
        ].iloc[0]
        primary_rows.append(
            f"{DISPLAY_NAMES[system]} & {row['threshold']:.4f} & "
            f"{row['accuracy']:.4f} & [{ci['lower']:.4f}, {ci['upper']:.4f}] & "
            f"{row['macro_f1']:.4f} & {row['precision']:.4f} & "
            f"{row['recall']:.4f} & {row['roc_auc']:.4f} \\\\"
        )
    layer_rows = [
        f"{int(row.system.removeprefix('layer_'))} & {row.threshold:.4f} & "
        f"{row.accuracy:.4f} & {row.macro_f1:.4f} & {row.roc_auc:.4f} \\\\"
        for row in layers.itertuples(index=False)
    ]
    selection_rows = []
    for family, key in (
        ("Static context", "static_context_candidates"),
        ("BERT", "bert_candidates"),
    ):
        for record in ledger[key]:
            selected = record["name"] in {
                ledger["selected_static_context"],
                ledger["selected_bert_representation"],
            }
            selection_rows.append(
                f"{family} & {_tex_escape(record['name'])} & "
                f"{record['tune_macro_f1']:.4f} & "
                f"{record['holdout_macro_f1']:.4f} & "
                f"{'yes' if selected else 'no'} \\\\"
            )
    paired_rows = [
        f"{PAIR_LABELS[row.contrast]} & {METRIC_LABELS[row.metric]} & "
        f"{row.estimate:.4f} & [{row.lower:.4f}, {row.upper:.4f}] \\\\"
        for row in paired.itertuples(index=False)
    ]
    tables = (
        "\\newcommand{\\PrimaryMetricRows}{%\n"
        + "\n".join(primary_rows)
        + "}\n"
        + "\\newcommand{\\LayerMetricRows}{%\n"
        + "\n".join(layer_rows)
        + "}\n"
        + "\\newcommand{\\SelectionRows}{%\n"
        + "\n".join(selection_rows)
        + "}\n"
        + "\\newcommand{\\PairedRows}{%\n"
        + "\n".join(paired_rows)
        + "}\n"
    )
    atomic_write_text(root / "appendix/generated_tables.tex", tables)


def stage_submission(root: Path) -> None:
    submission = root / "submission"
    submission.mkdir(parents=True, exist_ok=True)
    allowed = {"poster.pdf", "appendix.pdf"}
    unexpected = {path.name for path in submission.iterdir()} - allowed
    if unexpected:
        raise RuntimeError(f"submission contains unexpected files: {sorted(unexpected)}")
    sources = {
        "poster.pdf": root / "poster/poster.pdf",
        "appendix.pdf": root / "appendix/appendix.pdf",
    }
    missing = [str(path) for path in sources.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"cannot stage missing PDFs: {missing}")
    for name, source in sources.items():
        shutil.copy2(source, submission / name)
    actual = {path.name for path in submission.iterdir() if path.is_file()}
    if actual != allowed:
        raise RuntimeError(f"submission file gate failed: {sorted(actual)}")


def _append_signed_declaration(appendix: Path, declaration: Path) -> None:
    """Replace both prefilled official forms with two personally signed copies."""
    if not declaration.is_file():
        raise RuntimeError(f"signed declaration not found: {declaration}")
    base = PdfReader(appendix)
    if len(base.pages) < 2 or any(
        "Declaration of Academic Integrity" not in (page.extract_text() or "")
        for page in base.pages[-2:]
    ):
        raise RuntimeError("appendix does not end with two official declaration forms")
    signed = PdfReader(declaration)
    if len(signed.pages) != 2:
        raise RuntimeError("provide exactly two signed declaration pages, one per author")

    writer = PdfWriter()
    for page in base.pages[:-2]:
        writer.add_page(page)
    for page in signed.pages:
        writer.add_page(page)
    writer.add_metadata({"/DeclarationStatus": "user-supplied-signed-forms"})
    temporary = appendix.with_name("appendix.with-declaration.pdf")
    with temporary.open("wb") as handle:
        writer.write(handle)
    temporary.replace(appendix)


def compile_documents(root: Path, tectonic: Path, declaration: Path | None = None) -> None:
    """Compile the declaration first, then the two deliverables."""
    if not tectonic.is_file():
        raise RuntimeError(f"Tectonic executable not found: {tectonic}")
    sources = (root / "poster/poster.tex", root / "appendix/appendix.tex")
    for source in sources:
        subprocess.run(
            [
                str(tectonic),
                "-X",
                "compile",
                "--outdir",
                str(source.parent),
                "--outfmt",
                "pdf",
                "--print",
                "--untrusted",
                source.name,
            ],
            cwd=source.parent,
            check=True,
        )
    if declaration is not None:
        _append_signed_declaration(root / "appendix/appendix.pdf", declaration)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile", action="store_true", help="compile all PDFs with Tectonic")
    parser.add_argument("--stage", action="store_true", help="also copy built PDFs")
    parser.add_argument(
        "--tectonic",
        type=Path,
        help="path to the Tectonic executable (required with --compile)",
    )
    parser.add_argument(
        "--declaration",
        type=Path,
        help="two-page PDF containing one signed official declaration per author",
    )
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    generate_inputs(root)
    if arguments.compile:
        if arguments.tectonic is None:
            parser.error("--compile requires --tectonic PATH")
        declaration = arguments.declaration.resolve() if arguments.declaration else None
        compile_documents(root, arguments.tectonic.resolve(), declaration)
    elif arguments.declaration is not None:
        parser.error("--declaration requires --compile")
    if arguments.stage:
        stage_submission(root)
        print("Staged exactly poster.pdf and appendix.pdf")
    else:
        print("Generated LaTeX result inputs")


if __name__ == "__main__":
    main()
