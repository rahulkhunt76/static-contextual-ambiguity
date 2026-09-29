"""Fail closed unless the staged submission PDFs satisfy the release gates."""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections.abc import Iterator
from pathlib import Path

from PIL import Image
from pypdf import PdfReader
from pypdf.generic import DictionaryObject

A1_POINTS = (1683.78, 2383.94)
A4_POINTS = (595.28, 841.89)
EXPECTED_FILES = {"appendix.pdf", "poster.pdf"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _walk_resource_fonts(resources: DictionaryObject) -> Iterator[DictionaryObject]:
    """Yield font dictionaries, including fonts nested in imported PDF figures."""
    fonts = resources.get("/Font")
    if fonts:
        for reference in fonts.get_object().values():
            yield reference.get_object()
    xobjects = resources.get("/XObject")
    if xobjects:
        for reference in xobjects.get_object().values():
            xobject = reference.get_object()
            nested = xobject.get("/Resources")
            if nested:
                yield from _walk_resource_fonts(nested.get_object())


def _font_is_embedded(font: DictionaryObject) -> bool:
    # Matplotlib's PDF backend embeds glyph outlines as Type 3 CharProcs rather
    # than attaching a FontFile stream to the descriptor.
    if font.get("/Subtype") == "/Type3" and font.get("/CharProcs"):
        return True
    candidates = [font]
    descendants = font.get("/DescendantFonts")
    if descendants:
        candidates.extend(ref.get_object() for ref in descendants.get_object())
    for candidate in candidates:
        descriptor = candidate.get("/FontDescriptor")
        if descriptor:
            descriptor = descriptor.get_object()
            if any(descriptor.get(key) for key in ("/FontFile", "/FontFile2", "/FontFile3")):
                return True
    return False


def _assert_size(page: object, expected: tuple[float, float], label: str) -> None:
    actual = (float(page.mediabox.width), float(page.mediabox.height))
    if any(
        abs(measured - required) > 0.1
        for measured, required in zip(actual, expected, strict=True)
    ):
        raise RuntimeError(f"{label} has page size {actual}, expected {expected} points")


def _verify_fonts(reader: PdfReader, label: str) -> int:
    fonts: dict[str, bool] = {}
    for page in reader.pages:
        resources = page.get("/Resources")
        if not resources:
            continue
        for font in _walk_resource_fonts(resources.get_object()):
            name = str(font.get("/BaseFont", "unnamed font"))
            fonts[name] = fonts.get(name, False) or _font_is_embedded(font)
    missing = sorted(name for name, embedded in fonts.items() if not embedded)
    if missing:
        raise RuntimeError(f"{label} contains non-embedded fonts: {missing}")
    if not fonts:
        raise RuntimeError(f"{label} exposes no fonts for verification")
    return len(fonts)


def _verify_text(reader: PdfReader, required: tuple[str, ...], label: str) -> str:
    text = " ".join(" ".join(page.extract_text() or "" for page in reader.pages).split())
    missing = [heading for heading in required if heading not in text]
    if missing:
        raise RuntimeError(f"{label} is missing extractable text: {missing}")
    if "??" in text:
        raise RuntimeError(f"{label} contains a possible unresolved LaTeX reference")
    return text


def _verify_poster_sources(root: Path) -> int:
    source = (root / "poster/poster.tex").read_text(encoding="utf-8")
    required = (
        r"\newcommand{\bodyfont}{\fontsize{24}{27}\selectfont}",
        r"\input{generated_results.tex}",
        r"\sectiontitle{Conclusion}",
        r"\resultbar{GloVe target only}",
        "github.com/Mr-Dark-debug/static-contextual-ambiguity",
    )
    if any(marker not in source for marker in required):
        raise RuntimeError("poster is missing the current copy, chart, link, or 24 pt body type")
    if "tcolorbox" in source or "One success, one warning" in source or "Limitations" in source:
        raise RuntimeError("poster contains a removed panel or container")
    if r"\input{authors.tex}" not in source or "../assets/university-trier.pdf" not in source:
        raise RuntimeError("poster is missing its author block or official university logo")
    pngs = sorted((root / "figures").glob("*.png"))
    if not pngs:
        raise RuntimeError("no high-resolution figure previews found")
    for path in pngs:
        with Image.open(path) as image:
            dpi = image.info.get("dpi", (0, 0))
        if min(dpi) < 150:
            raise RuntimeError(f"{path.name} is below 150 PPI: {dpi}")
    preview = root / "poster/preview.png"
    with Image.open(preview) as image:
        preview_dpi = image.info.get("dpi", (0, 0))
    # PNG stores pixels per metre as an integer, so a requested 150 PPI is
    # represented as roughly 149.99 when converted back.
    if min(preview_dpi) < 149:
        raise RuntimeError(f"poster preview is below the 150 PPI target: {preview_dpi}")
    return len(pngs)


def _verify_result_trace(root: Path, poster_text: str) -> int:
    results = root / "results/final"
    with (results / "metrics.csv").open(encoding="utf-8", newline="") as handle:
        metrics = {row["system"]: row for row in csv.DictReader(handle)}
    with (results / "paired_differences.csv").open(encoding="utf-8", newline="") as handle:
        paired = list(csv.DictReader(handle))
    with (results / "predictions.csv").open(encoding="utf-8", newline="") as handle:
        predictions = list(csv.DictReader(handle))
    outcomes = {(True, True): 0, (True, False): 0, (False, True): 0, (False, False): 0}
    for row in predictions:
        columns = (row["bert_mean_last_four_correct"], row["glove_context_2_correct"])
        if any(value not in {"True", "False"} for value in columns):
            raise RuntimeError("invalid correctness value in prediction ledger")
        outcomes[tuple(value == "True" for value in columns)] += 1

    context = metrics["glove_context_2"]
    bert = metrics["bert_mean_last_four"]
    if len(predictions) != int(float(bert["count"])):
        raise RuntimeError("prediction count differs from the reported evaluation count")
    for record, correct in (
        (bert, outcomes[(True, True)] + outcomes[(True, False)]),
        (context, outcomes[(True, True)] + outcomes[(False, True)]),
    ):
        if correct != int(float(record["tn"]) + float(record["tp"])):
            raise RuntimeError("correct-answer count differs from the saved predictions")
    accuracy_difference = next(
        row
        for row in paired
        if row["contrast"] == "glove_context_2 - bert_mean_last_four"
        and row["metric"] == "accuracy"
    )
    bert_gain_correct = int(
        float(bert["tn"])
        + float(bert["tp"])
        - float(context["tn"])
        - float(context["tp"])
    )
    fragments = {
        f"{float(metrics['glove_target']['accuracy']) * 100:.1f}%",
        f"{float(context['accuracy']) * 100:.1f}%",
        f"{float(bert['accuracy']) * 100:.1f}%",
        f"{-float(accuracy_difference['upper']) * 100:.1f}",
        f"{-float(accuracy_difference['lower']) * 100:.1f}",
        f"{(float(bert['accuracy']) - float(context['accuracy'])) * 100:.1f}",
        str(bert_gain_correct),
        str(int(float(bert["tn"]) + float(bert["tp"]))),
        str(int(float(context["tn"]) + float(context["tp"]))),
        str(len(predictions)),
        str(int(float(bert["fp"]) + float(bert["fn"]))),
    }
    missing = sorted(fragment for fragment in fragments if fragment not in poster_text)
    if missing:
        raise RuntimeError(f"poster/result traceability check failed for values: {missing}")
    return len(fragments)


def verify(root: Path, *, signed: bool = False) -> None:
    submission = root / "submission"
    actual = {path.name for path in submission.iterdir() if path.is_file()}
    if actual != EXPECTED_FILES:
        expected = sorted(EXPECTED_FILES)
        raise RuntimeError(f"submission must contain exactly {expected}: {sorted(actual)}")

    source_paths = {
        "poster.pdf": root / "poster/poster.pdf",
        "appendix.pdf": root / "appendix/appendix.pdf",
    }
    for name, source in source_paths.items():
        staged = submission / name
        if _sha256(source) != _sha256(staged):
            raise RuntimeError(f"{name} differs from its built source")

    poster = PdfReader(submission / "poster.pdf")
    if len(poster.pages) != 1:
        raise RuntimeError(f"poster must have one page, found {len(poster.pages)}")
    _assert_size(poster.pages[0], A1_POINTS, "poster")
    poster_text = _verify_text(
        poster,
        (
            "Static vs Contextual Embeddings for Lexical Ambiguity",
            "A Word-in-Context Evaluation",
            "Hypothesis",
            "Methodology",
            "Results",
            "Conclusion",
            "Introduction",
            "Inspiration",
            "Background",
            "lexical ambiguity",
            "static embedding",
            "contextual",
            "Choudhary Prashant Santosh",
            "1910474",
            "Rahul Khunt",
            "1911272",
        ),
        "poster",
    )
    poster_fonts = _verify_fonts(poster, "poster")
    figure_count = _verify_poster_sources(root)
    traced_values = _verify_result_trace(root, poster_text)

    appendix = PdfReader(submission / "appendix.pdf")
    expected_pages = "at least eight"
    if len(appendix.pages) < 8:
        raise RuntimeError(
            f"appendix must have {expected_pages} pages, found {len(appendix.pages)}"
        )
    for index, page in enumerate(appendix.pages, start=1):
        _assert_size(page, A4_POINTS, f"appendix page {index}")
    page_texts = [" ".join((page.extract_text() or "").split()) for page in appendix.pages]
    if not page_texts[1].startswith("1 Research question and scope"):
        raise RuntimeError("appendix Section 1 must start on page 2")
    if not page_texts[-3].startswith("References"):
        raise RuntimeError("appendix references must start on their own page")
    required_appendix_text = [
        "Method and Results Appendix",
        "Primary validation results",
        "Author contributions and use of AI tools",
        "Canva",
        "GLM",
        "OpenAI Codex",
        "References",
    ]
    if not signed:
        required_appendix_text.append("Declaration of Academic Integrity")
    appendix_text = _verify_text(
        appendix,
        tuple(required_appendix_text),
        "appendix",
    )
    status = (appendix.metadata or {}).get("/DeclarationStatus")
    if signed and status != "user-supplied-signed-forms":
        raise RuntimeError("signed mode requires the two supplied declaration pages")
    if not signed and status == "user-supplied-signed-forms":
        raise RuntimeError("signed declarations were supplied; rerun with --signed")
    appendix_fonts = _verify_fonts(appendix, "appendix")

    print(
        "Submission verified: exactly two PDFs; poster=1 A1 page, "
        f"{poster_fonts} embedded fonts, {len(poster_text)} text characters, "
        f"{figure_count} figure PNGs >=150 PPI, {traced_values} traced result values; "
        f"appendix={len(appendix.pages)} A4 pages, "
        f"{appendix_fonts} embedded fonts, {len(appendix_text)} text characters."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--signed",
        action="store_true",
        help="require two author-supplied declaration pages instead of the unsigned forms",
    )
    arguments = parser.parse_args()
    verify(Path(__file__).resolve().parents[1], signed=arguments.signed)


if __name__ == "__main__":
    main()
