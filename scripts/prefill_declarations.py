"""Fill known identity fields on the official Trier forms; leave signatures blank.

This optional authoring step needs reportlab and pypdf. The resulting PDFs are
committed so the normal poster build does not need reportlab installed.
"""

from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, NameObject
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "appendix/official_declaration_en.pdf"
TITLE = "Static vs Contextual Embeddings for Lexical Ambiguity"
PROGRAMME = "MSc Natural Language Processing"
MODULE = "Trends in Natural Language Processing"
FONT_CANDIDATES = (
    Path(os.environ.get("TRIER_DECLARATION_FONT", "")),
    Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    Path("C:/Windows/Fonts/arial.ttf"),
    Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
)
AUTHORS = (
    ("Choudhary Prashant Santosh", "1910474"),
    ("Rahul Khunt", "1911272"),
)


def prefill(name: str, student_id: str) -> Path:
    reader = PdfReader(OFFICIAL)
    page = reader.pages[0]
    width, height = float(page.mediabox.width), float(page.mediabox.height)
    overlay = BytesIO()
    font = next((path for path in FONT_CANDIDATES if path.is_file()), None)
    if font is None:
        raise RuntimeError("set TRIER_DECLARATION_FONT to an embeddable TTF font")
    pdfmetrics.registerFont(TTFont("FormArial", str(font)))
    ink = canvas.Canvas(overlay, pagesize=(width, height))
    ink.setFont("FormArial", 10)
    ink.drawString(165, height - 211, name)
    ink.drawString(165, height - 239, student_id)
    ink.drawString(132, height - 409, TITLE)
    ink.drawString(145, height - 433, PROGRAMME)
    ink.setLineWidth(0.9)
    ink.line(72, height - 536, 78, height - 530)
    ink.line(72, height - 530, 78, height - 536)
    ink.setFont("FormArial", 8.5)
    ink.drawString(340, height - 536, MODULE)
    ink.save()
    overlay.seek(0)
    page.merge_page(PdfReader(overlay).pages[0])
    # ReportLab adds an unused Helvetica text state before the embedded Arial
    # glyphs. Remove both that empty state and its unembedded font resource.
    data = page.get_contents().get_data()
    empty_state = b"BT\n/F1 12 Tf\n14.4 TL\nET\n"
    if data.count(empty_state) != 1:
        raise RuntimeError("unexpected declaration overlay font setup")
    stream = DecodedStreamObject()
    stream.set_data(data.replace(empty_state, b"", 1))
    page[NameObject("/Contents")] = stream
    fonts = page["/Resources"]["/Font"].get_object()
    if fonts["/F1"].get_object().get("/BaseFont") != "/Helvetica":
        raise RuntimeError("unexpected declaration overlay font resource")
    del fonts[NameObject("/F1")]
    writer = PdfWriter()
    writer.add_page(page)
    output = ROOT / f"appendix/declaration_{student_id}_unsigned.pdf"
    with output.open("wb") as handle:
        writer.write(handle)
    return output


if __name__ == "__main__":
    for author_name, author_id in AUTHORS:
        print(prefill(author_name, author_id))
