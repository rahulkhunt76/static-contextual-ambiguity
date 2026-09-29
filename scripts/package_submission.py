"""Package the two verified PDFs after both official forms have been signed."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from verify_submission import verify

ROOT = Path(__file__).resolve().parents[1]
ZIP_NAME = "1910474_1911272.zip"


def main() -> None:
    try:
        verify(ROOT, signed=True)
    except RuntimeError as error:
        raise SystemExit(f"Cannot package submission: {error}") from error
    output = ROOT / ZIP_NAME
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for name in ("poster.pdf", "appendix.pdf"):
            archive.write(ROOT / "submission" / name, arcname=name)
    print(f"Created {output} with exactly poster.pdf and appendix.pdf")


if __name__ == "__main__":
    main()
