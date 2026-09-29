# Data provenance

`scripts/download_data.py` downloads the WiC portion of the official SuperGLUE v2 release, verifies the frozen archive checksum, extracts only the three expected JSONL files, validates their spans/labels/counts, and writes `processed/dataset_audit.json`.

The WiC dataset is licensed under CC BY-NC 4.0 by its authors. Raw data are deliberately ignored by Git; reproduce them with:

```powershell
uv run python scripts/download_data.py
```

SuperGLUE's `test.jsonl` has no labels. This project validates it structurally but never reports a local test score.

The GloVe 6B archive is downloaded from Stanford and is likewise ignored by Git.
The download script checks both archives before using them.
