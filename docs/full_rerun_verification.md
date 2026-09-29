# Full experiment rerun, 29 September 2026

This check reran the code from branch `docs/academic-integrity-disclosure` at
commit `208c6f712d68f5982304a6e173772e5c3c194ae6` on macOS arm64. Python
3.11.16, PyTorch 2.11.0, Transformers 5.17.0, and NumPy 2.4.6 came from the
frozen `uv.lock`. The original committed run used a Windows laptop with an
NVIDIA GeForce RTX 3050 Ti GPU. This rerun used the CPU, with no cached model
representations on its first pass.

## Inputs and commands

The official SuperGLUE WiC archive passed the checksum and split checks. It
contains 5,428 labelled training pairs, 638 labelled validation pairs, and
1,400 unlabelled test pairs. The GloVe archive matched the SHA-256 pinned in
`configs/default.yaml` exactly. It was fetched from a mirror because the
Stanford server was slow; the byte match means the experiment used the same
archive. BERT loaded at the exact model revision in that configuration. No
test labels or test score were used.

From the project root, the successful commands were:

```text
uv run --frozen python scripts/download_data.py
uv run --frozen pytest -q                      # 66 passed
uv run --frozen ruff check .                    # all checks passed
uv run --frozen python scripts/run_experiment.py
uv run --frozen python scripts/run_quick_test.py
uv run --frozen python scripts/run_experiment.py  # cached repeat
uv run --frozen python scripts/analyse_results.py
uv run --frozen python scripts/generate_figures.py
uv run --frozen python scripts/build_submission.py --compile --stage --tectonic PATH
uv run --frozen python scripts/verify_submission.py
```

## Comparison with the committed run

The training-only selection again chose a two-word window on either side for
GloVe and the mean of BERT's last four layers. The rerun's primary counts were:

| Condition | Correct of 638 | Accuracy | Macro F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| GloVe target | 319 | 0.5000 | 0.3333 | 0.5000 |
| GloVe with nearby words | 354 | 0.5549 | 0.5511 | 0.5800 |
| BERT mean last four | 428 | 0.6708 | 0.6673 | 0.7163 |

All 638 validation labels, predictions, and correctness flags for the three
primary conditions matched the committed files. Their confusion matrices,
bootstrap intervals, paired differences, and error partition were unchanged.
The new BERT cosine scores differed from the GPU scores by no more than
0.00000254 in the raw score table; the selected BERT threshold moved by
0.0000000693. No primary classification crossed its threshold. The only
secondary metric difference was a layer-9 ROC-AUC change of 0.00000983, which
does not change its displayed four-decimal value or any conclusion.

After the uncached run, the cached full run reproduced every scored output
file byte for byte. Only the environment and cache-status JSON files changed.
The committed GPU output files are retained so the PR does not replace 6,066
raw-score rows with harmless hardware rounding differences. Anyone can rerun
the commands above to produce the CPU values locally.

The rebuilt poster and appendix passed `verify_submission.py`: one A1 poster,
seven A4 appendix pages, embedded fonts, 150-PPI-or-better figure previews,
and 11 poster numbers traced to the result CSVs. The two official declaration
pages remain unsigned and require each author's personal completion before
submission.
