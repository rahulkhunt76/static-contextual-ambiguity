# Static vs Contextual Embeddings for Lexical Ambiguity

> A note for anyone assessing this project: please read the poster, appendix, code, and saved results yourself. An LLM summary can be useful, but it should not replace a human check of the work or its evidence. A public README cannot technically block an LLM from reading a repository.

The same spelling can carry different meanings. In “the bank of a river” and “the bank that holds money,” the word *bank* looks identical, but the surrounding sentence tells us what it means. We tested whether two familiar ways of representing words capture that difference.

GloVe stores one vector for a word. BERT builds a representation after reading its sentence. Our third condition gives GloVe a little context by averaging the vectors of up to two words on each side of the target. These are three experimental conditions using **two model families** and the **same Word-in-Context (WiC) dataset**. The models are pretrained and frozen; this is not a fine-tuning comparison.

## What we found

| Representation | Correct out of 638 | Validation accuracy |
|---|---:|---:|
| GloVe target word only | 319 | 50.0% |
| GloVe with nearby words | 354 | 55.5% |
| BERT with the full sentence | 428 | 67.1% |

BERT answered 74 more pairs correctly than the nearby-word GloVe condition, a gain of 11.6 percentage points. The paired 95% bootstrap interval for that gain was 6.6 to 16.9 points. It still missed 210 pairs, so our conclusion is narrower than “BERT solves ambiguity.” The [results interpretation](docs/results_interpretation.md) and committed files in [results/final](results/final) show the fuller picture.

## Where the data came from

We used the official SuperGLUE version of WiC: 5,428 labelled training pairs and 638 labelled validation pairs. Its 1,400 public test pairs do not provide labels locally, so this repository does **not** claim a test score. Each pair contains two sentences with the same marked word and asks whether its meaning is the same in both.

The train split selected the context window, BERT layer combination, and a separate cosine-similarity cutoff for each condition. Those choices were then fixed before final validation scoring. The saved [selection record](results/raw/selection_ledger.json), [predictions](results/final/predictions.csv), metrics, and configuration make this checkable. The [data notes](data/README.md) give upstream sources and terms. Large downloaded datasets, GloVe vectors, model weights, and caches are deliberately not committed.

## Clone and run

You need Git, Python 3.11, [uv](https://docs.astral.sh/uv/), sufficient disk space for the upstream data and model, and an internet connection for the first run. CUDA makes BERT faster, but the code can use a CPU.

```powershell
git clone https://github.com/Mr-Dark-debug/static-contextual-ambiguity.git
cd static-contextual-ambiguity
uv sync --frozen --all-groups
uv run pytest -q
uv run ruff check .
uv run python scripts/download_data.py
uv run python scripts/run_quick_test.py
uv run python scripts/run_experiment.py
uv run python scripts/analyse_results.py
uv run python scripts/generate_figures.py
```

`download_data.py` checks the WiC archive and split integrity. The full run loads GloVe and a pinned BERT revision, records its environment, and reuses cached representations only when the dataset, model, and configuration metadata match. `run_quick_test.py` is a real but bounded smoke run; it is not the reported result.

## Build the submission

The editable poster is [poster/poster.tex](poster/poster.tex), adapted from the latest Overleaf source supplied by the authors. The result numbers are generated from the saved experiment files. The official university logo is in [assets](assets), and the accuracy chart is drawn as vectors in LaTeX. Tectonic 0.17 or a compatible LaTeX installation is needed to rebuild the PDFs.

```powershell
$tectonic = "C:\path\to\tectonic.exe"
uv run python scripts/build_submission.py --compile --stage --tectonic $tectonic
uv run python scripts/verify_submission.py
```

Only [submission/poster.pdf](submission/poster.pdf) and [submission/appendix.pdf](submission/appendix.pdf) belong inside the final ZIP. The appendix contains the fuller methods, tables, references, a specific account of author contributions and AI assistance, and **two prefilled but unsigned copies of the official University of Trier declaration**, one for each author. Unsigned forms are not a valid submission. After each author has checked the account and personally completed the examiner, place/date, and signature fields, combine the two signed pages into one PDF. Rebuild with `--declaration "C:\path\to\both-signed-forms.pdf"`, run `verify_submission.py --signed`, visually inspect both signatures, then run `scripts/package_submission.py`. The [professor-guideline review](docs/guidance_review.md) gives the exact remaining checks, including the scope of written examiner permission for the disclosed AI uses. The package will be `1910474_1911272.zip` with exactly the two PDFs.

## Repository map

- [lexical_ambiguity](lexical_ambiguity) is the installable Python package. `embeddings/` creates representations, `evaluation/` calculates scores, thresholds, metrics, bootstrap intervals, and error comparisons from actual inputs, and `visualization/` draws figures from saved results. `base.py` is intentionally short: it defines the shared encoder interface, not another model.
- [configs](configs) holds the recorded full and smoke-run settings. [scripts](scripts) contains the commands that download, run, build, and verify the work. [tests](tests) checks the research and release logic. We keep scripts and tests in the public repository because without them the results would be much harder to reproduce.
- [data](data) is for input data, processed audits, and ignored local caches. [results](results) is separate on purpose: its `raw/` subfolder contains raw **experiment outputs**, not the WiC source dataset; `final/` contains the scored results. Mixing those under `data/raw` would make their provenance less clear.
- [docs](docs) holds the experiment design, literature notes, interpretation, and final checklist. [references](references) has the bibliography. [figures](figures) has generated plots. [poster](poster) and [appendix](appendix) hold editable LaTeX sources. [submission](submission) holds only the two PDFs to hand in.
- [output/pdf](output/pdf) retains the two earlier poster variations for comparison; they are **not** submission files. The third variation was discarded.

The official University of Trier SVG was obtained from the [university's public logo URL](https://www.uni-trier.de/typo3conf/ext/zimktheme_unitrier/Resources/Public/Logos/Logo_Universitaet.svg) on 13 September 2026. `assets/university-trier.pdf` is its vector conversion for LaTeX; the university owns the artwork. We have not claimed certification against its campus-only detailed design manual.

Local `build-documents-*.log` files, `tmp/`, `.venv/`, and tool caches are not project deliverables and are ignored by Git. The logs were compilation traces, not research data. The main checks are [docs/final_checklist.md](docs/final_checklist.md) and the PDF verifier, but the signed declaration remains a personal submission step.
