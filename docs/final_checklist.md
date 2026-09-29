# Final reproducibility and submission checklist

This checklist distinguishes completed project gates from the personal metadata that the
students must supply. Experiment evidence dates to 10 September 2026. Poster design
and artifact checks were refreshed on 29 September 2026.

## Experiment and evidence

- [x] Python is constrained to 3.11 and all direct/transitive dependencies are frozen in
  `uv.lock`.
- [x] The official SuperGLUE v2 WiC archive, expected members, split sizes,
  labels, and every half-open character span validate.
- [x] Target alignment selects every non-special WordPiece whose tokenizer offset overlaps
  the supplied target span; failures raise an error instead of guessing.
- [x] Model/window choice and all classification thresholds use official training data
  only. The selected configurations are saved in
  `results/raw/selection_ledger.json`.
- [x] The full 5,428-train / 638-validation experiment completed with the frozen GloVe and
  BERT revisions. The unlabeled 1,400-row test split has no reported score.
- [x] Accuracy, macro F1, positive-class precision/recall, ROC-AUC, confusion counts,
  paired 1,000-resample bootstrap intervals, layers 1--12, and row-level predictions are
  committed.
- [x] Qualitative examples come from saved prediction rows; interpretations are explicitly
  descriptive rather than causal.
- [x] A clean locked cached reproduction regenerated the full artifacts.

## Code and artifact gates

- [x] `uv sync --frozen --all-groups`
- [x] `uv run pytest -q` (65 tests after replacing the obsolete poster-copy checks)
- [x] `uv run ruff check .`
- [x] `uv run python scripts/run_quick_test.py`
- [x] `uv run python scripts/run_experiment.py` (full cache-integrity reproduction)
- [x] `uv run python scripts/analyse_results.py`
- [x] `uv run python scripts/generate_figures.py`
- [x] `uv run python scripts/build_submission.py --compile --stage --tectonic PATH`
- [x] `uv run python scripts/verify_submission.py`

The current verifier confirmed one 1683.78 x 2383.94 point A1 poster page, seven 595.28 x
841.89 point A4 appendix pages, extractable required text,
embedded fonts, six figure previews at or above 150 PPI, byte-identical staged copies, and exactly
`poster.pdf` plus `appendix.pdf` in `submission/`. Eleven headline/result values are also
recomputed from the committed CSV/JSON files and matched against extracted poster text.
The current canonical poster build contains no overfull-box or unresolved-reference
warning. Normal underfull-line warnings remain in narrow poster columns.
This revision ran all 66 tests and Ruff successfully. It also rebuilt and
verified both submission PDFs against the saved experiment results. The full
model inference was not rerun in this checkout because the large upstream data,
GloVe vectors, and BERT weights are not stored in the repository.

## Human PDF review

- [x] Poster reviewed as a full-page raster: centered header, balanced flowing columns,
  24 pt body text, vector accuracy chart, paragraph spacing, and margins are readable
  with no clipping. The redundant pairwise-outcome chart was removed.
- [x] The appendix and both prefilled unsigned official declaration pages were
  rerendered and inspected.
- [x] Poster visibly contains **Hypothesis**, **Methodology**, and **Results**.
- [x] The authors' latest Overleaf text is the canonical source in `poster/poster.tex`.
  It runs Introduction, Inspiration, Background, Hypothesis, Methodology, Results, and
  Conclusion across balanced columns. The headline results remain traceable to the
  saved experiment record.
- [x] Poster background/model claims use compact author-year citations. Complete
  verified references and DOI links are in the appendix.
- [x] Results use “validation” consistently and do not disguise it as a public test score.
- [x] The poster's detector joke and the extra closing sentence were removed.

## Required personal action before hand-in

- [x] Poster names and enrollment numbers match the details supplied by the user.
- [x] Official Trier logo is included as vector artwork with source recorded.
- [x] Requested limitations, success/warning, banner and bordered panels are removed.
- [x] Appendix author names, enrollment numbers, and repository link are personalized.
- [x] Check the professor's two-page project rules and linked official declaration;
  the requirement-by-requirement review is in `docs/guidance_review.md`.
- [ ] Confirm that the contribution and AI-use account names every tool actually
  used and accurately describes each author's work.
- [x] Append two copies of the linked official declaration form, one per author.
- [ ] Check the scope of the examiner's written permission for the disclosed AI
  uses. The student reports that permission exists, but the text was not supplied.
- [ ] Each author completes, signs, and dates their own official form.
- [ ] Rebuild with `--declaration`, then rerun `scripts/verify_submission.py --signed`.

The two prefilled official pages remain unsigned. The release script does not
recognize a signature automatically; each author must inspect the signed pages.
