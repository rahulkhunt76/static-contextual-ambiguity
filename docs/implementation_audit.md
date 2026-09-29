# Implementation and claim check

Checked on 29 September 2026. This note links the main statements in the
poster and appendix to the code and saved experiment files. It is a check of
the current repository, not a new model run.

| Claim in the PDFs | Where it is implemented or recorded |
|---|---|
| WiC has 5,428 training, 638 validation, and 1,400 unlabeled test pairs. | `lexical_ambiguity/data.py` validates the three splits. `data/processed/dataset_audit.json` records their counts and target-span checks. The test split has no local score. |
| GloVe uses one lemma vector for the target diagnostic and averages nearby words for the context condition. | `lexical_ambiguity/embeddings/glove.py` defines both encoders. `lexical_ambiguity/experiment.py` scores the target condition as 1 when the lemma exists and tries the four configured context windows. `results/final/run_diagnostics.json` records 18 missing training target scores and one target fallback per context window in training. |
| BERT uses the marked target, stays frozen, and compares its last layer with the mean of its last four layers. | `lexical_ambiguity/embeddings/bert.py` aligns word pieces by character offsets, calls `model.eval()` and `torch.inference_mode()`, and collects all 12 layers. `configs/default.yaml` names the two primary candidates. There is no optimizer or weight-update step in the experiment. |
| Model settings and thresholds come from training data. | `lexical_ambiguity/experiment.py` uses a 4,342/1,086 training split, writes `results/raw/selection_ledger.json`, and then scores validation. Validation text is read earlier for data checks and vocabulary; its labels are not used to choose the models or thresholds. |
| The poster's results are 319, 354, and 428 correct out of 638. | `results/final/metrics.csv` contains the confusion counts. `scripts/build_submission.py` generates the poster numbers and appendix tables from saved result files. `scripts/verify_submission.py` checks 11 displayed poster values. |
| The paired comparison and error counts use the same 638 examples. | `lexical_ambiguity/evaluation/bootstrap.py` uses paired resamples; `results/final/paired_differences.csv` stores the intervals. `results/final/predictions.csv` gives 250 both correct, 178 BERT only, 104 GloVe context only, and 106 both wrong. |

The current checkout passes all 66 tests, Ruff, the PDF build, and the
submission verifier. A separate check recomputed the split sizes, model
choices, correctness counts, and error groups from the saved JSON and CSV
files. The large WiC and GloVe downloads, model weights, and inference cache
are absent from this checkout, so the full 5,428/638 inference was not rerun
here. The saved result files and prior run record remain in the repository.
