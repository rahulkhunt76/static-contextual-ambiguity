# Experiment design

## Research question

Given two contexts containing the same ambiguous target word, how effectively can static and contextual word representations distinguish whether the target expresses the same meaning or a different meaning?

The operational task is deliberately narrow: represent the marked occurrence in each sentence, take cosine similarity, learn a scalar decision threshold from training data, and predict the binary WiC label. Higher similarity means “same meaning.” No condition is fine-tuned on WiC.

## Hypotheses

**H1 — contextual target.** A target representation from pretrained BERT should separate same-sense and different-sense pairs more effectively than a single context-independent GloVe target vector because BERT conditions each occurrence on its sentence.

**H2 — lexical context helps static vectors.** Averaging GloVe vectors around the marked occurrence should outperform the target-only diagnostic, but BERT's contextual target should retain an advantage because the static average loses order and composition.

These are predictions, not conclusions. They will not be reworded after the validation results are seen.

## Data

The project uses the WiC portion of the official SuperGLUE v2 archive:

- Source: `https://dl.fbaipublicfiles.com/glue/superglue/data/v2/WiC.zip`
- Upstream license: CC BY-NC 4.0, as stated by the original WiC site
- Train: 5,428 pairs; 2,714 false and 2,714 true
- Validation: 638 pairs; 319 false and 319 true
- Test: 1,400 pairs; no labels in the SuperGLUE release

Each JSONL row contains the lemma, two sentences, a unique split-local ID, two half-open character spans (`start1:end1`, `start2:end2`), and—outside test—the boolean same-sense label. The marked surface often differs morphologically from the lemma: 1,852 training and 251 validation pairs contain at least one such difference. The initial audit found 93 training contexts and one validation context in which the marked surface string occurs more than once. It also found four training pairs duplicated with sentence order reversed; they are retained as part of the official split and reported in the audit. Those observations rule out string search as an alignment strategy and make duplicate handling explicit.

The unlabeled SuperGLUE test split will be downloaded and validated structurally but not evaluated. The poster will call the 638 labeled examples “validation” or “held-out validation,” never “test.” The original WiC site separately offers a v1.0 package with test gold labels, but mixing that release with SuperGLUE v1.1 offsets would muddy provenance. This study uses one release consistently.

## Split protocol and leakage controls

The final validation labels are untouched until every primary-condition choice is frozen.

1. Stratify the official training split with seed 2026 into `tune_train` (80%, 4,342 examples) and `tune_holdout` (20%, 1,086 examples), preserving 50/50 labels.
2. For each static-context window and each BERT pooling candidate, fit its threshold on `tune_train` only.
3. Rank candidates by macro F1 on `tune_holdout`; break exact ties by accuracy, then ROC-AUC, then the simpler configuration declared below.
4. Freeze one context window and one primary BERT representation.
5. Refit each primary system's threshold on all 5,428 official training pairs.
6. Evaluate the three frozen systems once on the 638 validation pairs.
7. For the prespecified layer analysis, fit a separate threshold for every BERT layer on all official training pairs and evaluate every layer on validation. This analysis does not redefine the primary BERT result.

Before splitting, the loader checks unique IDs, exact duplicate task rows, reversed-order task pairs, class balance, valid spans, and missing fields. Reversed pairs inside official train are counted rather than silently removed. A task-equivalent pair crossing train and validation emits a hard leakage error.

## Conditions

### Static Target (diagnostic)

Model: GloVe 6B, 300 dimensions.

Both contexts receive the same lowercase lemma vector. When present, the pair cosine is therefore approximately 1.0 by construction. The condition demonstrates meaning conflation; it is not presented as a competitive contextual classifier. An OOV target is logged. If an OOV pair occurs, its score is assigned the training-set median valid score and coverage is reported.

### Static Context

Model: the same GloVe vectors.

The sentence is tokenised with deterministic regex word spans. The token overlapping the exact WiC target character span is excluded. Usable lowercase GloVe vectors in a symmetric token window are averaged. Prespecified candidates are ±2, ±5, ±10, and the whole sentence. If a side has no usable context vector, the encoder falls back to the target lemma vector and records the fallback. If the lemma is also OOV, the pair uses the training-set median valid similarity and remains in the denominator.

This baseline can use lexical context but discards word order, syntax, and composition. It is the meaningful static comparator.

### Contextual BERT Target

Model: `google-bert/bert-base-uncased`, used without task fine-tuning.

The fast tokenizer returns offset mappings. A WordPiece belongs to the target when its non-special half-open span overlaps the WiC character span:

```text
token_end > target_start AND token_start < target_end
```

All aligned WordPieces are mean-pooled. One inference pass stores layers 1–12. The two prespecified primary candidates are the final layer and the mean of the last four layers; selection uses only the internal training split. The model runs in evaluation and inference modes. Failed alignments are errors with sample IDs and offsets, never guessed tokens.

## Similarity and classification

For every pair and condition:

```text
score = cosine(representation(sentence 1), representation(sentence 2))
prediction = score >= frozen_threshold
```

Threshold candidates are score midpoints plus boundaries that permit all-negative and all-positive predictions. The objective is macro F1. Ties are resolved by accuracy, then the threshold closest to 0.5, then the smaller numeric threshold. The deterministic rule is tested, saved, and applied unchanged to validation.

## Metrics and uncertainty

The complete result table reports:

- accuracy;
- macro F1;
- precision and recall for the positive (“same meaning”) class;
- ROC-AUC with positive label = true;
- confusion-matrix counts in `[[TN, FP], [FN, TP]]` order;
- evaluated-pair count and representation coverage.

Accuracy, macro F1, and ROC-AUC receive paired non-parametric bootstrap 95% percentile confidence intervals using 1,000 resamples of validation pair indices and seed 2026. System differences use the same resampled indices. A difference is described as statistically distinguishable by this interval only if its 95% interval excludes zero; otherwise the report says the intervals are compatible with no difference. “Significant” is reserved for an explicitly reported inferential result.

## Secondary analysis

- **BERT layers:** layers 1–12, each with a train-fitted threshold and validation macro F1/ROC-AUC.
- **Score geometry:** same-sense versus different-sense similarity distributions for Static Context and BERT.
- **Errors:** BERT correct/static wrong, static correct/BERT wrong, all wrong, high-confidence BERT errors, threshold-borderline cases, and static successes.
- **Ambiguity type:** qualitative only. WiC does not provide defensible homonym/polysemy labels, so no post-hoc taxonomy will be invented.

## Reproducibility and resource plan

- Python 3.11 is managed by `uv`; dependencies and transitive versions are locked.
- Default seed is 2026 for Python, NumPy, PyTorch, splitting, and bootstrap.
- CUDA is preferred when available, with adaptive batching for the 4 GB RTX 3050 Ti and a clean CPU path.
- Logs record Python, Torch, Transformers, CUDA availability, GPU name, full configuration, Git commit, and dirty state.
- Dataset archives, GloVe files, model weights, and embedding caches are not committed.
- Expensive BERT layer outputs are cached by data checksum, split, model identifier/revision, max length, pooling schema, and cache schema version.
- Every poster value must come from `results/final/metrics.csv`, `bootstrap_ci.csv`, `predictions.csv`, `layer_metrics.csv`, or a figure generated from those files.

## Validity threats fixed before results

- WiC's binary boundary simplifies graded lexical meaning.
- GloVe context averaging is a deliberately simple compositional baseline.
- A global threshold may work differently for words with different similarity ranges.
- The experiment covers English, one static family, one contextual model, and cosine-based classification.
- Pretrained feature extraction tests encoded information, not the ceiling achievable through supervised fine-tuning.
- The internal train split introduces selection variance; it protects the final validation set at the cost of fewer examples during candidate selection.

## Stop conditions

Do not trust or publish results if span validation fails, labels are reversed, train/validation overlap is detected, thresholds touch validation labels, BERT is not in inference mode, cache metadata do not match, or the final numbers cannot be regenerated from a clean locked environment.
