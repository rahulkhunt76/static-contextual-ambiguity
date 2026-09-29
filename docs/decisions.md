# Decision log

This file records choices before the final validation results are opened. Later methodological fixes must append an entry describing the bug and every affected rerun.

## D001 — One consistent WiC release

**Decision:** Use the WiC task files from the official SuperGLUE v2 archive and report the 638-example labeled validation split as final evaluation.

**Reason:** This release supplies exact character offsets and has a checksum mirrored in the Hugging Face dataset metadata. Its 1,400-example test file has no labels, so it cannot support a local test metric.

**Alternatives considered:** The original WiC v1.0 package advertises test gold labels, and Hugging Face mirrors expose Parquet files. Mixing original v1.0 test data with SuperGLUE v1.1 rows risks provenance/version ambiguity; relying on a mutable mirror is less direct than the benchmark archive.

**Evidence:** Original WiC site and SuperGLUE archive inspection.

The validation audit also found four training examples repeated with the two sentence sides swapped. They remain in the official training split, are counted in `dataset_audit.json`, and are not present in validation. Removing them post hoc would change the named benchmark split for negligible benefit.

## D002 — Internal selection split

**Decision:** Use a deterministic stratified 80/20 split of official training data for candidate selection, then refit thresholds on all training data after choices are frozen.

**Reason:** It keeps window and pooling comparisons separate from both their threshold-fitting examples and the final validation labels.

**Alternatives considered:** Optimising and ranking on the same full training data is optimistic. Cross-validation uses data more efficiently but adds complexity and repeated threshold fits without changing the central research question.

**Evidence:** Prespecified protocol in `docs/experiment_design.md`.

## D003 — GloVe 6B 300d

**Decision:** Use the uncased 300-dimensional GloVe vectors trained on Wikipedia 2014 + Gigaword 5 (6B tokens).

**Reason:** They are the requested static model, small enough for a laptop, and widely documented. One vocabulary supports both the diagnostic and context average.

**Alternatives considered:** GloVe 840B has broader coverage but a much larger download and memory footprint. Word2Vec or FastText would change the requested comparison; FastText subwords would also blur the intended OOV behavior.

**Evidence:** Pennington, Socher, and Manning (2014) and the Stanford GloVe release documentation.

## D004 — Static context candidates

**Decision:** Compare symmetric ±2, ±5, ±10 token windows and whole-sentence averaging, excluding the exact target span. Do not add distance weighting unless all candidates fail a documented sanity check.

**Reason:** The candidates cover local to global lexical evidence while keeping one interpretable degree of freedom.

**Alternatives considered:** Distance weighting and learned attention could improve the baseline but would expand tuning and obscure the comparison.

**Evidence:** Project brief and the frozen selection protocol.

## D005 — BERT-base without fine-tuning

**Decision:** Use `google-bert/bert-base-uncased` as a frozen feature extractor.

**Reason:** The study asks whether pretrained contextual representations already encode useful sense distinctions on hardware with 4 GB VRAM. Fine-tuning would test a different question and introduce training choices.

**Alternatives considered:** BERT-large is closer to several published analyses but exceeds the comfortable hardware budget. RoBERTa or modern encoders would make the study less directly tied to the stated hypothesis.

**Evidence:** Devlin et al. (2019); Loureiro et al. (2021); the project hardware constraint.

## D006 — Character-offset alignment

**Decision:** Align all WordPieces whose offset overlaps the half-open target span, then mean-pool them.

**Reason:** Surface forms differ from lemmas and some contexts repeat the surface, making string or token-ID search unsafe. Overlap handles punctuation and subword splits without guessing.

**Alternatives considered:** Exact token equality fails on WordPieces and inflection. First-match search fails on repeated occurrences.

**Evidence:** Direct WiC archive audit and fast-tokenizer offset semantics.

## D007 — Primary BERT candidates

**Decision:** Compare final-layer target vectors with mean-last-four target vectors on the internal training split. Analyse all individual layers separately but do not let validation layer results redefine the primary model.

**Reason:** These are common, interpretable feature-extraction choices and can be cached from one pass.

**Alternatives considered:** Learned layer mixing or dozens of pooling schemes would overfit the study design.

**Evidence:** Loureiro et al. (2021), Haber and Poesio (2021), and the prespecified brief.

## D008 — Cosine, macro F1, and threshold policy

**Decision:** Use cosine similarity, a global threshold selected for macro F1, and deterministic tie-breaking.

**Reason:** Cosine tests representation geometry directly; macro F1 weights both balanced labels and remains informative if predictions collapse to one class.

**Alternatives considered:** A supervised classifier could exploit dimensions beyond cosine but would confound representation quality with classifier capacity. Accuracy-only tuning can hide class collapse.

**Evidence:** WiC task framing, RAW-C's cosine-based analysis, and the balanced observed labels.

## D009 — Missing static context

**Decision:** If no usable context vector remains on a side, fall back to that side's target lemma vector and record the event. If the target is also OOV, impute the median valid training similarity and report coverage.

**Reason:** Every held-out pair stays in the denominator, while empty/OOV context cannot silently disappear. Target fallback reduces to the diagnostic representation rather than inventing a vector.

**Alternatives considered:** Dropping pairs changes the evaluated population; a zero vector has undefined cosine; a random vector adds noise.

**Evidence:** Defined before GloVe results are inspected.

## D010 — Uncertainty

**Decision:** Use 1,000 fixed-seed paired bootstrap resamples and percentile 95% intervals for primary metrics and pairwise differences.

**Reason:** Resampling examples makes few distributional assumptions and preserves paired system predictions.

**Alternatives considered:** Unpaired intervals waste pairing; a single point estimate hides sampling variability.

**Evidence:** Prespecified project requirement.

## D011 — Frozen winners and final evaluation

**Decision:** Freeze the ±2-token GloVe context average and BERT mean-last-four
target representation after the internal 4,342/1,086 training split. Refit each
system's threshold on all 5,428 training pairs, then evaluate the 638 validation
pairs once.

**Reason:** The ±2 context candidate had the best internal-holdout macro F1
(0.5961), while mean-last-four had the best BERT macro F1 (0.7247). These choices
were recorded in `results/raw/selection_ledger.json` before validation scoring.

**Validation evidence:** Accuracy was 0.5000 for the deliberately context-free
GloVe target diagnostic, 0.5549 for GloVe ±2 context, and 0.6708 for BERT
mean-last-four. The paired bootstrap accuracy difference for GloVe context minus
BERT was -0.1160 with a 95% interval of [-0.1693, -0.0658]. These are empirical
results for this protocol, not general model rankings.
