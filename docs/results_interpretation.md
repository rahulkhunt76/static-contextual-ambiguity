# Results interpretation

## Direct answer

Under the frozen WiC protocol, the unfine-tuned BERT target representation
outperformed both GloVe systems. Mean-last-four BERT reached 67.08% accuracy and
66.73 macro F1 on 638 validation pairs. The train-selected ±2 GloVe context
average reached 55.49% accuracy and 55.11 macro F1. The static target diagnostic
is deliberately context-free: its two vectors are identical, every similarity is
exactly 1, and it therefore predicts one class at 50% accuracy and 33.33 macro F1.

The paired accuracy difference for GloVe context minus BERT was -11.60 percentage
points, with a fixed-seed 95% bootstrap interval of [-16.93, -6.58]. The macro-F1
difference was -11.61 points, interval [-16.90, -6.64]. Both intervals exclude
zero. This supports the narrower claim that contextualized target vectors are
more useful than this bag-of-nearby-GloVe-words baseline on this validation set;
it does not establish a universal ranking of static and contextual models.

## What selection found

All choices were made on the official 5,428-pair training split. On its fixed
4,342/1,086 tuning partition, ±2 had the highest static-context holdout macro F1
(0.5961), narrowly ahead of ±5 (0.5928), ±10 (0.5863), and the full sentence
(0.5845). BERT mean-last-four reached 0.7247, versus 0.7145 for layer 12. The
selection record was written before validation scoring and is saved with the
experiment outputs.

The secondary layer analysis rises broadly from layer 1 (57.05% validation
accuracy) through layer 9 (66.61%), then declines to 63.64% at layer 12. Layer 10
has the highest individual-layer ROC-AUC (0.7197). These validation observations
help explain the prespecified mean-last-four result, but they were not used to
replace the primary model after the fact.

## Error structure

BERT was correct on 178 pairs that the selected static context missed; GloVe
context was correct on 104 pairs that BERT missed. Both primary systems were
correct on 250 pairs and both were wrong on 106. Of those shared errors, 54 also
defeated the one-class target diagnostic.

The systems show opposite class skews. BERT recovered 77.43% of same-sense pairs
but only 56.74% of different-sense pairs. GloVe context recovered 64.58% of
different-sense pairs and 46.39% of same-sense pairs. This is a thresholded
validation observation, not evidence that one representation intrinsically
prefers a class.

Among BERT's 210 errors, 33 lie within 0.025 cosine units of its frozen threshold;
53 are in the top error-margin quartile. The high-margin cases show why aggregate
accuracy is not the whole story:

- BERT strongly separated *ball* / *balls* in “base of the thumb” versus “balls
  of his feet,” although WiC marks the pair same-sense.
- It assigned high similarity to different-sense pairs such as *engrave a
  letter* versus *engrave a pen* and *map the genes* versus *map the surface*.
- Conversely, BERT correctly linked sparse same-sense pairs such as *rust
  remover* / *paint remover*, *bean curd* / *lemon curd*, and the Nile /
  Mississippi *delta*, where lexical context overlap is low and GloVe context
  failed.

These examples are observations from the saved rows. Plausible explanations—fine
sense granularity, constructional similarity, or contextual over-specialization—
remain hypotheses because this experiment does not manipulate those factors.

## Robustness and limitations

- Validation has 638 balanced examples; the reported intervals quantify sampling
  variability under example resampling, not model or dataset-design uncertainty.
- BERT is used only as a frozen feature extractor. No fine-tuning result is
  implied.
- GloVe context is an unweighted token mean. A learned static classifier,
  subword-aware static model, or syntactic context could behave differently.
- Surface forms differ from the supplied lemma in 251 validation pairs. BERT
  accuracy is descriptively lower for those rows (63.75% versus 69.25%), while
  GloVe context is higher (60.56% versus 52.20%); these groups are not randomized
  and differ on other properties.
- The label-hidden 1,400-example SuperGLUE test split is not reported. Calling the
  validation result a test-set score would be inaccurate.
- Four symmetric duplicates exist within training, none within validation, and no
  sentence-pair overlap was found across train and validation. They were retained
  because they are part of the checksummed official release and never cross the
  final evaluation boundary.

All numeric claims above trace to `results/final/metrics.csv`,
`bootstrap_ci.csv`, `paired_differences.csv`, `layer_metrics.csv`, and
`analysis_summary.json`; quoted cases trace to `results/examples/error_cases.csv`.
