# Guidance and academic-integrity review

Reviewed 29 September 2026 against the two-page professor document
`Poster-Examination-Guidelines.pdf` supplied by the student. Its linked official
form is the [University of Trier Declaration of Academic Integrity](https://www.uni-trier.de/fileadmin/organisation/ABT2/HPA/BA-MA-Arbeit/Eigenstaendigkeitserklaerung_EN.pdf)
(June 2025). The course document controls the poster submission; the university
form controls what each author must personally declare and sign.

| Topic | Available guidance and project evidence | Current action |
|---|---|---|
| Empirical NLP work | The professor requires empirical NLP work related to the module. | WiC validation experiments compare frozen GloVe and BERT representations; saved scores and predictions support the displayed result. |
| Mandatory poster sections | Hypothesis, Methodology, and Results, with context, method justification, plots or tables, interpretation, and limitations. | All three named sections are present. The result bars display validation accuracy; the text gives a paired interval and the 210 BERT errors. |
| Reproducibility | Technical work needs an external repository link. | The poster gives the GitHub repository URL; scripts, configuration, selection ledger, and saved results are committed. |
| Poster format and legibility | A1 (594 x 841 mm), paragraph type at least 24 pt, pixel figures at least 150 PPI, sufficient contrast and meaningful layout. | The poster source sets A1 portrait and 24 pt body type. Its chart and logo are vectors. The release verifier checks dimensions, fonts, figure previews, and the source type setting. |
| Appendix references | A complete list, mainly academic papers; websites alone do not qualify. | The appendix has research-paper references, DOIs, and the dataset source. |
| AI-use transparency | The linked form requires product names, a full list in the appendix, explicit marking of generated passages, and use only within the examiner's written permission. | The appendix describes GLM, Canva, and OpenAI Codex by role. The exact GLM product/version and any other earlier coding assistant still need author confirmation. The new AI-assisted disclosure prose and poster closing sentence are identified. The student reports written permission, but its scope has not been reviewed. |
| Academic declaration | A signed copy of the official linked form is mandatory. | Two official copies are appended with names, IDs, title, programme, and module prefilled. Examiner, place/date, and signatures remain for the authors; unsigned copies do not meet the requirement. |
| Submission package | Exactly two PDFs in one ZIP, named `1910474_1911272.zip`, uploaded to the group's STUD.IP posters folder by 30 September 2026 end of day Central European time. | The repository stages only `poster.pdf` and `appendix.pdf`. Make the final ZIP only after the signed forms are inserted and checked. |

## How to write the final account clearly

Describe what was done, where each tool was used, and what the authors checked.
Use ordinary, specific language: "We selected the WiC split and froze the
threshold before validation" is easier to verify than a broad claim that the
project was completed independently. Keep draft versions, the selection ledger,
and result files so decisions can be explained. Correct inaccurate or overly
polished phrasing for clarity; do not edit merely to obtain a detector score.

AI-writing detectors classify patterns in text; they do not reconstruct who
made a research decision. [Turnitin's own guidance](https://guides.turnitin.com/hc/en-us/articles/27139000787853-How-should-I-review-the-AI-Writing-report)
says its indicator is one data point rather than a definitive verdict. A
[published study](https://doi.org/10.1016/j.patter.2023.100779) found that
several detectors misclassified substantial numbers of essays by non-native
English writers. These findings support transparent process records and careful
human review, not a promise that a particular style will avoid detection.

## Remaining author checks

1. Compare the account with both authors' actual tool history. Name the exact
   earlier coding assistant and GLM product/version, and identify any further
   AI-generated passages if applicable.
2. The student reports that written examiner permission exists. Check that its
   scope covers the specific declared uses before signing; the permission text
   has not been reviewed in this repository.
3. Complete one official form per author, sign and date both, combine them into
   a two-page PDF, and build with `--declaration` followed by `--signed` verification.
4. Visually inspect both signed pages, run `scripts/package_submission.py` to
   make `1910474_1911272.zip` with exactly the two final PDFs, and upload it to
   the correct STUD.IP seminar group before the deadline.
