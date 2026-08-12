# Module 5: Fine-Tuned Correctness Scorer (Experiment — Not Adopted)

## What This Is

An experiment, not a production module: could a small, locally-trainable
classifier learn to predict `llm_judge_correct` directly from a query +
generated answer + audit status, as a cheaper/faster alternative to the
LLM-as-judge grading used throughout Modules 2–4? This was tested once,
end to end, specifically to find out whether the training data collected
so far (`eval/calibration_dataset.json`, 49 labeled examples) is actually
sufficient to build something useful — and the answer is no, clearly and
informatively so.

## Data

- **Source:** `eval/calibration_dataset.json` (49 examples: 39
  `llm_judge_correct=True`, 10 `False` — an 80/20 imbalance).
- **Format:** each example rendered as
  `{"input": "Query: {query}\nAnswer: {final_answer}\nAudit status: {audit_status}", "label": 1 if correct else 0}`,
  saved to `eval/finetune_data.json`.
- **Split:** stratified 80/20 by label, fixed seed, saved to
  `eval/finetune_train.json` (39 examples: 31 label=1, 8 label=0) and
  `eval/finetune_val.json` (10 examples: 8 label=1, 2 label=0).

## Method

Given no GPU/cloud resources and a training set this small, a full LoRA
fine-tune or a small MLP would have more free parameters than the data
could meaningfully constrain — more likely to memorize noise than learn
signal. Instead: embed each `input` string with `BAAI/bge-small-en-v1.5`
(the same sentence-transformers model already used throughout this
pipeline — no new dependency) into a 384-dim vector, then train a
`scikit-learn` `LogisticRegression` classifier on top, with
`class_weight="balanced"` to actively counteract the 80/20 imbalance
rather than let the model trivially collapse to the majority class.

## Results

| Metric | Value |
|---|---|
| Overall validation accuracy | 60.0% |
| Majority-class baseline (always predict "correct") | **80.0%** |

**The trained classifier performs worse than doing nothing.** Per-class
breakdown, which is the metric that actually matters on an imbalanced set
(overall accuracy is misleading here — a model that always predicts
"correct" scores 80% while being useless for the one thing this classifier
would need to do: catch incorrect answers):

| Class | Precision | Recall | F1 |
|---|---|---|---|
| incorrect (0) | **0.000** | **0.000** | **0.000** |
| correct (1) | 0.750 | 0.750 | 0.750 |

Confusion matrix (rows = true label, cols = predicted):

```
              pred=incorrect  pred=correct
true=incorrect       0              2
true=correct          2              6
```

**The classifier caught zero of the 2 incorrect-answer validation
examples** — the entire reason to build it — and also misclassified 2 of
the 8 correct examples, which is why it landed below the majority-class
baseline despite the class-weighting actively working against a trivial
always-predict-correct collapse. Predicted probabilities all fell in a
narrow 0.47–0.60 band around the 0.5 decision boundary (never confidently
near 0 or 1 in either direction), indicating the model did not learn a
real separating signal — it did not simply memorize the majority class
either (predictions did vary), it just didn't have enough negative
examples to find a boundary that generalizes at all.

## Class-Imbalance Limitation

With only **8 negative (incorrect) training examples**, there is
essentially no way for any model — this one or a more sophisticated one —
to learn what "incorrect" looks like in embedding space from this dataset.
8 examples is too few to characterize a class, whatever the modeling
approach. This is consistent with (and now empirically confirms) the
caution already raised when this fine-tuning data was first prepared: the
79/49 = 80% positive class dominance was flagged as a bigger problem than
the raw example count, and this experiment is the concrete evidence that
the flag was correct, not just cautious.

## Not Adopted

This classifier is **not integrated into `app.py` or the production
pipeline**. It exists only as this documented experiment. The existing
LLM-as-judge grading (`judge_correctness` / `judge_graceful_decline` in
`stage4_build_labels.py`) remains the only correctness signal used
anywhere in this project.

## Future Work

The identified fix is **targeted collection of more incorrect-answer
examples**, not a better model or more training data of the same kind
already collected. Specifically:
- Deliberately construct or curate queries expected to produce incorrect
  answers (ambiguous phrasing, questions bordering on the multi-hop and
  cross-company limitations documented in `README_module2.md`, edge cases
  in company-name detection) rather than relying on incorrect answers to
  occur incidentally within a general eval set.
- Target a less imbalanced ratio — even 30–40% negative examples would be
  a large improvement over the current 20%, and would need to come from
  deliberate curation, since the pipeline's own accuracy (39/49 ≈ 80%
  correct) means passively growing the eval set will keep reproducing
  roughly this same imbalance.
- Re-run this same experiment (unchanged methodology) once a better-balanced
  dataset exists, as a cheap way to check whether the imbalance was really
  the binding constraint before investing in a more sophisticated modeling
  approach.
