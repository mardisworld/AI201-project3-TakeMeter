<!-- TEMP FILE — copy the blocks you want into README.md, then delete this file. -->

### Headline metrics

| metric   | Fine-tuned DistilBERT (v1) | Zero-shot llama-3.3-70b (v1) | Fine-tuned DistilBERT (v2 – Wikipedia) | Zero-shot llama-3.3-70b (v2 – Wikipedia) |
|----------|----------------------------|------------------------------|----------------------------------------|------------------------------------------|
| Accuracy | 0.471                      | 0.618                        | 0.486                                  | 0.486                                    |
| Macro-F1 | 0.258                      | 0.56                         | 0.164                                  | 0.27                                     |

These are the NOTEBOOK runs (macro-F1 values supplied from the notebook's printed
per-class report). A separate local re-run on the current 34-row test set scores
differently (baseline 0.676 / macro-F1 0.700; fine-tuned Run C 0.441 / 0.417) because
the local corpus is 220 comments vs the notebook's 227 and the split differs — see
"Why the local results differ from the notebook" below. Local baseline per-row
predictions are saved in data/baseline_predictions.csv.

---

=== Evaluation 1 (original prompt) — fine-tuned distilbert-base-uncased ===
scored rows : 34
accuracy    : 0.471
macro-F1    : 0.258

label          prec  recall     f1  support
analysis      0.000   0.000  0.000        4
op_response   0.500   0.500  0.500       16
response      0.444   0.667  0.533       12
question      0.000   0.000  0.000        2

|                 | analysis | op_response | response | question |
|-----------------|----------|-------------|----------|----------|
| **analysis**    |    0     |      4      |    0     |    0     |
| **op_response** |    0     |      8      |    8     |    0     |
| **response**    |    0     |      4      |    8     |    0     |
| **question**    |    0     |      0      |    2     |    0     |

Confusion matrix (rows = true, cols = predicted):
                analysis  op_response    response    question
analysis               0           4           0           0
op_response            0           8           8           0
response               0           4           8           0
question               0           0           2           0

Zero-shot baseline (Groq llama-3.3-70b), original prompt:
accuracy : 0.618   (per-class metrics not available — accuracy only in JSON)
improvement (finetuned − baseline) : -0.147

---

=== Evaluation 2 (Wikipedia prompt) — fine-tuned distilbert-base-uncased ===
scored rows : 35
accuracy    : 0.486
macro-F1    : 0.164

label          prec  recall     f1  support
analysis      0.000   0.000  0.000        4
op_response   0.486   1.000  0.654       17
response      0.000   0.000  0.000       12
question      0.000   0.000  0.000        2

|                 | analysis | op_response | response | question |
|-----------------|----------|-------------|----------|----------|
| **analysis**    |    0     |      4      |    0     |    0     |
| **op_response** |    0     |     17      |    0     |    0     |
| **response**    |    0     |     12      |    0     |    0     |
| **question**    |    0     |      2      |    0     |    0     |

Confusion matrix (rows = true, cols = predicted):
                analysis  op_response    response    question
analysis               0           4           0           0
op_response            0          17           0           0
response               0          12           0           0
question               0           2           0           0

Zero-shot baseline (Groq llama-3.3-70b), Wikipedia prompt:
accuracy : 0.486   (per-class metrics not available — accuracy only in JSON)
improvement (finetuned − baseline) : 0.000

---

### Head-to-head (same 34-row test set, original prompt) — Run C vs. baseline

| metric   | Fine-tuned DistilBERT (Run C) | Zero-shot llama-3.3-70b |
|----------|-------------------------------|-------------------------|
| Accuracy | 0.441                         | 0.676                   |
| Macro-F1 | 0.417                         | 0.700                   |

=== Zero-shot baseline: llama-3.3-70b-versatile ===
scored rows : 34
accuracy    : 0.676
macro-F1    : 0.700

label          prec  recall     f1  support
analysis      1.000   0.500  0.667        4
op_response   0.643   0.600  0.621       15
response      0.625   0.833  0.714       12
question      1.000   0.667  0.800        3

|                 | analysis | op_response | response | question |
|-----------------|----------|-------------|----------|----------|
| **analysis**    |    2     |      2      |    0     |    0     |
| **op_response** |    0     |      9      |    6     |    0     |
| **response**    |    0     |      2      |   10     |    0     |
| **question**    |    0     |      1      |    0     |    2     |

Confusion matrix (rows = true, cols = predicted):
                analysis  op_response    response    question
analysis               2           2           0           0
op_response            0           9           6           0
response               0           2          10           0
question               0           1           0           2

Per-row predictions saved to: data/baseline_predictions.csv

---

### Zero-shot v1 (original) vs. zero-shot v2 (Wikipedia rule) — SAME 34-row test set

Both runs use the identical current data/test.csv (34 rows). Only the `analysis`
definition differs (v2 requires Wikipedia-verifiable evidence).

| metric         | v1 (original) | v2 (Wikipedia) | Δ      |
|----------------|---------------|----------------|--------|
| Accuracy       | 0.676         | 0.706          | +0.030 |
| Macro-F1       | 0.700         | 0.717          | +0.017 |
| analysis F1    | 0.667         | 0.667          | 0.000  |
| op_response F1 | 0.621         | 0.643          | +0.022 |
| response F1    | 0.714         | 0.759          | +0.045 |
| question F1    | 0.800         | 0.800          | 0.000  |

=== Zero-shot baseline: llama-3.3-70b-versatile (v2 Wikipedia prompt) ===
scored rows : 34
accuracy    : 0.706
macro-F1    : 0.717

label          prec  recall     f1  support
analysis      1.000   0.500  0.667        4
op_response   0.692   0.600  0.643       15
response      0.647   0.917  0.759       12
question      1.000   0.667  0.800        3

|                 | analysis | op_response | response | question |
|-----------------|----------|-------------|----------|----------|
| **analysis**    |    2     |      2      |    0     |    0     |
| **op_response** |    0     |      9      |    6     |    0     |
| **response**    |    0     |      1      |   11     |    0     |
| **question**    |    0     |      1      |    0     |    2     |

Confusion matrix (rows = true, cols = predicted):
                analysis  op_response    response    question
analysis               2           2           0           0
op_response            0           9           6           0
response               0           1          11           0
question               0           1           0           2

Per-row predictions: data/baseline_predictions_v2.csv

Takeaways:
- On a fixed test set the two zero-shot approaches are nearly identical: the
  Wikipedia rule flipped exactly ONE of 34 predictions (a `response` that v1 read
  as `op_response` is correctly kept as `response` in v2). That single flip is the
  entire +0.03 accuracy / +0.017 macro-F1 gain.
- The Wikipedia rule had ZERO effect on the `analysis` class (its intended target):
  the `analysis` confusion row is identical in both runs (2 correct, 2 → op_response).
- This CONTRADICTS the stored JSONs (baseline v2 acc 0.486 < v1 0.618). That drop
  was a test-split artifact: the notebook's v1 and v2 runs used different random
  splits (34- vs 35-row test sets), and with rare classes of 2-4 examples the metrics
  are noisy (e.g. `question` swings from F1 1.00 to 0.00 on its 2 examples). The gold
  labels were NOT changed (dataset is identical, 227 examples, analysis 25 in both).
  Holding the test set fixed, the prompt change is a wash.
- Both zero-shot runs clear the planning.md §6 "good enough" bar (macro-F1 >= 0.70,
  no class < 0.50 F1, analysis recall > 0.4) and both beat the fine-tuned model
  (macro-F1 0.417) decisively.

---

### Reading of the two evaluations

- **Macro-F1, not accuracy, is the honest metric here, and the two disagree.**
  Evaluation 2 has the *higher* accuracy (0.486 vs 0.471) but the *lower* macro-F1
  (0.164 vs 0.258) — because it collapses to a constant classifier that predicts
  `op_response` for every comment. By the success criterion in planning.md §6
  ("high accuracy with low macro-F1 counts as failure"), Evaluation 2 is the worse
  run despite the better headline number.

- **The Wikipedia requirement hurt the FINE-TUNED model but was a wash for the
  zero-shot model.** Fine-tuned macro-F1 fell 0.258 → 0.164 (it collapsed to a
  constant `op_response` classifier). The stored baseline drop (0.618 → 0.486) looked
  like the prompt hurting too, but a controlled re-run on a FIXED test set shows the
  Wikipedia rule changed only 1 of 34 zero-shot predictions (macro-F1 0.700 → 0.717,
  essentially unchanged). So the stored baseline drop was a test-set/labeling
  artifact, not the prompt — see the zero-shot v1-vs-v2 section above.

- **Errors concentrate exactly on the designed edge cases** — `analysis`↔`op_response`
  and `op_response`↔`response`. With only 17 `analysis` and 11 `question` training
  examples, the model learned the class prior rather than the substance distinction
  (argument vs. reaction). `analysis` recall is 0.000 in both evaluations.

- **No run meets the success bar** (macro-F1 ≥ 0.70, no class < 0.50 F1, analysis
  recall > 0.4), and the fine-tuned model does **not** beat the zero-shot baseline on
  accuracy in either evaluation. This is a legitimate, reportable negative result and
  is consistent with the small, heavily imbalanced dataset.


The headline result: the zero-shot Groq baseline (acc 0.676, macro-F1 0.700) decisively beats the fine-tuned DistilBERT (acc 0.441, macro-F1 0.417) and actually clears your "good enough" success bar — while the fine-tuned model fails it. That's the central, honest finding of the project.

---

## Error analysis (fine-tuned DistilBERT)

Based on the fine-tuned model's per-row predictions in `data/finetuned_predictions.csv`
(the saved ./model run on the current 34-row test set): **19 of 34 wrong (acc 0.441)**.
Errors by direction:

| true → predicted        | count |
|-------------------------|-------|
| op_response → response  | 8     |
| analysis → op_response  | 3     |
| response → op_response  | 3     |
| op_response → analysis  | 2     |
| question → response     | 2     |
| response → analysis     | 1     |

The errors land **exactly on the two boundaries planning.md §3 predicted** —
`analysis`↔`op_response` (edge case A) and `op_response`↔`response` (edge case B).

### Three errors, analyzed

**1. `analysis` read as `op_response` — edge case A (missed load-bearing evidence).**
> *"This bothsidesism is factually incorrect... Fox News settled with Dominion Voting
> Systems for $788M in 2023. Fox hosts and executives privately knew the stolen election
> claims were false and continued to broadcast them anyway... It's documented in the
> court record."*
> **true: analysis · predicted: op_response.**
This is a textbook `analysis` post: a specific, verifiable figure ($788M) and a court
record that supports the claim on its own. The model read it as just another opinion. It
did not learn to recognize evidence that is *load-bearing* — the exact distinction the
label was designed around.

**2. `op_response` read as `response` — edge case B (missed the reasoning).**
> *"...this constant false equivalence through disregarding context and reason. Yeah if
> you look past the surface it is very different. The reasons why people do things is
> important."*
> **true: op_response · predicted: response.**
The comment offers a reason ("the reasons why people do things is important" → context
changes the judgment), which clears the `op_response` bar. The model collapsed it to a
bare reaction. This is the single largest error bucket (8 of 19), i.e. the model
systematically under-credits reasoning and defaults opinion-with-a-reason down to
`response`.

**3. `question` read as `response` — edge case C (genuine question missed).**
> *"Which one 2008 2012 or 2016?"*
> **true: question · predicted: response.**
A genuine clarifying question seeking information — the model filed it as a reaction.
With only 11 `question` training examples, the model never built a reliable signal for
the class (question F1 = 0.50, recall 0.33).

### A revealing over-prediction: surface proxies, not substance

> *"...Joe Biden's Bipartisan Infrastructure Framework gave 61% of the funding to red
> states just an example."*
> **true: op_response · predicted: analysis.**
By edge rule A, a lone cherry-picked statistic used to score a point is `op_response`,
not `analysis`. The model predicted `analysis` here — apparently because a number is
present. Of the 4 times it predicted `analysis`, 3 were wrong, and the cases it chose
tended to contain a numeric token. So the model latched onto **"has a statistic" as a
proxy for `analysis`**, while missing genuinely evidence-backed arguments (error #1
above) — the opposite of the substance distinction intended.

### What the model learned vs. what I intended

I designed the labels around *substance* — argument vs. opinion-with-reason vs. bare
reaction. The model instead learned **surface proxies and the class prior**:
- It defaults to the two majority classes (`op_response` 46% + `response` 35% = 81% of
  the data); 14 of 19 errors are collapses into `op_response`/`response`.
- It uses "presence of a number" as a weak `analysis` cue rather than judging whether
  evidence is load-bearing.
- It under-credits reasoning, pushing `op_response` down to `response` (the biggest
  error bucket).
This is consistent with the macro-F1 (0.417) sitting below the baseline's (0.700): the
model's competence is concentrated in the common classes, and the rare classes it was
supposed to surface (`analysis` recall 0.25, `question` recall 0.33) are weak. With only
~150 training examples and 17 `analysis` / 11 `question`, there was too little signal to
learn the boundary rather than the prior.

### Did it meet the success criteria?

**No.** Success bar (planning.md §6): beat the zero-shot baseline on macro-F1; macro-F1
≥ 0.70 with no class below 0.50 F1; `analysis` recall > 0.4.
- Beat baseline on macro-F1? **No** — 0.417 vs 0.700 (baseline wins decisively).
- Macro-F1 ≥ 0.70? **No** — 0.417.
- No class below 0.50 F1? **No** — `analysis` F1 = 0.25.
- `analysis` recall > 0.4? **No** — 0.25.

The fine-tuned model fails every criterion, and even the *minimum* bar (beat the
baseline on macro-F1) is not met. The honest project finding is the inverse of the
hypothesis: on this small, imbalanced dataset the **zero-shot LLM is the better
classifier** and already clears the "good enough for a real community tool" bar, while
fine-tuning DistilBERT mostly learned the label distribution.

---

## Why the local results differ from the notebook (v1)

Two distinct sets of numbers exist for this project: the **Colab notebook** results
(returned as the PNGs + JSON + printed stats) and a **local re-run** (the current
`./model/`, `finetuned_predictions.csv`, and fresh `baseline.py` runs, all timestamped
Jun 26). They differ — and the difference is in the *data*, not just the model.

### Notebook v1 (the authoritative notebook run)

```
==================================================
RESULTS COMPARISON
==================================================
Model                               Accuracy
---------------------------------------------
Zero-shot baseline (Groq)              0.618
Fine-tuned DistilBERT                  0.471
---------------------------------------------
Fine-tuning regression: 0.147
```

Per-class metrics (from the notebook):

| run | accuracy | macro-F1 |
|-----|----------|----------|
| Zero-shot baseline (Groq)   | 0.618 | 0.56 |
| Fine-tuned DistilBERT       | 0.471 | 0.26 |

Notebook baseline per-class: analysis F1 0.00 (P0.00 R0.00), op_response F1 0.52
(P0.64 R0.44), response F1 0.73 (P0.57 R1.00), question F1 1.00 (P1.00 R1.00).
Notebook fine-tuned per-class: analysis F1 0.00, op_response F1 0.50, response F1 0.53,
question F1 0.00 — i.e. it never predicts the two rare classes. The fine-tuned macro-F1
of 0.26 matches confusion_matrix.png exactly, confirming that PNG is the notebook's v1
fine-tuned run.

### Cause 1 — the datasets are different sizes

| label       | Notebook v1 | Local clean.csv | diff |
|-------------|-------------|-----------------|------|
| analysis    | 25          | 25              | —    |
| op_response | 108         | 101             | -7   |
| response    | 77          | 77              | —    |
| question    | 17          | 17              | —    |
| **total**   | **227**     | **220**         | **-7** |

Exactly **7 `op_response` comments were dropped** locally; every other class is
identical. The notebook's original input CSV is no longer in the project (only the
`-v2` export remains), so the specific 7 can't be diffed, but the aggregate is
unambiguous: the two pipelines are not scoring the same corpus.

### Cause 2 — the test splits differ even at the same ~15% ratio

- Notebook test: op_response 16 / response 12 / analysis 4 / question **2** (=34)
- Local test:    op_response 15 / response 12 / analysis 4 / question **3** (=34)

`preprocess.py` computes `round(17 * 0.15) = 3` question examples for test, but the
notebook produced 2 — so the notebook used a different split method or seed. Different
comments land in the test set, which moves the metrics independently of any model change.
(This is also why the fresh local baseline scores 0.676 vs the notebook's 0.618: same
prompt, different test comments.)

### Not a bug — the model load report

The notebook's load report (`pre_classifier`/`classifier` MISSING, `vocab_*` UNEXPECTED)
is normal. Loading `distilbert-base-uncased` for sequence classification always discards
the pretraining masked-LM head (UNEXPECTED) and randomly initializes a fresh
classification head (MISSING). Expected behavior, not an error.

### Training curve (notebook v1 fine-tuning)

| epoch | training loss | validation loss | accuracy |
|-------|---------------|-----------------|----------|
| 1     | 1.401         | 1.393           | 0.294    |
| 2     | 1.367         | 1.332           | 0.559    |
| 3     | 1.274         | 1.231           | 0.529    |

Loss barely moves (1.40 → 1.27 over 3 epochs) and validation accuracy peaks at epoch 2
then drops — consistent with a model that learned the class prior rather than the
substance distinction on ~150 imbalanced examples.

## Why the local results differ from the notebook (v2 — Wikipedia prompt)

```
==================================================
RESULTS COMPARISON
==================================================
Model                               Accuracy
---------------------------------------------
Zero-shot baseline (Groq)              0.486
Fine-tuned DistilBERT                  0.486
---------------------------------------------
Fine-tuning improvement: 0.000
```

| run | accuracy | macro-F1 |
|-----|----------|----------|
| Zero-shot baseline (Groq)   | 0.486 | 0.27 |
| Fine-tuned DistilBERT       | 0.486 | 0.16 |

Notebook v2 baseline per-class: analysis F1 0.00, op_response F1 0.48 (P0.50 R0.47),
response F1 0.58 (P0.47 R0.75), question F1 0.00.
Notebook v2 fine-tuned per-class: analysis F1 0.00, op_response F1 0.65 (P0.49 R1.00),
response F1 0.00, question F1 0.00 — i.e. it predicts `op_response` for everything (a
constant classifier). macro-F1 0.16 matches confusion_matrix-v2.png exactly, confirming
that PNG is the notebook's v2 fine-tuned run.

### Key point — the v2 experiment did NOT change the dataset

The notebook v2 corpus is **identical to v1**: 227 examples, distribution op_response 108
/ response 77 / analysis 25 / question 17. The `analysis` count is unchanged (25 in both),
so the Wikipedia rule was applied **only as a prompt/definition shown to the models** — the
gold labels were NOT re-annotated. (This corrects an earlier guess that gold labels were
re-cut.)

### What actually changed between notebook v1 and v2

1. **A different random split.** v1 was train 158 / val 35 / test 34; v2 is train 158 /
   val 34 / test 35. Same training set, but val/test sizes swap and the test composition
   changes (v1 test: op16/resp12/ana4/q2; v2 test: op17/resp12/ana4/q2).
2. **The Wikipedia rule in the prompt** (the `analysis` definition).

### The baseline "drop" (0.618 → 0.486) is split noise, not the prompt

It looks like the Wikipedia rule tanked the baseline (0.618 → 0.486, macro-F1 0.56 →
0.27). It did not. The two notebook runs used different test sets, and with rare classes
of 2-4 examples the metrics are extremely noisy — e.g. `question` (2 examples) swings from
F1 1.00 (v1) to 0.00 (v2), which alone drags the macro-F1 down. The controlled experiment
above (zero-shot v1 vs v2 on a FIXED 34-row test set) isolates the prompt effect and shows
it changes only 1 of 34 predictions (macro-F1 0.700 → 0.717). So the prompt is a wash; the
notebook's apparent drop is an artifact of comparing across different splits.

### v2 fine-tuning collapsed completely

| epoch | training loss | validation loss | accuracy |
|-------|---------------|-----------------|----------|
| 1     | 1.363         | 1.352           | 0.471    |
| 2     | 1.330         | 1.310           | 0.471    |
| 3     | 1.271         | 1.224           | 0.471    |

Validation accuracy is frozen at 0.471 across all three epochs — the model never moves off
predicting the majority class. This is more degenerate than v1 (which at least reached
0.559 at epoch 2), and it produces the all-`op_response` confusion matrix.

### Summary table — all four notebook runs vs. the local re-run

| run | accuracy | macro-F1 | notes |
|-----|----------|----------|-------|
| Notebook v1 baseline      | 0.618 | 0.56 | 34-row test split |
| Notebook v1 fine-tuned    | 0.471 | 0.26 | = confusion_matrix.png |
| Notebook v2 baseline      | 0.486 | 0.27 | 35-row test split (split noise, not prompt) |
| Notebook v2 fine-tuned    | 0.486 | 0.16 | = confusion_matrix-v2.png (constant classifier) |
| Local baseline (orig prompt) | 0.676 | 0.700 | 34-row local split, 220-comment corpus |
| Local baseline (wiki prompt) | 0.706 | 0.717 | same local split |
| Local fine-tuned (Run C)  | 0.441 | 0.417 | current ./model, predicts all 4 classes |

The local corpus is 220 (notebook 227, −7 op_response) and the local split differs, so
local numbers are not directly comparable to the notebook — they are a separate
reproduction, consistent in its qualitative conclusion (baseline > fine-tuned).
