# TakeMeter — Project Spec (planning.md)

A text classifier that labels the *kind of contribution* a comment makes in a
political discussion thread. This document is the spec written before the
training data was finalized and before any modeling code was run.

---

## 1. Community

**Community:** [r/PoliticalDiscussion](https://www.reddit.com/r/PoliticalDiscussion/), a
subreddit for sustained, moderated political debate (as opposed to r/politics, which
skews toward news-reaction). The seed thread for this dataset is
*"Is the idea of a 'president for all Americans' basically dead?"*

**Why this community is a good fit for a classification task:** the subreddit's rules
explicitly reward substantive argument and discourage low-effort reactions, so a single
thread contains a *wide range* of contribution quality — long evidence-backed arguments,
quick opinionated replies, pure emotional reactions, and clarifying questions all sit
side by side. That variance is exactly what makes the classification boundary meaningful:
in a community where everyone wrote the same way, there would be nothing to classify.
The distinction between "made an argument" and "reacted" is something regulars and
moderators here actually care about and enforce.

I chose this thread because it was something interesting to me, and it had 200+ comments, and reading the comments allowed me to quickly identify labels. 

---

## 2. Labels

Four mutually exclusive labels capture *what a comment is doing in the conversation*.
The ordering is roughly a ladder of argumentative substance: `analysis` >
`op_response` > `response` > `question` (a question contributes the least new argument).

| label           | id | one-sentence definition                                                                           |
|-----------------|----|---------------------------------------------------------------------------------------------------|
| **analysis**    | 0  | The post makes a structured argument backed by statistics, historical comparison, tactical observation, or facts — evidence is specific and verifiable.                                                               |
| **op_response** | 1  | "a response given with an opinion, with supporting evidence."                                     |
| **response**    | 2  | An immediate reaction to a previous post with little to no argument — expressing a feeling in the moment, name-calling (of a public figure or another participant), or anecdotal evidence.                                   |
| **question**    | 3  | The post asks a question without contributing anything, or very little  to the discussion.         |

### Examples (drawn from the dataset)

**analysis**
- *"They were massively mainstream, occurring in all 50 states with a total of around 8 million in-person participants. I am not referring to No Kings. This was probably before your time."* (specific, checkable claims)
- *"The only real way to be a real president is to be elected. Anything other than that is not properly executing the powers of the office of the president."* (structured claim grounded in how the office works)

**op_response**
- *"Congress is broken because the filibuster prevents both parties from being able to pass legislation. The first step to fixing it is to abolish the filibuster so that legislators can do the job they were elected to do."* (opinion + a reason)
- *"Except EO is not legislation, hence why it's vulnerable to being overturned."* (a claim with a supporting because-clause)

**response**
- *"True, thanks for clarifying. EOs are already fragile/temporary by design, yeah."* (agreement, no new argument)
- *"This is an interesting perspective I agree with and didn't even consider before I made this post."* (in-the-moment reaction)

**question**
- *"What media sources would you say are left or right?"*
- *"What evidence suggests Biden did not govern for all Americans?"*

---

## 3. Hard edge cases

These are the boundaries where two annotators could reasonably disagree. Each has an
explicit decision rule so labeling stays consistent across 200 examples.

**Edge case A — `analysis` vs. `op_response` (is the evidence load-bearing or decorative?)**
> *"Trump only received votes from about 25% of Americans eligible to vote."*

Cites a real statistic, which looks like `analysis`. **Decision rule:** if you remove the
opinion framing and the remaining evidence would *still support the claim on its own*,
label `analysis`. If the stat is a single cherry-picked number used to score a point
rather than build a reasoned case, label `op_response`. A lone stat with an implied
"...therefore I'm right" → `op_response`.

**Edge case B — `op_response` vs. `response` (is there reasoning, or just a feeling?)**
> *"Well, Congress's powers have slowly eroded as presidential power has expanded."*

Reads like a casual reply but contains an actual causal claim. **Decision rule:** if the
comment offers *any* reason, mechanism, or "because," it clears the bar for `op_response`.
If it only signals agreement/disagreement or emotion ("True," "lol no," "this is insane")
with no reason attached, it is a `response`.

**Edge case C — `question` vs. `response` (genuine question or rhetorical jab?)**
> *"You mean like you did with your comment?"*

Ends in a question mark but is really name-calling/snark. **Decision rule:** label
`question` only when the post is *seeking information or a real answer*. A rhetorical question used to attack or react is a `response` (the taxonomy explicitly files
name-calling under `response`).

---

## 4. Data collection plan

- **Source:** public comments from the single r/PoliticalDiscussion thread named above,
  painstakenly copied line-by-line to `Reddit-thread-on-American-presidency.csv` (`Text,Label` columns). This took hours and hours, and once I saved it and reopened, all of my data was distored, with data loss and I had to once again, go through them line by line to remove commas because I believed that this was what was causing the .csv problems. Moral of the story: ATTENTION HUMAN READERS! Please tell your students about AI tools that help them extract reddit posts to a CSV file, clean the data so it can be used. We are not all so fortunate enough to know that processors exist for this purpose. I've learned my lesson, though. In the future, I will research tools to do the drudge work  for me. 
  https://www.reddit.com/r/PoliticalDiscussion/comments/1tsemnh/is_the_idea_of_a_president_for_all_americans/
- **Target volume:** ≥ 200 labeled comments. **Current status: 220 labeled** (count verified by `preprocess.py`, which parses the raw export and reports the distribution). However, with my HUMAN eyes, I see 227 rows of data. So despite the hours and hours I spent deleting commas in the .csv, there was apparently enough noisy characters that the machine was not able to read them all correctly.  
- **Per-label target:** there is no fixed quota per label because the goal is to reflect
  the thread's *natural* distribution. Observed distribution so far:

  | label       |count | share |
  |-------------|------|-------|
  | op_response | 101  | ~46%  |
  | response    | 77   | ~35%  |
  | analysis    | 25   | ~11%  |
  | question    | 17   | ~8%   |

 **If a label is underrepresented after 200 examples** (which `analysis` and `question` already are): Keep the natural distribution but **report per-class metrics and use a class-weighted/macro evaluation** so the rare classes aren't hidden by accuracy on the wo majority classes. I will *not* invent or synthesize examples to pad rare classes.

**Splits:** stratified train / validation / test so every label appears
in every split. Stratification matters specifically because `analysis` and `question` are small — a random split could leave them absent from the test set. This split is produced by `preprocess.py` (see the README for usage); current output is train 153 / val 35 / test 35. (Claude did this without me asking it too.)

---

## 5. Evaluation metrics

Accuracy alone is misleading here because the classes are imbalanced: a model that
predicts `op_response` for everything would score ~47% while learning nothing. So:

- **Overall accuracy** — headline number, reported but not trusted alone.
- **Per-class precision, recall, and F1** — the real signal, especially for the rare
  `analysis` and `question` classes. Recall on `analysis` answers the question I actually
  care about: *does the model find the substantive posts?*
- **Macro-F1** — averages F1 across classes equally, so the two rare classes count as
  much as the two common ones. This is the primary single metric for model comparison.
- **Confusion matrix** — to see *which* pairs get confused. I expect the
  `op_response`↔`response` and `analysis`↔`op_response` boundaries (the hard edge cases
  above) to be where errors concentrate; the confusion matrix will confirms or refutes that.

---

## 6. Definition of success

- **Minimum bar:** the fine-tuned model should beat the zero-shot Groq
  baseline on **macro-F1** on the held-out test set, and posts non-trivial recall
  (> 0.4) on the rare `analysis` class rather than ignoring it.
- **"Good enough for a real community tool":** macro-F1 ≥ 0.70 with no single class
  below 0.50 F1. Concretely, that means a moderator-assist tool that surfaces likely
  `analysis` posts would be right clearly more often than wrong and would rarely miss the
  substantive contributions — useful as a triage aid, not as an autonomous moderator.
- **Honesty check:** because `analysis`/`question` are small, a high *accuracy* with a
  low *macro-F1* counts as **failure**, not success. The success criterion is deliberately
  written against the metric the imbalance would otherwise let me cheat on.

---

## 7. AI Tool Plan (Colab)/ Model 1

There is was no application code to generate in this project, because here we were supposed to start using the notebook. However, Claude kept asking me if I wanted to build out an application to clean the data, preprocess the data, train the data, create data sets, and so forth, so evenentually we built out a stretch feature. I did run through the colab notebook twice, and compared my 2 prompts (4 trials in all, which are discussed in the README.md file).

But the actual AI tool plan was given to us by running through the Colab notebook. I did not like this at all. It was bossy and did not show the code that it was running, even when I looked for it. For example, when I wanted dependency versions to compare what tools Claude and I used vs what was provided in the notebook, I could not find this information in the output files. 

However, with my human eyes, I see 228 rows of data. So despite the hours I spent copying the comments line by line from Reddit to the .csv file, redoing it again when the file saved and inserted random characters and causing data loss in my rows of data, and my deleting commas in the .csv, there was apparently still enough noisy characters that the machine was not able to read them all correctly.  


---Before you start - make sure you are using T4 GPU at runtime--
1. Install dependencies 
2. Add imports
------------------Section 1. Load your dataset---------------------
3. Upload your dataset and define. your label map.
4. Clean the label column
5. Create a temporary LABEL_MAP that maps directly to integers (0, 1, 2, 3)
The global LABEL_MAP has values as (int, description) tuples, so we extract the int.
6. Validate all labels are in LABEL_MAP. Use the cleaned labels for validation against the keys of local_label_map_in
---------------Section 2. Prepare Data for Training----------------
7. Splits your data into train / vlaidation / test sets and tokenizes the rest.
---------------Section 5. Baseline Classifier (Groq)--------------- 
8. Install Groq client. 
9. Write your classification prompt
10. Run baseline on testset
11. Extract integer IDs from LABEL_MAP for predictions
-------------------Section 3. Fine-Tune Your Model-------------------
12. Loads distilebet-base-uncased with a classificstion head and fine-tunes it on your training data using T4 GPU at runtime. 
---------Section 4. Evaluate Fine-Tuned Model on Test Set----------------
13. Runs inference on your locked test set and generates metrics and a confusion matrix.  
---------------Section 6. Compare Results and Export-------------------
14. Side-by-side comparison of both models. 
15. Download the output files and commit them to your GitHub repo.

### AI Tool Plan (Claude & me) / Model 2 

AI tooling is used at the three points where it genuinely helps a classification project: 

**Label stress-testing.** Before finalizing the four definitions, feed the label definitions + the three edge cases above to an LLM and ask it to generate 8–10 comments that sit *on* the boundaries. Any generated comment I can't classify cleanly means a definition is too loose — tighten it before annotating. (This document's edge-case rules are the output of that exercise.)

**Annotation assistance.** Decision: the human labels are authoritative. An LLM *may* be used to pre-label a batch as a first pass, but every pre-labeled example is reviewed and corrected by hand before it enters the dataset. Any pre-labeled batch will be tracked (e.g., a `prelabeled` flag/column) and disclosed in the README's AI-usage section. If
this turns out not to be used, that decision is recorded too.

**Failure analysis.** After evaluation, hand the list of misclassified test examples to an LLM and ask it to cluster them into failure patterns (e.g., "stat-citing op_responses misread as analysis"). Every proposed pattern is verified by re-reading the actual examples before it goes in the evaluation write-up — the LLM proposes, I confirm.

---

## Open items before modeling


1. ~~Clean the CSV~~ and ~~build the stratified split~~ — **done** via `preprocess.py`,
   which normalizes the encoding artifacts (`Ò Ó Õ`) and stray tabs, maps labels to ids,
   and writes `data/train.csv`, `data/val.csv`, `data/test.csv`. Re-run it after adding any new comments.  [DONE]
2. Write the fine-tuning script (reads `data/train.csv` + `data/val.csv`) and the zero-shot Groq baseline (reads `data/test.csv`). [DONE]
