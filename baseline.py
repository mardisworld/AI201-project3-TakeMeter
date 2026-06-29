"""
baseline.py — zero-shot baseline: classify the test set with Groq's
llama-3.3-70b-versatile (no task-specific training) and score it.

This is the comparison point for the fine-tuned DistilBERT model. It uses the exact
same label definitions as training, via the prompts in prompts.py.

Setup:
    pip install groq
    export GROQ_API_KEY=...        # get one at https://console.groq.com

Usage:
    python baseline.py
    python baseline.py --test data/test.csv --model llama-3.3-70b-versatile
    python baseline.py --limit 10            # smoke-test on the first 10 rows

Outputs:
    - prints overall accuracy, macro-F1, per-class precision/recall/F1, confusion matrix
    - writes data/baseline_predictions.csv  (text, true_label, pred_label, correct)
"""

import argparse
import csv
import os
import sys
import time

from labels import LABELS, ID_TO_LABEL  # single source of truth for the label taxonomy
from prompts import classification_system_prompt, classification_user_prompt


def read_test(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"No rows in {path!r}. Run preprocess.py first.")
    return rows


def parse_label(raw):
    """Map a raw model response to one of our label names, tolerantly."""
    if not raw:
        return None
    text = raw.strip().lower().strip(".\"' \n")
    if text in LABELS:
        return text
    # tolerate extra words / punctuation: pick the label that appears in the response
    matches = [lab for lab in LABELS if lab in text]
    # prefer 'op_response' over the substring 'response' if both appear
    if "op_response" in matches:
        return "op_response"
    return matches[0] if matches else None


def classify(client, model, comment, system_prompt, retries=4):
    """One zero-shot classification call with simple exponential backoff."""
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=10,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": classification_user_prompt(comment)},
                ],
            )
            return parse_label(resp.choices[0].message.content)
        except Exception as e:  # rate limits, transient errors
            if attempt == retries - 1:
                print(f"  ! giving up on a row after {retries} tries: {e}", file=sys.stderr)
                return None
            time.sleep(2 ** attempt)


def score(rows):
    """Compute accuracy, per-class P/R/F1, macro-F1, and a confusion matrix."""
    # confusion[true][pred]
    confusion = {t: {p: 0 for p in LABELS} for t in LABELS}
    correct = total = 0
    skipped = 0
    for r in rows:
        true = r["true_label"]
        pred = r["pred_label"]
        if pred not in LABELS:
            skipped += 1
            continue
        total += 1
        confusion[true][pred] += 1
        if true == pred:
            correct += 1

    accuracy = correct / total if total else 0.0

    per_class = {}
    f1s = []
    for lab in LABELS:
        tp = confusion[lab][lab]
        fp = sum(confusion[t][lab] for t in LABELS if t != lab)
        fn = sum(confusion[lab][p] for p in LABELS if p != lab)
        support = sum(confusion[lab].values())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[lab] = (precision, recall, f1, support)
        f1s.append(f1)

    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    return accuracy, macro_f1, per_class, confusion, total, skipped


def print_report(accuracy, macro_f1, per_class, confusion, total, skipped, model):
    print(f"\n=== Zero-shot baseline: {model} ===")
    print(f"scored rows: {total}" + (f"  (skipped {skipped} unparseable)" if skipped else ""))
    print(f"accuracy : {accuracy:.3f}")
    print(f"macro-F1 : {macro_f1:.3f}\n")

    print(f"{'label':<12}{'prec':>7}{'recall':>8}{'f1':>7}{'support':>9}")
    for lab in LABELS:
        p, r, f1, sup = per_class[lab]
        print(f"{lab:<12}{p:>7.3f}{r:>8.3f}{f1:>7.3f}{sup:>9}")

    print("\nConfusion matrix (rows = true, cols = predicted):")
    header = " " * 12 + "".join(f"{lab[:10]:>12}" for lab in LABELS)
    print(header)
    for t in LABELS:
        row = f"{t:<12}" + "".join(f"{confusion[t][p]:>12}" for p in LABELS)
        print(row)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Zero-shot Groq baseline on the test set.")
    ap.add_argument("--test", default=os.path.join(here, "data", "test.csv"))
    ap.add_argument("--model", default="llama-3.3-70b-versatile")
    ap.add_argument("--out", default=os.path.join(here, "data", "baseline_predictions.csv"))
    ap.add_argument("--limit", type=int, default=0, help="classify only the first N rows")
    args = ap.parse_args()

    if not os.environ.get("GROQ_API_KEY"):
        sys.exit("GROQ_API_KEY is not set. Run: export GROQ_API_KEY=...")
    try:
        from groq import Groq
    except ImportError:
        sys.exit("The 'groq' package is not installed. Run: pip install groq")

    rows = read_test(args.test)
    if args.limit:
        rows = rows[: args.limit]

    client = Groq()
    system_prompt = classification_system_prompt()

    results = []
    print(f"Classifying {len(rows)} test comments with {args.model} ...")
    for i, r in enumerate(rows, 1):
        text = r["text"]
        true_label = r.get("label") or ID_TO_LABEL.get(int(r["label_id"]))
        pred = classify(client, args.model, text, system_prompt)
        results.append({
            "text": text,
            "true_label": true_label,
            "pred_label": pred if pred else "UNPARSEABLE",
            "correct": int(pred == true_label),
        })
        if i % 10 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)}")

    # save predictions
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text", "true_label", "pred_label", "correct"])
        w.writeheader()
        w.writerows(results)

    accuracy, macro_f1, per_class, confusion, total, skipped = score(results)
    print_report(accuracy, macro_f1, per_class, confusion, total, skipped, args.model)
    print(f"\nWrote per-row predictions to {args.out}")


if __name__ == "__main__":
    main()
